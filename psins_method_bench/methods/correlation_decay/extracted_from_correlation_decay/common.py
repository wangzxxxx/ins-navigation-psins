"""Shared core extracted from test_calibration_correlation_decay.py"""
import numpy as np
import sys, os, math, re
from dotenv import load_dotenv
from openai import OpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from tmp_psins_py.psins_py.nav_utils import glv, posset, Earth
from tmp_psins_py.psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from tmp_psins_py.psins_py.kf_utils import kfupdate, alignsb, nnts
from tmp_psins_py.psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew

load_dotenv()
try:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
    provider_id = os.environ.get("MODEL_PROVIDER_ID", "azure_openai")
    client = OpenAI(api_key=api_key, base_url=base_url, default_headers={"X-Model-Provider-Id": provider_id}) if api_key and base_url else None
except Exception:
    client = None
    model_name = None

STATE_LABELS_36 = (
    ['phi_x', 'phi_y', 'phi_z'] + ['dv_x', 'dv_y', 'dv_z'] +
    ['eb_x', 'eb_y', 'eb_z'] + ['db_x', 'db_y', 'db_z'] +
    ['Kg_xx', 'Kg_yx', 'Kg_zx', 'Kg_xy', 'Kg_yy', 'Kg_zy', 'Kg_xz', 'Kg_yz', 'Kg_zz'] +
    ['Ka_xx', 'Ka_xy', 'Ka_xz', 'Ka_yy', 'Ka_yz', 'Ka_zz'] +
    ['Ka2_x', 'Ka2_y', 'Ka2_z'] +
    ['rx_x', 'rx_y', 'rx_z'] + ['ry_x', 'ry_y', 'ry_z']
)

def get_llm_scd_params(iteration, prev_stats=None):
    if client is None:
        return {'alpha_sf': 0.98, 'alpha_bias': 1.0, 'transition_duration': 2.0}
    return {'alpha_sf': 0.98, 'alpha_bias': 1.0, 'transition_duration': 2.0}

def imuadderr_full(imu_in, ts, arw=0.0, vrw=0.0, bi_g=0.0, tau_g=3600.0, bi_a=0.0, tau_a=3600.0):
    np.random.seed(42)
    imu = np.copy(imu_in); m = imu.shape[0]; sts = math.sqrt(ts)
    if arw > 0: imu[:, 0:3] += arw * sts * np.random.randn(m, 3)
    if vrw > 0: imu[:, 3:6] += vrw * sts * np.random.randn(m, 3)
    if bi_g > 0 and tau_g > 0:
        c = math.exp(-ts/tau_g); sw = bi_g*math.sqrt(2*ts/tau_g); b = np.zeros(3)
        for k in range(m): b = c*b + sw*np.random.randn(3); imu[k, 0:3] += b*ts
    if bi_a > 0 and tau_a > 0:
        c = math.exp(-ts/tau_a); sw = bi_a*math.sqrt(2*ts/tau_a); b = np.zeros(3)
        for k in range(m): b = c*b + sw*np.random.randn(3); imu[k, 3:6] += b*ts
    return imu

def get_default_clbt():
    Kg = np.eye(3) - np.diag([10.,20.,30.])*glv.ppm + np.array([[0,10,20],[30,0,40],[50,60,0]])*glv.sec
    Ka = np.eye(3) - np.diag([10.,20.,30.])*glv.ppm + np.array([[0,10,20],[0,0,40],[0,0,0]])*glv.sec
    return {'sf':np.ones(6),'Kg':Kg,'Ka':Ka,'eb':np.array([.1,.2,.3])*glv.dph,'db':np.array([100,200,300])*glv.ug,'Ka2':np.array([10,20,30])*glv.ugpg2,'rx':np.array([1,2,3])/100.,'ry':np.array([4,5,6])/100.}

def Ka_from_upper(x):
    d=np.zeros((3,3)); d[0,0]=x[0];d[0,1]=x[1];d[0,2]=x[2];d[1,1]=x[3];d[1,2]=x[4];d[2,2]=x[5]; return d

def clbtkfinit_36(nts):
    n=36; kf={'nts':nts,'n':n,'m':3}
    qv=np.zeros(n); qv[0:3]=0.01*glv.dpsh; qv[3:6]=100*glv.ugpsHz
    kf['Qk']=np.diag(qv)**2*nts; kf['Rk']=np.diag([.001,.001,.001])**2
    pv=np.zeros(n)
    pv[0:3]=np.array([.1,.1,1.])*glv.deg; pv[3:6]=1.; pv[6:9]=.1*glv.dph; pv[9:12]=1.*glv.mg
    pv[12:15]=[100*glv.ppm,100*glv.sec,100*glv.sec]; pv[15:18]=[100*glv.sec,100*glv.ppm,100*glv.sec]; pv[18:21]=[100*glv.sec,100*glv.sec,100*glv.ppm]
    pv[21]=100*glv.ppm;pv[22]=100*glv.sec;pv[23]=100*glv.sec; pv[24]=100*glv.ppm;pv[25]=100*glv.sec;pv[26]=100*glv.ppm; pv[27:30]=100*glv.ugpg2; pv[30:33]=.1; pv[33:36]=.1
    kf['Pxk']=np.diag(pv)**2
    Hk=np.zeros((3,n));Hk[:,3:6]=np.eye(3); kf['Hk']=Hk;kf['xk']=np.zeros(n);kf['I']=np.eye(n)
    return kf

def getFt_36(fb, wb, Cnb, wnie, SS):
    n=36; wX=askew(wnie); fX=askew(Cnb@fb); fx,fy,fz=fb[0],fb[1],fb[2]; wx,wy,wz=wb[0],wb[1],wb[2]; CDf2=Cnb@np.diag(fb**2)
    Ca=np.zeros((3,6)); Ca[:,0]=Cnb[:,0]*fx;Ca[:,1]=Cnb[:,0]*fy;Ca[:,2]=Cnb[:,0]*fz; Ca[:,3]=Cnb[:,1]*fy;Ca[:,4]=Cnb[:,1]*fz;Ca[:,5]=Cnb[:,2]*fz
    Ft=np.zeros((n,n)); Ft[0:3,0:3]=-wX;Ft[0:3,6:9]=-Cnb; Ft[0:3,12:15]=-wx*Cnb;Ft[0:3,15:18]=-wy*Cnb;Ft[0:3,18:21]=-wz*Cnb; Ft[3:6,0:3]=fX;Ft[3:6,9:12]=Cnb;Ft[3:6,21:27]=Ca;Ft[3:6,27:30]=CDf2; Ft[3:6,30:36]=Cnb@SS[:,0:6]
    return Ft

def clbtkffeedback_pruned(kf, clbt):
    xk=kf['xk']; dKg=xk[12:21].reshape(3,3).T; clbt['Kg']=(np.eye(3)-dKg)@clbt['Kg']; dKa=Ka_from_upper(xk[21:27]); clbt['Ka']=(np.eye(3)-dKa)@clbt['Ka']; clbt['Ka2']+=xk[27:30];clbt['eb']+=xk[6:9];clbt['db']+=xk[9:12]; clbt['rx']+=xk[30:33];clbt['ry']+=xk[33:36]; return clbt

def run_calibration(imu1, pos0, ts, scd_mode=False, label=""):
    # Placeholder extracted interface. Real logic should be copied incrementally from source.
    return {'label': label, 'scd_mode': scd_mode, 'status': 'extracted_interface_only'}
