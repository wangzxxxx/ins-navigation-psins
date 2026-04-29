import numpy as np
import math
from .math_utils import askew

class GLV:
    def __init__(self, Re=6378137.0, f=1/298.257, wie=7.2921151467e-5):
        self.Re = Re
        self.f = f
        self.Rp = (1 - self.f) * self.Re
        self.e = math.sqrt(2 * self.f - self.f**2)
        self.e2 = self.e**2
        self.ep = math.sqrt(self.Re**2 - self.Rp**2) / self.Rp
        self.ep2 = self.ep**2
        self.GM = 3.986004418e14
        self.J2 = 1.08262982131e-3
        self.J4 = -2.37091120053e-6
        self.J6 = 6.08346498882e-9
        self.wie = wie
        self.meru = self.wie / 1000
        self.g0 = 9.7803267715
        
        m = self.Re * self.wie**2 / self.g0
        self.beta = 5/2*m - self.f - 17/14*m*self.f
        self.beta1 = (5*m*self.f - self.f**2) / 8
        self.mg = 1.0e-3 * self.g0
        self.ug = 1.0e-6 * self.g0
        self.ugph = self.ug / 3600
        self.mGal = 1.0e-3 * 0.01
        self.uGal = self.mGal / 1000
        self.ugpg = self.ug / self.g0
        self.ugpg2 = self.ug / self.g0**2
        self.ws = 1 / math.sqrt(self.Re / self.g0)
        self.ppm = 1.0e-6
        self.deg = math.pi / 180
        self.min = self.deg / 60
        self.sec = self.min / 60
        self.hur = 3600
        self.dps = math.pi / 180 / 1
        self.dph = self.deg / self.hur
        self.dpss = self.deg / math.sqrt(1)
        self.dpsh = self.deg / math.sqrt(self.hur)
        self.dphpsh = self.dph / math.sqrt(self.hur)
        self.Hz = 1.0
        self.ugpsHz = self.ug / math.sqrt(self.Hz)
        self.I33 = np.eye(3)
        self.o33 = np.zeros((3, 3))
        
        self.cs = np.array([
            [2/3, 0, 0, 0, 0],
            [9/20, 27/20, 0, 0, 0],
            [54/105, 92/105, 214/105, 0, 0],
            [250/504, 525/504, 650/504, 1375/504, 0],
            [2315/4620, 4558/4620, 7296/4620, 7834/4620, 15797/4620]
        ])
        self.csmax = self.cs.shape[0] + 1
        
glv = GLV()

class Earth:
    def __init__(self, pos, vn=None):
        if vn is None:
            vn = np.zeros(3)
        self.pos = np.asarray(pos).flatten()
        self.vn = np.asarray(vn).flatten()
        
        self.sl = math.sin(self.pos[0])
        self.cl = math.cos(self.pos[0])
        self.tl = self.sl / self.cl
        self.sl2 = self.sl**2
        self.sl4 = self.sl2**2
        sq = 1 - glv.e2 * self.sl2
        sq2 = math.sqrt(sq)
        
        self.RMh = glv.Re * (1 - glv.e2) / sq / sq2 + self.pos[2]
        self.RNh = glv.Re / sq2 + self.pos[2]
        self.clRNh = self.cl * self.RNh
        
        self.wnie = np.array([0, glv.wie * self.cl, glv.wie * self.sl])
        vE_RNh = self.vn[0] / self.RNh
        self.wnen = np.array([-self.vn[1] / self.RMh, vE_RNh, vE_RNh * self.tl])
        self.wnin = self.wnie + self.wnen
        self.wnien = self.wnie + self.wnin
        
        gL = glv.g0 * (1 + glv.beta * self.sl2 - glv.beta1 * (2 * self.sl * self.cl)**2)
        hR = self.pos[2] / (glv.Re * (1 - glv.f * self.sl2))
        self.g = gL * (1 - 2 * hR - 5 * hR**2)
        self.gn = np.array([0, 0, -self.g])
        
        self.gcc = np.array([
            self.wnien[2] * self.vn[1] - self.wnien[1] * self.vn[2],
            self.wnien[0] * self.vn[2] - self.wnien[2] * self.vn[0],
            self.wnien[1] * self.vn[0] - self.wnien[0] * self.vn[1] + self.gn[2]
        ])

def posset(lat, lon=0.0, hgt=0.0, isdeg=0):
    '''
    Geographic position = [latitude; logititude; height] setting.
    '''
    pos0 = np.array([lat, lon, hgt])
    if abs(pos0[0]) > 15959.999:
        isdeg = 3
    elif abs(pos0[0]) > 159.999:
        isdeg = 2
    elif abs(pos0[0]) > math.pi / 2:
        isdeg = 1
        
    if isdeg == 0:
        return pos0
    elif isdeg == 1:
        return np.array([pos0[0] * glv.deg, pos0[1] * glv.deg, pos0[2]])
    # Ignoring dms conversions for simplicity as 19pos simulation uses isdeg=0 explicitly since values are already converted to deg.
    return pos0
