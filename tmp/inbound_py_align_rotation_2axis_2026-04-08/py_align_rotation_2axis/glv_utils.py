import numpy as np
import math

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
        
glv = GLV()
