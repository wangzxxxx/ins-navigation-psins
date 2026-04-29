import numpy as np
import math
from .glv_utils import glv

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

def posset(lat, lon=0.0, hgt=0.0):
    '''
    Geographic position = [latitude; logititude; height] setting.
    '''
    pos0 = np.array([lat, lon, hgt], dtype=float)
    if abs(pos0[0]) > 15959.999:
        isdeg = 3
    elif abs(pos0[0]) > 159.999:
        isdeg = 2
    elif abs(pos0[0]) > math.pi / 2:
        isdeg = 1
    else:
        isdeg = 0
        
    if isdeg == 0:
        return pos0
    elif isdeg == 1:
        return np.array([pos0[0] * glv.deg, pos0[1] * glv.deg, pos0[2]])
    return pos0
