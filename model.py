import numpy as np

class system:
    def __init__(self):
        self.M=1 # 1 Msun
        self.Mp=0.001 # 1 Mjupiter
        self.R=1 # 1 Rsun
        self.Rp=0.1 # 1 Rjupiter
        self.a=0.1 # in AU
        self.e=0.01
        self.T=5000 # Tsun
        self.Tp=500
        
        self.D=100 # in kpc
        
        self.Ag=0.2 # geometric albedo
        self.beta=1.03 # stellar response to tides for n=3 polytrope
        
        self.t0=0 # time of periapse
        self.vPhi=0.5*np.pi # azimuthal projection angle (relative to periapse)
        self.vTheta=0.49*np.pi # polar projection angle (approx pi/2 for transiting planet)