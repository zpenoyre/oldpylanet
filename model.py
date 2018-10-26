import numpy as np

import scipy.optimize
import scipy.integrate
Gyr=3.925e8 #in Rsun, Msun and yrs
G=2946 #in Rsun, Msun and days
Rsun=6.96e8 #in m
Msun=2e30 #in kg
yr=3.15e7 #in seconds
day=86400 #in seconds

#Mj=0.000954 #in Msun
#Me=0.000003 #in Msun
#Rj=0.10049  #in Rsun
#Re=0.009158 #in Rsun

#AU=214.94 #in Rsun

#rUnit=1 #planet radii unit in rSun
#mUnit=1 #planet mass unit in mSun
#tUnit=1 #time unit in days
#aUnit=1 #semi-major axis unit in rSun

class system:
    def __init__(self):
        self.M=1 # 1 Msun
        self.Mp=0.001 # 1 Mjupiter
        self.R=1 # 1 Rsun
        self.Rp=0.1 # 1 Rjupiter
        self.a=20 # ~ orbit of mercury in Rsun
        self.e=0.01
        self.T=5000 # Tsun
        self.Tp=500
    
        self.period=1 # in days
        
        self.D=100 # in kpc
        
        self.Ag=0.2 # geometric albedo
        self.beta=1.03 # stellar response to tides for n=3 polytrope
        
        # The reflectivity and tidal response of the companion - any differences negligible in all but most extreme cases
        self.Agp=0.2 # geometric albedo of companion
        self.betap=1.03 # companion response to tides for n=3 polytrope
        
        self.tPeri=0 # time of periapse
        self.vPhi=0.5*np.pi # azimuthal projection angle (relative to periapse)
        self.vTheta=0.49*np.pi # polar projection angle (approx pi/2 for transiting planet)
    
    # A series of functions to find (and set) parameters that can be derived from other known parameters (e.g. a <-> period) 
    # note: could include mass here - though technically will just give M+Mp - leaving out for now   
    def find_a(self):
        period=2*np.pi*np.sqrt(self.a**3/(G*(self.M+self.Mp)))
        self.a=self.a*(period/self.period)**(2/3)
        return self.a
    def find_period(self):
        self.period=2*np.pi*np.sqrt(self.a**3/(G*(self.M+self.Mp)))
        return self.period
    
    # General orbital parameters as a function of time
    # Note: eta is generally a more useful angle than Phi, but Phi is more easily understood (the angle from periapse of the planet relative to c.o.m.)
      
    # Finds the seperation of the planet and star (not the seperation of each from the center of mass, that is rp=d/(1+Mp/M) and r=d-rp)
    def find_d(self,t):
        eta=self.find_eta(t)
        return self.a*(1-self.e*np.cos(eta))
    # Finds Phi at a given time
    def find_Phi(self,t):
        eta=self.find_eta(t)
        return self.find_Phi_eta(eta)
    
    # Functions that convert eta<->Phi
    # If we know Phi and would like to find eta (and subsequently t) [unitless->unitless]
    def find_eta_Phi(self,Phi): 
        return 2*np.arctan(np.sqrt((1-self.e)/(1+self.e))*np.tan(Phi/2)) # note that eta can be outside of range 0->2*pi but this will only give values in that range
    # If we know eta and would like to find Phi
    def find_Phi_eta(self,eta):
        return 2*np.arctan(np.sqrt((1+self.e)/(1-self.e))*np.tan(eta/2)) % (2*np.pi)
        
    # Similarly can find time of center of transit
    # note: at the moment this just gives the time when Phi=vPhi - could also find ingress/egress or check for transit
    def find_tTransit(self):
        eta0=self.find_eta_Phi(self.vPhi)
        return self.find_t(eta0)
        
    # Can be used to find t (when t argument omitted) or solved to find eta (when t supplied) [raw unit->days]
    def find_t(self,eta,t=0):
        return np.sqrt((self.a**3)/(G*(self.M+self.Mp))) * (eta - self.e*np.sin(eta)) - (t-self.tPeri)
    
    # Finds eta for a given time (inverting t=(T/2pi)*(eta - e sin(eta)) )
    # note: currently using approximate solution to O(e^3) from Penoyre and Sandford 2018 - see OoT for exact solution
    def find_eta(self,t): #[raw units->unitless]
        eta0=(t-pl.tp)*tUnit*np.sqrt((G*pl.M) / ((pl.a*aUnit)**3))
            #eta1=pl.e*np.sin(eta0)/(1-pl.e*np.cos(eta0))
        eta1=self.e*np.sin(eta0)
        eta2=(self.e**2)*np.sin(eta0)*np.cos(eta0)
        eta3=(self.e**3)*np.sin(eta0)*(1-(3/2)*(np.sin(eta0)**2))
        thisEta=eta0+eta1+eta2+eta3 #accurate up to and including O(e^3)
        return thisEta
    