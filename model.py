import numpy as np

import scipy.optimize
import scipy.integrate
import astropy.units as u
import astropy
#Gyr=3.925e8 #in Rsun, Msun and yrs
G=astropy.constants.G #2946 #in Rsun, Msun and days
h=astropy.constants.h
k=astropy.constants.k_B
c=astropy.constants.c
#Rsun=6.96e8 #in m
#Msun=2e30 #in kg
#yr=3.15e7 #in seconds
#day=86400 #in seconds

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
        self.M=1*u.Msun # 1 Msun
        self.Mp=0.001*u.Msun # 1 Mjupiter
        self.R=1*u.Rsun # 1 Rsun
        self.Rp=0.1*u.Rsun # 1 Rjupiter
        self.T=5000*u.K # Tsun
        self.Tp=500*u.K
    
        self.a=20*u.Rsun # ~ orbit of mercury in Rsun
        self.e=0.01
        self.period=1*u.d # in days
        
        self.D=100*u.kpc # in kpc
        
        self.Ag=0.2 # geometric albedo
        self.beta=1.03 # stellar response to tides for n=3 polytrope
        
        # The reflectivity and tidal response of the companion - any differences negligible in all but most extreme cases
        self.Agp=0.2 # geometric albedo of companion
        self.betap=1.03 # companion response to tides for n=3 polytrope
        
        self.tPeri=0*u.d # time of periapse
        self.vPhi=0.5*np.pi # azimuthal projection angle (relative to periapse)
        self.vTheta=0.49*np.pi # polar projection angle (approx pi/2 for transiting planet)
    
    # A series of functions to find (and set) parameters that can be derived from other known parameters (e.g. a <-> period) 
    # note: could include mass here - though technically will just give M+Mp - leaving out for now   
    def find_a(self):
        period=2*np.pi*np.sqrt(self.a**3/(G*(self.M+self.Mp)))
        self.a=(self.a*(period/self.period)**(2/3)).to(u.AU)
        return self.a
    def find_period(self):
        self.period=(2*np.pi*np.sqrt(self.a**3/(G*(self.M+self.Mp)))).to(u.d)
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
        return (np.sqrt((self.a**3)/(G*(self.M+self.Mp))) * (eta - self.e*np.sin(eta)) - (t-self.tPeri)).to(u.d)
    
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
    
    def find_epsilons(self,t): # Finds and returns the fractional radius variation for both objects
        epsilon=find_epsilon(self,t,self.beta,self.Mp,self.M,self.R,self.vTheta,self.vPhi) # fractional radius change of star
        epsilon_c=find_epsilon(self,t,self.betap,self.M,self.Mp,self.Rp,self.vTheta,self.vPhi+np.pi) # fractional radius change of planet
        return epsilon,epsilon_c
    def find_dL_t(self,t): # The luminosity change of both the star and companion due to tides
        epsilon,epsilon_c=self.find_epsilons(t)
        delta=-(49+16*(self.beta**-1))*epsilon/40
        delta_c=-(49+16*(self.betap**-1))*epsilon_c/40
        lum=1 # this is where we calculate the luminosity of the star and the companion
        lum_c=0.001
        return lum*delta + lum_c*delta_c
        
    def find_vs(self,t): # Finds and returns the line of sight velocity of both objects
        v=find_v(self,t,self.M,self.vPhi)
        v_c=find_v(self,t,self.Mp,self.vPhi+np.pi)
        return v,v_c
    def find_dL_b(self,t): # The luminosity change of both the star and companion due to beaming
        v,v_c=self.find_v(t)
        # need a good way of integrating over F_l over waveband of telescope
        return
        
    def find_dL_r(self,t): # The reflected luminosity of both star and planet
        Phi=self.find_Phi(t)
        gamma=np.arccos(-np.sin(self.vTheta)*np.cos(self.vPhi-Phi))
        gamma_c=np.arccos(-np.sin(self.vTheta)*np.cos(self.vPhi+np.pi-Phi))
        d=self.find_d(t)
        delta=self.Ag*np.power(self.Rp/d,2)*(np.sin(gamma)+(np.pi-gamma)*np.cos(gamma))
        delta_c=self.Agp*np.power(self.R/d,2)*(np.sin(gamma_c)+(np.pi-gamma_c)*np.cos(gamma_c))
        lum=1
        lum_c=0.001
        return lum*delta + lum_c*delta_c
        
# Finds the fractional radius variation at some angle (theta,phi) along the line of sight due to tides (epsilon = R/R0 - 1) for a particular body (can be star or companion)
def find_epsilon(sys,t,beta,m,M,r,theta,phi):
    eta=system.find_eta(t)
    alpha=beta*(m/M)*np.power(r/sys.a,3)/4
    gamma=alpha*np.power(1-sys.e*np.cos(eta),-3)
    Phi=sys.find_Phi(t)
    psi=phi-Phi
    return gamma*2*(3 * np.sin(theta)**2 * np.cos(psi)**2 - 1)
# Finds the line of sight velocity of an object
def find_v(sys,t,M,vPhi):
    # Using equation from Lovis+Fischer 2010 (I.e. not rederived by hand yet)
    Phi=sys.find_Phi(t)
    psi=vPhi-Phi
    v=-np.sqrt(G/((sys.M+sys.Mp)*sys.a*(1-sys.e**2)))*M*np.sin(sys.vTheta)*(np.sin(psi)+sys.e*np.sin(vPhi))
    # this should have sign such that it's positive when moving away from the observer
    return v
    
def F_l(R,D,T,l): # dF/dl at a given wavelength
    return 2*np.pi*np.power(R/D,2)*(h*c**2/l**5)/(np.exp(h*c/(l*k*T))-1)
    