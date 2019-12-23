from __future__ import division
import numpy as np

import scipy.optimize
import scipy.integrate
import astropy.units as u
import astropy
from astropy.io import ascii
import os
import inspect

G=astropy.constants.G #2946 #in Rsun, Msun and days
h=astropy.constants.h
k=astropy.constants.k_B
c=astropy.constants.c
sigma=astropy.constants.sigma_sb

class system:
    def __init__(self):
        self.M=1.*u.Msun # 1 Msun
        self.Mp=0.001*u.Msun # 1 Mjupiter
        self.R=1.*u.Rsun # 1 Rsun
        self.Rp=0.1*u.Rsun # 1 Rjupiter
        self.T=5000.*u.K # Tsun
        self.Tp=500.*u.K

        self.a=0.1*u.AU # ~ orbit of mercury in Rsun
        self.e=0.01
        self.period=1.0*u.d # in days

        self.D=100.*u.kpc # in kpc

        self.Ag=0.2 # geometric albedo
        self.beta=1.03 # stellar response to tides for n=3 polytrope

        # The reflectivity and tidal response of the companion
        # (any differences negligible in all but most extreme cases)
        self.Agp=0.2 # geometric albedo of companion
        self.betap=1.03 # companion response to tides for n=3 polytrope

        self.tPeri=0.*u.d # time of periapse
        self.vPhi=0.5*np.pi # azimuthal projection angle (relative to periapse)
        self.vTheta=0.49*np.pi # polar projection angle (approx pi/2 for transiting planet)

        self.L=1.*u.Lsun
        self.Lp=0.001*u.Lsun
        # these are the effective luminosities given the filter (< actual luminosity)
        self._Lobs=1.*u.Lsun
        self._Lpobs=0.001*u.Lsun

    # A series of functions to find (and set) parameters that can be derived from other known parameters
    def find_a(self): # given M+Mp and period -> a
        period=(2*np.pi*np.sqrt(self.a**3/(G*(self.M+self.Mp)))).to(u.d)
        self.a=(self.a*(self.period/period)**(2./3.)).to(u.AU)
        return self.a
    def find_period(self): # given M+Mp and a -> period
        self.period=(2*np.pi*np.sqrt(self.a**3/(G*(self.M+self.Mp)))).to(u.d)
        return self.period
    def find_mass(self,massFactor=1e-3): # given period and a -> M+Mp
        totalMass=4*(np.pi**2)*(self.a**3)/(G*self.period**2)
        self.M=totalMass/(1*massFactor)
        self.Mp=massFactor*totalMass/(1*massFactor)
        return self.M,self.Mp

    def update_Ls(self): # given R, Rp, T and Tp -> L and Lp
        L=Luminosity(self.R,self.T).to(u.Lsun)
        Lp=Luminosity(self.Rp,self.Tp)
        self.L=4*np.pi*sigma*(self.R**2)*(self.T**4)
        self.Lp=4*np.pi*sigma*(self.Rp**2)*(self.Tp**4)
        self._Lobs=L.to(u.Lsun)
        self._Lpobs=Lp.to(u.Lsun)
        return L,Lp

    #_________________________________
    # General orbital parameters as a function of time
    # Note: eta is generally a more useful angle than Phi,
    # but Phi is more easily understood (the angle from periapse of the planet relative to c.o.m.)

    # Finds the seperation of the planet and star
    def find_d(self,t):
        eta=self.find_eta(t)
        return self.a*(1.-self.e*np.cos(eta))
    # Finds Phi at a given time
    def find_Phi(self,t):
        eta=self.find_eta(t)
        return self.find_Phi_eta(eta)

    # Functions that convert eta<->Phi
    # If we know Phi and would like to find eta (and subsequently t)
    def find_eta_Phi(self,Phi):
        # note that eta can be outside of range 0->2*pi but this will only give values in that range
        return 2*np.arctan(np.sqrt((1.-self.e)/(1.+self.e))*np.tan(Phi/2.))
    # If we know eta and would like to find Phi
    def find_Phi_eta(self,eta):
        return 2*np.arctan(np.sqrt((1.+self.e)/(1.-self.e))*np.tan(eta/2.)) % (2*np.pi)

    # Similarly can find time of center of transit
    # note: at the moment this just gives the time when Phi=vPhi - could also find ingress/egress or check for transit
    def find_tTransit(self):
        eta0=self.find_eta_Phi(self.vPhi)
        return self.find_t(eta0)

    # Can be used to find t (when t argument omitted) or solved to find eta (when t supplied - see find_eta)
    def find_t(self,eta,t=0.):
        return (np.sqrt((self.a**3)/(G*(self.M+self.Mp))) * (eta - self.e*np.sin(eta)) - (t-self.tPeri)).to(u.d)

    # Finds eta for a given time (inverting t=(T/2pi)*(eta - e sin(eta)) )
    # note: currently using approximate solution to O(e^3) from Penoyre and Sandford 2018 - see OoT for exact solution
    def find_eta(self,t): #[raw units->unitless]
        eta0=((t-self.tPeri)*np.sqrt(G*(self.M+self.Mp)/ self.a**3)).to(u.m/u.m).value
        eta1=self.e*np.sin(eta0)
        eta2=(self.e**2)*np.sin(eta0)*np.cos(eta0)
        eta3=(self.e**3)*np.sin(eta0)*(1.-(3./2.)*(np.sin(eta0)**2))
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
        return (self._Lobs*delta + self._Lpobs*delta_c)/(self._Lobs+self._Lpobs)

    def find_vs(self,t): # Finds and returns the line of sight velocity of both objects
        v_c=find_v(self,t,self.M,self.vPhi)
        v=find_v(self,t,self.Mp,self.vPhi+np.pi)
        return v,v_c
    def find_dL_b(self,t): # The luminosity change of both the star and companion due to beaming
        v,v_c=self.find_vs(t)
        ls,phis=windowFunction()
        l0s=np.array([l/(1+(v/c)) for l in ls])*u.m
        intensity=I_l(l0s,self.T)
        # integrating the luminosity function over the window (using trapezoidal integration)
        product=np.multiply(phis,intensity.T).T
        Ls=4 * np.pi**2 * self.R**2 * np.trapz(product,dx=ls[1]-ls[0],axis=0)

        Delta=(1-5*(v/c))*Ls

        # Repeating the above process for the companion
        l0s_c=np.array([l/(1+(v/c)) for l in ls])
        intensity_c=I_l(l0s,self.Tp)
        product_c=np.multiply(phis,intensity_c.T).T
        Ls_c=4 * np.pi**2 * self.Rp**2 * np.trapz(product_c,dx=ls[1]-ls[0],axis=0)

        Delta_c=(1-5*(v_c/c))*Ls_c

        return (Delta+Delta_c) / (self._Lobs+self._Lpobs) - 1

    def find_dL_r(self,t): # The reflected luminosity of both star and planet
        Phi=self.find_Phi(t)
        gamma=np.arccos(-np.sin(self.vTheta)*np.cos(self.vPhi-Phi))
        gamma_c=np.arccos(-np.sin(self.vTheta)*np.cos(self.vPhi+np.pi-Phi))
        d=self.find_d(t)
        delta=self.Agp*np.power(self.Rp/d,2)*(np.sin(gamma)+(np.pi-gamma)*np.cos(gamma))/np.pi
        delta_c=self.Ag*np.power(self.R/d,2)*(np.sin(gamma_c)+(np.pi-gamma_c)*np.cos(gamma_c))/np.pi
        return (self._Lobs*delta + self._Lpobs*delta_c)/(self._Lobs+self._Lpobs)

    def lightcurve(self,t):
        deltas=self.find_dL_t(t)+self.find_dL_b(t)+self.find_dL_r(t)
        return 1.+deltas

# Finds the fractional radius variation at some angle (theta,phi) along the line of sight due to tides (epsilon = R/R0 - 1) for a particular body (can be star or companion)
def find_epsilon(sys,t,beta,m,M,r,theta,phi):
    eta=sys.find_eta(t)
    alpha=beta*(m/M)*np.power(r/sys.a,3)/4.
    gamma=alpha*np.power(1.-sys.e*np.cos(eta),-3)
    Phi=sys.find_Phi(t)
    psi=phi-Phi
    return gamma*2.*(3. * np.sin(theta)**2 * np.cos(psi)**2 - 1.)

# Finds the line of sight velocity of an object
def find_v(sys,t,M,vPhi):
    # Using equation from Lovis+Fischer 2010 (I.e. not rederived by hand yet)
    Phi=sys.find_Phi(t)
    psi=vPhi-Phi
    v=-np.sqrt(G/((sys.M+sys.Mp)*sys.a*(1.-sys.e**2)))*M*np.sin(sys.vTheta)*(np.sin(psi)+sys.e*np.sin(vPhi))
    # this should have sign such that it's positive when moving away from the observer
    return v
def I_l(l,T): # dF/dl at a given wavelength
    hc2_l5=(h*c**2/l**5)
    return 2*hc2_l5/(np.exp(h*c/(l*k*T))-1.)

def Luminosity(R,T,nPoints=36,window='kepler'): # currently inetgrating over a very rough kepler bandpass
    ls,phis=windowFunction(nPoints=nPoints,window=window)
    intensity=I_l(ls,T)
    return (4 * np.pi**2 * R**2 * np.trapz(phis*intensity,dx=ls[1]-ls[0])).to(u.Lsun)

#_________________________________
# Below functions for finding the response function of the telescope in a computable format
def interpolateYs(xs,ys,trapXs): #assumes xs in ascending order!
    diffs=np.array([trapX-xs for trapX in trapXs])
    diffs[diffs<0]=100 # want to make sure we always get the low index
    lowIndex=np.argmin(diffs,axis=1)
    lowIndex[lowIndex==xs.size-1]=xs.size-2
    below=trapXs-xs[lowIndex]
    #if (np.max(lowIndex)==xs.size-1) | (xs[np.min(lowIndex)]<xs[0]):
    #    return np.inf*np.ones_like(trapXs) # can break if tryinging to sample xs beyond the recorded data
    above=xs[lowIndex+1]-trapXs
    intYs=ys[lowIndex]+(below/(below+above))*(ys[lowIndex+1]-ys[lowIndex])
    intYs[intYs<0]=0 # interpolation may take some values into the negative
    return intYs


def windowFunction(nPoints=36,window='kepler'):
    if window=='uniform':
        xs=np.linspace(100,2000,nPoints)*1e-9*u.m
        ys=np.ones(nPoints)
        return xs,ys
    thisDir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    if window=='kepler':
        digitize=ascii.read(thisDir+'/responseFunctions/keplerDigitize.csv')
        ls=digitize['x']*u.m
        phis=digitize['responseFunction']
        xs=np.linspace(ls[0],ls[-1],nPoints)
        ys=interpolateYs(ls,phis,xs)
    return xs,ys
