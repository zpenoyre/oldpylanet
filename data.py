from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import wget
import kplr
import os
import glob
from astropy.io import fits
from numba import jit

class keplerLC(object):
    def __init__(self, KIC, LCdirectory, period, cadence="both"):
        self.KIC = KIC
        self.LCdirectory = LCdirectory
        self.LCfilelist = []
        self.cadence = cadence
        self.period=period
        self._times = None
        self._cadencenos = None
        self._sap_fluxs = None
        self._sap_flux_errs = None
        self._sap_bkgs = None
        self._sap_bkg_errs = None
        self._pdcsap_fluxs = None
        self._pdcsap_flux_errs = None
        self._sap_qualitys = None
        self._psf_centr1s = None
        self._psf_centr1_errs = None
        self._psf_centr2s = None
        self._psf_centr2_errs = None
        
        #get list of LC fits files
        for lcName in os.listdir(self.LCdirectory):
            if (str(self.KIC) not in lcName) or not lcName.endswith(".fits"):
                pass
            else:
                if self.cadence == "short":
                    #print("short-cadence only")
                    if lcName[-8:] == "slc.fits":
                        self.LCfilelist.append(lcName)

                elif self.cadence == "long":
                    #print("long-cadence only")
                    if lcName[-8:] == "llc.fits":
                        self.LCfilelist.append(lcName)
                else:
                    #print("both cadences")
                    self.LCfilelist.append(lcName)

        #sort LC file names by date
        self.dates = []
        for lc in self.LCfilelist:
            self.dates.append(int(lc[14:-9]))

        self.dates = np.array(self.dates)
        dateIdxs = np.argsort(self.dates)

        self.LCfilelist = np.array(self.LCfilelist)[dateIdxs]
        
    def getHDUdata(self, hdu_key):
        dataList = []
        for lc in self.LCfilelist:
            with fits.open(self.LCdirectory+'/'+lc, memmap=False) as f:
                # The lightcurve data are in the first FITS HDU.
                hdu_data = f[1].data
                dataList.append(hdu_data[hdu_key])

        return dataList

    @property
    def times(self):
        """ Get list of time arrays """
        if self._times is None:
            self._times = self.getHDUdata("time")
        return self._times

    @property
    def cadencenos(self):
        """ Get list of cadenceno arrays """
        if self._cadencenos is None:
            self._cadencenos = self.getHDUdata("cadenceno")
        return self._cadencenos

    @property
    def sap_fluxs(self):
        """ Get list of sap_flux arrays """
        if self._sap_fluxs is None:
            self._sap_fluxs = self.getHDUdata("sap_flux")
        return self._sap_fluxs

    @property
    def sap_flux_errs(self):
        """ Get list of sap_flux_err arrays """
        if self._sap_flux_errs is None:
            self._sap_flux_errs = self.getHDUdata("sap_flux_err")
        return self._sap_flux_errs

    @property
    def sap_bkgs(self):
        """ Get list of sap_bkg arrays """
        if self._sap_bkgs is None:
            self._sap_bkgs = self.getHDUdata("sap_bkg")
        return self._sap_bkgs

    @property
    def sap_bkg_errs(self):
        """ Get list of sap_bkg_err arrays """
        if self._sap_bkg_errs is None:
            self._sap_bkg_errs = self.getHDUdata("sap_bkg_err")
        return self._sap_bkg_errs

    @property
    def pdcsap_fluxs(self):
        """ Get list of pdcsap_flux arrays """
        if self._pdcsap_fluxs is None:
            self._pdcsap_fluxs = self.getHDUdata("pdcsap_flux")
        return self._pdcsap_fluxs

    @property
    def pdcsap_flux_errs(self):
        """ Get list of pdcsap_flux_err arrays """
        if self._pdcsap_flux_errs is None:
            self._pdcsap_flux_errs = self.getHDUdata("pdcsap_flux_err")
        return self._pdcsap_flux_errs

    @property
    def sap_qualitys(self):
        """ Get list of sap_quality arrays """
        if self._sap_qualitys is None:
            self._sap_qualitys = self.getHDUdata("sap_quality")
        return self._sap_qualitys

    @property
    def psf_centr1s(self):
        """ Get list of psf_centr1 arrays """
        if self._psf_centr1s is None:
            self._psf_centr1s = self.getHDUdata("psf_centr1")
        return self._psf_centr1s

    @property
    def psf_centr1_errs(self):
        """ Get list of psf_centr1_err arrays """
        if self._psf_centr1_errs is None:
            self._psf_centr1_errs = self.getHDUdata("psf_centr1_err")
        return self._psf_centr1_errs

    @property
    def psf_centr2s(self):
        """ Get list of psf_centr2 arrays """
        if self._psf_centr2s is None:
            self._psf_centr2s = self.getHDUdata("psf_centr2")
        return self._psf_centr2s

    @property
    def psf_centr2_errs(self):
        """ Get list of psf_centr2_err arrays """
        if self._psf_centr2_errs is None:
            self._psf_centr2_errs = self.getHDUdata("psf_centr2_err")
        return self._psf_centr2_errs

    def plotLC(self, **kwargs):
        """ Plot flux vs. time"""
        fig=plt.figure(figsize=(8,6))
        plt.plot(np.hstack(self.times),
            (np.hstack(self.sap_fluxs)/np.nanmean(np.hstack(self.sap_fluxs))),
            'k.',
            alpha=0.25)
        plt.show()
        return fig

def downloadKepler(KIC, dataDirectory):
    """Download Kepler data for a target.
    Args:
        KIC (int)
        dataDirectory(str): The relative path to the folder where the target's 
            light curve fits files will be downloaded.
    """
    KICname = str(KIC).zfill(9)
    KICshort=KICname[0:4]

    #folder to hold the downloaded light curves
    print("Downloading light curves to {0}.".format(dataDirectory))

    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)

    ftpfolder='http://archive.stsci.edu/pub/kepler/lightcurves//'+KICshort+'/'+KICname+'/'
    #print ftpfolder

    fileList = wget.download(ftpfolder)

    with open(fileList, 'r') as listOfLinks:
        lol=listOfLinks.read()

    lol = lol.split('href="kplr')[1:-2]

    fileNames = []

    for linkString in lol:
        linkString = linkString.split('">')[1]
        linkString = linkString.split('<')[0]
        fileNames.append(linkString)

    #list to hold names of long cadence light curve fits files
    n_lcs = 0
    for i,fileToDownload in enumerate(fileNames):
        if i%5==0:
            print('downloading file ',i,' of ',len(fileNames))
        targetFile = ftpfolder+fileToDownload
        #print targetFile
        lcName = dataDirectory+'/'+fileToDownload
        #download fits file if necessary
        if not os.path.exists(lcName):
            wget.download(targetFile, lcName)
            n_lcs += 1
    
    print("{0} light curve files downloaded.".format(n_lcs))
    #delete download list
    os.system('rm ./download.wget')

    return

def getData(target, dataFolderPath, cadence="both", plot=True):
    """Combines the functionality of download and readIn, below.
    Args:
        (float, int, or str): The target to download.
            Can be a KOI number (float), KIC (int), or Kepler planet name (str).
        dataFolderPath(str): The relative path to the folder where the target's 
            light curve fits files (a) already are or (b) will be downloaded.
        cadence ("short","long","both"): Read in only short-cadence data,
            only long-cadence data, or both?
        plot (bool): Plot the read-in data, or no?
    """

    client = kplr.API()
    period=-1
    
    # Find the target.
    if isinstance(target, str):
        target = client.planet(target)
        #period=target.koi_period
        KIC = target.kepid

    elif isinstance(target, int):
        KIC = target
        target = client.star(target)

    elif isinstance(target, float):
        target = client.koi(target)
        #period=target.koi_period
        KIC=target.kepid
        period=target.koi_period
        

    print("Target: KIC {0}.".format(KIC))
    dataDirectory = os.getcwd() + dataFolderPath.split(".")[1]
    #print(dataDirectory)

    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)

    #print("{0}KIC_10666592/*{1}*.fits".format(dataDirectory,KIC))
    #print(len(glob.glob("{0}/*{1}*.fits".format(dataDirectory,KIC))))
    
    nDataFiles = len(glob.glob("{0}/*{1}*.fits".format(dataDirectory,KIC)))
    print("{0} data files.".format(nDataFiles))

    if nDataFiles==0:
        downloadKepler(KIC, dataDirectory)
    
    print("cadence is {0}".format(cadence))
    print("period is {0}".format(period))
    return keplerLC(KIC, dataDirectory, period, cadence)

# lots of nans in kepler data, this will find them and either approximate the value or cut our losses and split the observing period up
# nMin can be used to specify the number of data points below which we won't bother keeping the data (i.e. if it's less than a full period)
def cleanGaps(ts,fs,es,nMin=0):
    nPeriods=len(ts)
    if type(ts)!=list: #probably we've passed this a single numpy array (rather than a list of them)
        ts=[ts]
        fs=[fs]
        es=[es]
        nPeriods=1
    #print(ts)
    times=[]
    fluxs=[]
    errors=[]
    for i in range(nPeriods):
        theseTs=ts[i]
        theseFs=fs[i]
        theseEs=es[i]
        nPoints=theseTs.size
        if nPoints<nMin:
            continue
        gaps=np.flatnonzero(~np.isfinite(theseTs+theseFs+theseEs)) # if any of the three is NaN this will give there indices 
        gaps=np.hstack([0,gaps,nPoints]) # adds in 0 and N as boundaries
        skips=gaps[1:]-gaps[:-1] # where this equals 1 we have multiple skipped data points concurrently (i.e. a true gap)
        #print('skips: ',skips)
        singles=gaps[np.flatnonzero(skips>1)[1:]]
        # where there is just one bad data point we just average the two either side
        theseTs[singles]=0.5*(theseTs[singles+1] + theseTs[singles-1])
        theseFs[singles]=0.5*(theseFs[singles+1] + theseFs[singles-1])
        theseEs[singles]=0.5*(theseEs[singles+1] + theseEs[singles-1])
        
        jumps=np.flatnonzero((skips[1:]==1) & (skips[:-1]!=1))+1 #first nan in a long string of nans
        endJumps=np.flatnonzero((skips[:-1]==1) & (skips[1:]!=1))+1 #last nan in a long string of nans
        #print('jumps: ',jumps)
        #print('endJumps: ',endJumps)
        #print(gaps[jumps])
        #print(gaps[endJumps])
        if jumps.size==0: # no big jumps to cut output
            times.append(theseTs)
            fluxs.append(theseFs)
            errors.append(theseEs)
        else:
            for j in range(jumps.size+1):
                #print('j: ',j)
                #print('of ',jumps.size+1)
                if j==0:
                    lowEnd=0
                    while ~np.isfinite(theseTs[lowEnd]+theseFs[lowEnd]+theseEs[lowEnd]):
                        lowEnd+=1
                else:
                    lowEnd=gaps[endJumps[j-1]]+1 # go one beyond the end of the last jumps
                if j==(jumps.size):
                    highEnd=nPoints
                    while ~np.isfinite(theseTs[highEnd-1]+theseFs[highEnd-1]+theseEs[highEnd-1]):
                        highEnd+=-1
                else:
                    highEnd=gaps[jumps[j]] # end one before the next jump starts 
                if (highEnd-lowEnd)<nMin:
                    continue
                #print('lowEnd: ',lowEnd)
                #print('highEnd: ',highEnd)
                #print('sum: ',np.sum(theseTs[lowEnd:highEnd]))
                times.append(theseTs[lowEnd:highEnd])
                fluxs.append(theseFs[lowEnd:highEnd])
                errors.append(theseEs[lowEnd:highEnd])
    return times,fluxs,errors

def detrend(times,fluxs,errors,period,method='box',cadence=-1):
    nObs=len(times)
    if cadence==-1:
        cadence=1765.5/(24*60**2)
    #print('cadence: ',cadence)
    if method=='box':
    #    windowWidth=int(0.5*period/cadence)
    #    #print('wW: ',windowWidth)
    #    if nObs<2*windowWidth:
    #        return np.array([]),np.array([]),np.array([])
    #    ts=times[windowWidth:-windowWidth]
    #    fs=fluxs[windowWidth:-windowWidth]
    #    es=errors[windowWidth:-windowWidth] # do we want to try and find the error in the median?
    #    indices=np.array([np.arange(i,i+2*windowWidth) for i in range(nObs-2*windowWidth)])
    #    fMed=np.median(fluxs[indices],axis=1)
    #    eMed=1.4826*np.median(np.abs(fluxs[indices]-fMed[:,None]),axis=1)/np.sqrt(2*windowWidth+1)
        #print(np.max(indices))
        #print(indices.shape)
        #print(fs.size)
    #    return ts,fs-fMed,np.sqrt(es**2 + eMed**2)
        windowSize=2*int(0.5*period/cadence)+1
        windowWidth=int(windowSize/2)
        weights=np.ones(windowSize)/windowSize
        return convolve(times,fluxs,errors,weights)
    if method=='triangle':
        windowSize=2*int(period/cadence) + 1
        windowWidth=int(windowSize/2)
        window=np.hstack([np.arange(int(windowSize/2)+1),np.arange(int(windowSize/2),0,-1)])
        weights=window/np.sum(window)
        return convolve(times,fluxs,errors,weights)
    if method=='cos':
        windowSize=2*int(period/cadence) + 1
        windowWidth=int(windowSize/2)
        dt=2*period/windowSize
        ts=np.linspace(-period,period+dt,windowSize)
        rawWeights=np.cos(np.pi*ts/(1.136*period))/(1-4*(ts/(1.136*period))**2)
        weights=rawWeights/np.sum(rawWeights)
        return convolve(times,fluxs,errors,weights)
    if method=='cosalt':
        windowSize=2*int(1.193*period/cadence) + 1
        windowWidth=int(windowSize/2)
        ts=np.linspace(-1.193*period,1.193*period,windowSize)
        rawWeights=np.cos(np.pi*ts/period)/(1-4*(ts/period)**2)
        weights=rawWeights/np.sum(rawWeights)
        return convolve(times,fluxs,errors,weights)
    if method=='castle':
        windowSize=6*int(period/cadence)+1 # always odd
        windowWidth=int(windowSize/2)
        if nObs<windowSize:
            return np.array([]),np.array([]),np.array([])
    #    nData=nObs-windowSize # number of usable data points
    #    ts=times[1+windowWidth:-windowWidth]
    #    fs=fluxs[1+windowWidth:-windowWidth]
    #    es=errors[1+windowWidth:-windowWidth] # do we want to try and find the error in the median?
        #print('nData: ',nData)
        #print('fs.size: ',fs.size)
        
        #setting up the weighting function
        #print('windowSize: ',windowSize)
        Ts=np.arange(-windowWidth,windowWidth+1)
        #print('windowWidth: ',windowWidth)
        #print(np.max(Ts))
        #print(np.min(Ts))
        #print('nObs: ',nObs)
        #print('windowWidth: ',windowWidth)
        #print('Ts.size: ',Ts.size)
        sinc=np.sin(2*np.pi*Ts/windowWidth) / (2*np.pi*Ts/windowWidth)
        nans=np.flatnonzero(~np.isfinite(sinc)) # should be single nan at sinc(0) - numpy can't handle this...
        sinc[nans]=1
        weights=sinc/((0.5*windowWidth)**2 - Ts**2)
        wNans=np.flatnonzero(~np.isfinite(weights))
        weights[wNans]=1
        weightSum=np.sum(weights)
        weightsNorm=weights/weightSum
        return convolve(times,fluxs,errors,weightsNorm)
    #    #print(windowWidth)
    #    #print(nObs)
    #    indices=np.array([np.arange(i,i+windowSize) for i in range(nData)])
    #    #print('indices.shape: ',indices.shape)
    #    #print('max index: ',np.max(indices))
    #    argsort=np.argsort(fluxs[indices],axis=1)
    #    #print('argsort shape: ',argsort.shape)
    #    #print('argsort: ',argsort)
    #    sortWeights=weightsNorm[argsort]
    #    #print('sortWeights shape: ',sortWeights.shape)
    #    #print('sortWeights: ',sortWeights)
    #    cumSum=np.cumsum(sortWeights,axis=1)
    #    #print('cumSum shape: ',cumSum.shape)
    #    #print('cumSum: ',cumSum)
    #    medians=np.argmin(np.abs(cumSum-0.5),axis=1)
    #    #print('medians shape: ',medians.shape)
    #    #print('medians: ',medians)
    #    medianIndices=indices[np.arange(medians.size),argsort[np.arange(medians.size),medians]]
    #    #print('medianIndices: ',medianIndices)
    #    cosTerm=fluxs[medianIndices]
    #    #print(0.5*period/cadence)
    #    halfWidth=int((period/cadence))
    #    #print('halfWidth: ',halfWidth)
    #    sinTerm=fs-0.5*(fluxs[1+windowWidth-halfWidth:1+windowWidth-halfWidth+nData]+fluxs[1+windowWidth+halfWidth:1+windowWidth+halfWidth+nData])
    #    fMed=cosTerm#+sinTerm
        #print('fs.size: ',fluxs[0:-2*windowWidth].size)
        #print('fluxs[medianIndices].size: ',fluxs[2*windowWidth:].size)
    #    eMed=1.4826*np.median(np.abs(fs-fMed))/np.sqrt(windowSize)
    #    return ts,fs-fMed,np.sqrt(es**2 + eMed**2)
    if 'double' in method:
        if 'os' in method:
            times,fluxs,errors=detrend(times,fluxs,errors,period,method='cos',cadence=-1) # initially detrends out the low frequency noise
        else:
            times,fluxs,errors=detrend(times,fluxs,errors,period,method='box',cadence=-1)
        nObs=len(times)
        windowSize=2*int(0.5*period/cadence)+1 # always odd
        windowWidth=int(windowSize/2)
        if nObs<windowSize+1:
            return np.array([]),np.array([]),np.array([])
        ts=times[windowWidth:-windowWidth]
        #fs=fluxs[windowWidth:-windowWidth]
        es=errors[windowWidth:-windowWidth]
        
        #print('window size: ',windowSize)
        #print('window width: ',windowWidth)
        #print(windowSize-windowWidth)
        #print(fluxs[windowWidth:windowSize].size)
        #print(len(fluxs[windowWidth:windowSize]))
        #print(fluxs[windowWidth:windowSize])
        #print('hello: ',np.arange(10)[5:15])
        #fs=np.zeros_like(fluxs)
        #fs[windowWidth:-windowWidth]=0.5*(fluxs[:-windowSize+1]+fluxs[windowSize-1:])
        fs=0.5*(fluxs[:-windowSize+1]+fluxs[windowSize-1:])
        # i think it's safer to just not perform the second detrending on the edge points
        #fs[:windowWidth+1]=fluxs[windowWidth:windowSize]
        #fs[:windowWidth+1]=fluxs[:windowWidth+1]
        
        #fs[-windowWidth:]=fluxs[-windowSize+1:-windowWidth]
        #fs[-windowWidth:]=fluxs[-windowWidth:]
        #print(np.flatnonzero(fs==0))
        #print(ts.size)
        
        return ts,fs,es

def cleanData(ts,fs,es,period,nMin=1000,cadence=-1,detrendMethod='box'):
    times,fluxs,errors=cleanGaps(ts,fs,es,nMin=nMin)
    nPeriods=len(times)
    allTs=np.array([])
    allFs=np.array([])
    allEs=np.array([])
    for i in range(nPeriods):
        meanFlux=np.median(fluxs[i]) # normalise fluxs centred around 1
        pTs,pFs,pEs=detrend(times[i],fluxs[i]/meanFlux,errors[i]/meanFlux,period,method=detrendMethod,cadence=cadence)
        allTs=np.hstack([allTs,pTs])
        allFs=np.hstack([allFs,pFs])
        allEs=np.hstack([allEs,pEs])
    #medianFs=np.median(allFs)
    #allFs=(allFs/medianFs)-1
    #allEs=allEs/medianFs
    return allTs,allFs,allEs
    
def convolve(times,fluxs,errors,weights):
    #nObs=int(times.size)
    width=int(weights.size/2)
    ts=times[width:-width]
    fs=fluxs[width:-width]
    es=errors[width:-width]
    fMeds=loop(fluxs,weights)#np.zeros(nObs-2*width)
    #for i in range(nObs-2*width):
    #    fMeds[i]=weightedMedian(fluxs[i:i+2*width+1],weights)
    return ts,fs-fMeds,es
    
@jit(nopython=True)
def loop(fluxs,weights):
    fMeds=np.zeros(fluxs.size-weights.size+1)
    width=weights.size
    for i in range(fMeds.size):
        fMeds[i]=weightedMedian(fluxs[i:i+width],weights)
    return fMeds

@jit(nopython=True)
def weightedMedian(xs,ws):
    argsort=np.argsort(xs)
    #w=np.sum(ws) # weights must be normalised!
    cumsum=np.cumsum(ws[argsort])
    medianIndex=argsort[np.argmin(np.abs(cumsum-0.5))]
    return xs[medianIndex]    

def stackData(ts,fs,es,period,nTs=100,offset=0.1234):
    dt=period/(nTs+1)
    binTs=np.arange(-(period-dt)/2,(period-dt)/2,dt+1e-6)
    binFs=np.zeros(nTs)
    binEs=np.zeros(nTs)
    #if offset==0.1234:
    #    offset=ts[np.argmin(fs)]
    for i in range(nTs):
        #inBin=np.flatnonzero(np.abs(((ts+(period/2)-offset)%period - period/2)-binTs[i])<dt/2)
        inBin=np.flatnonzero(np.abs(((ts+(period/2))%period - period/2)-binTs[i])<dt/2)
        #argsort=np.argsort(fs[inBin])
        #w=np.sum(1/es[inBin])
        #w_i=1/es[inBin[argsort]]
        #cumsum=np.cumsum(w_i/w)
        #weightedMedian=np.argmin(np.abs(cumsum-0.5))
        #binFs[i]=fs[inBin[argsort[weightedMedian]]]#np.median(fs[inBin])
        weights=(1/es[inBin])/np.sum(1/es[inBin])
        binFs[i]=weightedMedian(fs[inBin],weights)
        #mad=1.4826*np.median(np.abs(fs[inBin]-binFs[i]))/np.sqrt(inBin.size)
        mad=1.4826*weightedMedian(np.abs(fs[inBin]-binFs[i]),weights)/np.sqrt(inBin.size)
        binEs[i]=mad#np.sqrt(mad**2 + np.median(es[inBin])**2)
        #binEs[i]=np.sqrt(np.sum((binFs[i]-fs[inBin])**2))/np.sqrt(inBin.size)
    return binTs,binFs,binEs
    