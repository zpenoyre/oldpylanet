from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import wget
import kplr
import os
import glob
from astropy.io import fits

class keplerLC(object):
    def __init__(self, KIC, LCdirectory, **kwargs):
        self.KIC = KIC
        self.LCdirectory = LCdirectory
        self._LCfilelist = None
        self._cadence = None
        self._times = None
        self._fluxs = None
        self._fluxerrs = None
        self._qualitys = None

        allowed_keys = ["LCfilelist","cadence","times","fluxs","fluxerrs","qualitys"]
        
        
        
        #update with values passed in kwargs. values not passed in kwargs will remain None
        self.__dict__.update((k,v) for k,v in kwargs.iteritems() if k in allowed_keys)
        

    @property
    def LCfilelist(self):
        """ Get list of LC .fits files. """
        if self._LCfilelist is None:
            self._LCfilelist = []
            for lcName in os.listdir(self.LCdirectory):
                if (str(self.KIC) not in lcName) or not lcName.endswith(".fits"):
                    pass
                else:
                    if self._cadence == "short":
                        if lcName[-8:] == "slc.fits":
                            self._LCfilelist.append(lcName)

                    elif self._cadence == "long":
                        if lcName[-8:] == "llc.fits":
                            self._LCfilelist.append(lcName)
                    else:
                        self._LCfilelist.append(lcName)

            dates = []
            for lc in self._LCfilelist:
                dates.append(int(lc[14:-9]))

            dates = np.array(dates)
            dateIdxs = np.argsort(dates)

            self._LCfilelist = np.array(self._LCfilelist)[dateIdxs]
        
        return self._LCfilelist

    @property
    def times(self):
        """ Get list of time arrays """
        if self._times is None:
            self._times = []
            for lc in self.LCfilelist:
                f = fits.open(self.LCdirectory+'/'+lc)

                # The lightcurve data are in the first FITS HDU.
                hdu_data = f[1].data
                self._times.append(hdu_data["time"])
                f.close()
            
        return self._times

    @property
    def fluxs(self):
        """ Get list of SAP flux arrays """
        if self._fluxs is None:
            self._fluxs = []
            for lc in self.LCfilelist:
                f = fits.open(self.LCdirectory+'/'+lc)

                # The lightcurve data are in the first FITS HDU.
                hdu_data = f[1].data
                self._fluxs.append(hdu_data["sap_flux"])
                f.close()
            
        return self._fluxs

    @property
    def fluxerrs(self):
        """ Get list of SAP flux err arrays """
        if self._fluxerrs is None:
            self._fluxerrs = []
            for lc in self.LCfilelist:
                f = fits.open(self.LCdirectory+'/'+lc)

                # The lightcurve data are in the first FITS HDU.
                hdu_data = f[1].data
                self._fluxerrs.append(hdu_data["sap_flux_err"])
                f.close()
            
        return self._fluxerrs

    @property
    def qualitys(self):
        """ Get list of data quality flag arrays """
        if self._qualitys is None:
            self._qualitys = []
            for lc in self.LCfilelist:
                f = fits.open(self.LCdirectory+'/'+lc)

                # The lightcurve data are in the first FITS HDU.
                hdu_data = f[1].data
                self._qualitys.append(hdu_data["sap_quality"])
                f.close()
            
        return self._qualitys


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
    for fileToDownload in fileNames:
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

def getData(target, dataFolderPath, readInCadence="both", plot=True):
    """Combines the functionality of download and readIn, below.
    Args:
        (float, int, or str): The target to download.
            Can be a KOI number (float), KIC (int), or Kepler planet name (str).
        dataFolderPath(str): The relative path to the folder where the target's 
            light curve fits files (a) already are or (b) will be downloaded.
        readInCadence ("short","long","both"): Read in only short-cadence data,
            only long-cadence data, or both?
        plot (bool): Plot the read-in data, or no?

    """

    client = kplr.API()
    
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
    
    return keplerLC(KIC, dataDirectory)

    """
    lcs = []

    
    for lcName in os.listdir(dataDirectory):
        if (str(KIC) not in lcName) or not lcName.endswith(".fits"):
            pass
        else:
            if readInCadence == "short":
                if lcName[-8:] == "slc.fits":
                    lcs.append(lcName)

            elif readInCadence == "long":
                if lcName[-8:] == "llc.fits":
                    lcs.append(lcName)
            else:
                lcs.append(lcName)

    dates = []
    for lc in lcs:
        dates.append(int(lc[14:-9]))

    dates = np.array(dates)
    dateIdxs = np.argsort(dates)

    lcs = np.array(lcs)[dateIdxs]
    
    # Loop over the datasets and read in the data.
    time, flux, ferr, quality = [], [], [], []
    for lc in lcs:
        f = fits.open(dataDirectory+'/'+lc)

        # The lightcurve data are in the first FITS HDU.
        hdu_data = f[1].data

        time.append(hdu_data["time"])
        flux.append(hdu_data["sap_flux"])
        ferr.append(hdu_data["sap_flux_err"])
        quality.append(hdu_data["sap_quality"])
        f.close()

    firstObs=0
    lastObs=len(time)-1
    
    if plot is True:
        firstPlot=plt.gca()
        firstPlot.plot(np.hstack(time[firstObs:lastObs+1]),(np.hstack(flux[firstObs:lastObs+1])/np.nanmean(np.hstack(flux[firstObs:lastObs+1])))-1,'k.',alpha=0.25)
        plt.show()

    return time, flux, ferr, quality
    """
