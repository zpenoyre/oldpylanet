from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import wget
import kplr
import os
import glob
from astropy.io import fits

class keplerLC(object):
    def __init__(self, KIC, LCdirectory, cadence="both"):
        self.KIC = KIC
        self.LCdirectory = LCdirectory
        self.LCfilelist = []
        self.cadence = cadence
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
    
    print("cadence is {0}".format(cadence))
    return keplerLC(KIC, dataDirectory, cadence)

