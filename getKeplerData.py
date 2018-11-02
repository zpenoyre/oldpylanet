from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import wget
import kplr
import os
from astropy.io import fits


def download(target, downloadFolderPath="downloadedData"):
    """Download Kepler data for a target.
    Args:
        target (float, int, or str): The target to download.
            Can be a KOI number (float), KIC (int), or Kepler planet name (str).
        downloadFolderPath(str): The relative path to the folder where the target's 
            light curve fits files will be downloaded.
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

    KICname = str(KIC).zfill(9)
    KICshort=KICname[0:4]

    #folder to hold the downloaded light curves
    dataDirectory = os.getcwd()+"{0}/KIC_{1}".format(downloadFolderPath.split(".")[1],KIC)
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

#next functions to write:

def readIn(KIC, dataFolder, cadence="both", plot=True):
    """Having downloaded the Kepler light curves for a target (see above), read in the
    time, flux, fluxerr columns. 
    Args:
        KIC (int): KIC number for this star
        dataFolder (str): The relative path to the folder where the target's 
            light curve fits are.
        cadence (str): Read in only short-cadence data, only long-cadence data, or both?
        plot (bool): Plot the data, or no?
    """

    dataDirectory = os.getcwd() + dataFolder.split(".")[1]

    lcs = []

    for lcName in os.listdir(dataDirectory):
        if cadence == "short":
            if lcName[-8:] == "slc.fits":
                lcs.append(lcName)

        elif cadence == "long":
            if lcName[-8:] == "llc.fits":
                lcs.append(lcName)
        else:
            lcs.append(lcName)

    #sort light curve fits files by date
    dates = []
    for lc in lcs:
        dates.append(int(lc[14:-9]))

    dates = np.array(dates)
    dateIdxs = np.argsort(dates)

    lcs = np.array(lcs)[dateIdxs]
    
    # Loop over the datasets and read in the data.
    time, flux, ferr, quality = [], [], [], []
    print(len(lcs),' lightcurves')
    for lc in lcs:
        f = fits.open(dataDirectory+'/'+lc)

        # The lightcurve data are in the first FITS HDU.
        hdu_data = f[1].data

        time.append(hdu_data["time"])
        flux.append(hdu_data["sap_flux"])
        ferr.append(hdu_data["sap_flux_err"])
        quality.append(hdu_data["sap_quality"])

    firstObs=0
    lastObs=len(time)-1
    
    if plot==True:
        firstPlot=plt.gca()
        firstPlot.plot(np.hstack(time[firstObs:lastObs+1]),(np.hstack(flux[firstObs:lastObs+1])/np.nanmean(np.hstack(flux[firstObs:lastObs+1])))-1,'k.',alpha=0.25)
        plt.show()

    return time, flux, ferr, quality



