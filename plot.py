from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['hatch.linewidth'] = 5.0
mpl.rcParams['axes.facecolor']='whitesmoke'
mpl.rcParams['axes.edgecolor']='white'
mpl.rcParams['axes.linewidth']=3

mpl.rcParams['text.color'] = 'dimgrey'
mpl.rcParams['xtick.color']='k'
mpl.rcParams['ytick.color']='k'
mpl.rcParams['axes.labelcolor']='k'

mpl.rcParams['font.size']=16
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'
mpl.rcParams['xtick.major.size'] = 5.5
mpl.rcParams['ytick.major.size'] = 5.5
mpl.rcParams['xtick.minor.size'] = 3.5
mpl.rcParams['ytick.minor.size'] = 3.5

mpl.rcParams["text.usetex"] = True

def plotData(thisGrid,ts,fs,es,period=-1,c='#05668D'): #wants a 3x2 grid to plot everything on
    darkBlue='#05668D'
    lightBlue='#427AA1'
    offWhite='#EBF2FA'

    if period==-1:
        period=np.max(ts)-np.min(ts) #almost true...

    # finds minimum data point and assumes it;s the transit middle
    offset=ts[np.argmin(fs)]
    ts=(ts-offset+period/2)%period - period/2 
    sort=np.argsort(ts)
    nTs=sort.size


    fullTs=np.hstack([ts[sort[-int(nTs/4):]]-period,ts,period+ts[sort[:int(nTs/4)]]])
    fullFs=np.hstack([fs[sort[-int(nTs/4):]],fs,fs[sort[:int(nTs/4)]]])
    fullEs=np.hstack([es[sort[-int(nTs/4):]],es,es[sort[:int(nTs/4)]]])
    mainPlot=plt.subplot(thisGrid[0,0:2])
    #mainTs=np.hstack(ts)
    mainPlot.errorbar(fullTs,1e6*(fullFs),yerr=1e6*fullEs,fmt="none",markersize='1',lw=1,c='grey',zorder=2)
    mainPlot.scatter(fullTs,1e6*(fullFs),s=20,edgecolors='k',facecolors=c,zorder=3)
    mainPlot.axvline(np.min(ts),c='grey',alpha=0.5)
    mainPlot.axvline(np.max(ts),c='grey',alpha=0.5)
    mainPlot.axhline(0,c='grey',alpha=0.5)
    mainPlot.set_xlim(fullTs[0],fullTs[-1])
    mainPlot.set_title('Full Lightcurve',fontsize=12)

    trans=np.flatnonzero(np.abs(ts)<period/8)
    nonTrans=np.flatnonzero(np.abs(ts)>period/8)

    average = np.average(fs[nonTrans], weights=1/es[nonTrans])
    variance = np.sqrt(np.average((fs[nonTrans]-average)**2, weights=1/es[nonTrans]**2))
    fullAverage = np.average(fs, weights=1/es)
    fullVariance = np.sqrt(np.average((fs-fullAverage)**2, weights=1/es**2))
    #notTransits=np.flatnonzero(np.abs(fs-average)<variance)

    ootPlot=plt.subplot(thisGrid[1,0:2])
    ootPlot.errorbar(fullTs,1e6*(fullFs),yerr=1e6*fullEs,fmt="none",lw=1,c='grey',zorder=2)
    ootPlot.scatter(fullTs,1e6*(fullFs),s=20,edgecolors='k',facecolors=c,zorder=3)
    ootPlot.axvline(np.min(ts),c='grey',alpha=0.5)
    ootPlot.axvline(np.max(ts),c='grey',alpha=0.5)
    ootPlot.set_xlim(fullTs[0],fullTs[-1])
    try:
        ootPlot.set_ylim(1e6*(np.min(fs[nonTrans])-1.5*variance),1e6*(np.max(fs[nonTrans])+1.5*variance))
    except ValueError:
        #ootPlot.set_ylim(1e6*(np.min(fs[nonTrans])),1e6*(np.max(fs[nonTrans])))
        pass

    ootPlot.axhline(0,c='grey',alpha=0.5)
    ootPlot.set_title('Out of Transit',fontsize=12)

    transitPlot=plt.subplot(thisGrid[0,2])
    transitPlot.errorbar(ts[trans],1e6*(fs[trans]),yerr=1e6*es[trans],fmt="none",lw=1,c='grey',zorder=2)
    transitPlot.scatter(ts[trans],1e6*(fs[trans]),s=20,edgecolors='k',facecolors=c,zorder=3)
    transitPlot.axhline(1,c='grey',alpha=0.5)
    transitPlot.set_xlim(-period/8,period/8)
    transitPlot.set_title(r'Transit ($\Phi=0$)',fontsize=12)
    try:
        transitPlot.set_ylim(1e6*(np.min(fs)-1.5*fullVariance),1e6*(np.max(fs)+1.5*fullVariance))
    except ValueError:
        #transitPlot.set_ylim(1e6*(np.min(fs)),1e6*(np.max(fs)))
        pass
    secondaryPlot=plt.subplot(thisGrid[1,2])
    second=np.flatnonzero(np.abs(fullTs+(period/2))<period/8)
    #secondaryPlot.errorbar(fullTs[second],1e6*(fullFs[second]),yerr=1e6*fullEs[second],fmt="none",markersize='2',lw=1,c='darkgrey')
    secondaryPlot.errorbar(fullTs[second],1e6*(fullFs[second]),yerr=1e6*fullEs[second],fmt="none",lw=1,c='grey',zorder=2)
    secondaryPlot.scatter(fullTs[second],1e6*(fullFs[second]),s=20,edgecolors='k',facecolors=c,zorder=3)
    secondaryPlot.axvline(np.min(ts),c='grey',alpha=0.5)
    secondaryPlot.axhline(1,c='grey',alpha=0.5)
    secondaryPlot.set_title(r'Secondary ($\Phi=\pi$)',fontsize=12)
    try:
        secondaryPlot.set_ylim(1e6*(np.min(fullFs[second])-1.5*variance),1e6*(np.max(fullFs[second])+1.5*variance))
    except ValueError:
        #secondaryPlot.set_ylim(1e6*(np.min(fullFs[second])),1e6*(np.max(fullFs[second])))
        pass
    
    secondaryPlot.set_xlim(-(5/8)*period,-(3/8)*period)
    return offset

def plotModel(thisGrid,ts,fs,period=-1,c='k',offset=0,alpha=0.5,lw=2,label='None'):
    if period==-1:
        period=np.max(ts)-np.min(ts)
    offset=ts[np.argmin(fs)]
    ts=(ts-offset+period/2)%period - period/2

    sort=np.argsort(ts)
    nTs=sort.size

    ts=ts[sort]
    fs=fs[sort]

    if ts[-1]-ts[0]<1.1*period: # extending the plot either side of one period
        fullTs=np.hstack([ts[-32:]-period,ts,period+ts[:32]])
        fullFs=np.hstack([fs[-32:],fs,fs[:32]])
    else:
        fullTs=ts
        fullFs=fs

    ootPlot=plt.subplot(thisGrid[1,0:2])
    ootPlot.plot(fullTs,1e6*(fullFs),c=c,alpha=alpha,lw=lw)

    transitPlot=plt.subplot(thisGrid[0,2])
    transitPlot.plot(fullTs,1e6*(fullFs),c=c,alpha=alpha,lw=lw)

    secondaryPlot=plt.subplot(thisGrid[1,2])
    secondaryPlot.plot(fullTs,1e6*(fullFs),c=c,alpha=alpha,lw=lw)

    mainPlot=plt.subplot(thisGrid[0,0:2])
    mainPlot.plot(fullTs,1e6*(fullFs),c=c,alpha=alpha,lw=lw,label=label)

def makeFig():
    dataFig=plt.figure(figsize=(12,8))
    dataGrid=mpl.gridspec.GridSpec(2,3)
    dataFig.text(0.5, 0.04, 'Time (days)', ha='center')
    dataFig.text(0.04, 0.5, 'Flux (ppm)', va='center', rotation='vertical')
    return dataFig,dataGrid
