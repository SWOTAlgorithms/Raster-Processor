'''
Copyright (c) 2020-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Brent Williams

'''
import numpy as np
import matplotlib.pyplot as plt
import mpl_scatter_density
from metrics import compute_metrics_from_error

def scatter_density(x_in, y_in,
        uncert=None, bin_edges=100, cmap='jet', exclude_outliers=True):
    """
    plot a 2d histogram with 50%-ile and |68|%-tile
    """
    if len(y_in)<=0:
        # dont try to plot empty arrays
        return
    x = x_in
    y = y_in
    # setup customized limits
    if isinstance(bin_edges, int):
        binsy = np.linspace(np.min(y_in), np.max(y_in), bin_edges)
        binsx = np.linspace(np.min(x_in), np.max(x_in), bin_edges)
    else:
        binsy, binsx = bin_edges
    if isinstance(binsy, int):
        binsy = np.linspace(np.min(y_in), np.max(y_in), binsy)
    if isinstance(binsx, int):
        binsx = np.linspace(np.min(x_in), np.max(x_in), binsx)
    if exclude_outliers:
        # this only excludes outliers for choosing the y-bin extents
        msk = np.abs(y_in)<np.percentile(np.abs(y_in),95)
        num_binsy = len(binsy)
        binsy = binsy = np.linspace(
            np.min(y_in[msk]), np.max(y_in[msk]), num_binsy)
    # generate the 2d histogram and plot
    h, by, bx = np.histogram2d(y, x, bins=(binsy, binsx))
    extent = [np.min(binsx), np.max(binsx),
              np.min(binsy),np.max(binsy)]
    plt.figure()
    plt.imshow(10*np.log10(h),
               aspect='auto', extent=extent, origin='lower',
               cmap=cmap)
    plt.colorbar(label='dB num points')
    
    # plot the 50%-ile and |68|%-ile
    p50 = []
    p68 = []
    unc = []
    unc_med = []
    for start, stop in zip(binsx[:-1],binsx[1:]):
        msk = np.logical_and(x>=start, x<=stop)
        this_y = y[msk]
        metrics = compute_metrics_from_error(this_y)
        p50.append(metrics['50_pct'])
        p68.append(metrics['|68_pct|'])
        if uncert is not None:
            if len(uncert[msk])>0:
                unc_med.append(np.nanmedian(uncert[msk]))
                unc.append(np.nanmean(uncert[msk]))
            else:
                unc_med.append(np.nan)
                unc.append(np.nan)
    metrics = compute_metrics_from_error(y)
    p50 = np.array(p50)
    p68 = np.array(p68)
    unc = np.array(unc)
    unc_med = np.array(unc_med)

    binsx_cen = binsx[:-1] + (binsx[1]-binsx[0]) / 2.0
    plt.plot(binsx_cen, p50,'--k')
    plt.plot(binsx_cen, p68,'x-g')
    leg_text = ['50%-ile   '+'(Tot: %2.4f)'%(metrics['50_pct']),
        '|68|%-ile '+'(Tot: %2.4f)'%(metrics['|68_pct|']),]
    if uncert is not None:
        #plt.plot(binsx_cen, unc,'--b')
        plt.plot(binsx_cen, unc_med,'-b')
        plt.plot(binsx_cen, -unc_med,'-b')
        #plt.plot(binsx_cen, -unc,'--b')
        #leg_text.append('uncert, mean')
        leg_text.append('uncert, median')

        
    plt.plot(binsx_cen, -p68,'x-g')
    plt.xlim((extent[0],extent[1]))
    plt.ylim((extent[2],extent[3]))
    plt.legend(leg_text)
    plt.grid()
