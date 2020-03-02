'''
Copyright (c) 2020-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Brent Williams

'''
import numpy as np
import matplotlib.pyplot as plt
import mpl_scatter_density

def scatter_density(x_in, y_in, bin_edges=100, cmap='jet'):
    """
    plot a 2d histogram with 50%-ile and |68|%-tile
    """
    if len(y_in)<=0:
        # dont try to plot empty arrays
        return
    #exclude outliers
    msk = np.abs(y_in)<np.percentile(np.abs(y_in),95)
    x = x_in[msk]
    y = y_in[msk]

    h, binsy, binsx = np.histogram2d(y, x, bins=bin_edges)
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
    for start, stop in zip(binsx[:-1],binsx[1:]):
        this_y = y_in[np.logical_and(x_in>start, x_in<=stop)]
        if np.size(this_y)>50:
            p50.append(np.percentile(this_y, 50))
            p68.append(np.percentile(np.abs(this_y), 68))
        else:
            p50.append(np.nan)
            p68.append(np.nan)
    p50 = np.array(p50)
    p68 = np.array(p68)
    binsx_cen = binsx[:-1] + (binsx[1]-binsx[0]) / 2.0
    plt.plot(binsx_cen, p50,'--k')
    plt.plot(binsx_cen, p68,'--g')
    plt.plot(binsx_cen, -p68,'--g')
    plt.legend(['50%-ile', '|68|%-ile'])
    
