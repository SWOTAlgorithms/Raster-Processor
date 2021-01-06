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
from collections import Counter

def nanmean_masked(value):
    # TODO: this is a hack to fix a bug in nanmean with fully masked arrays...
    unmasked_value = value[~value.mask]
    if len(unmasked_value)!=0:
        out = np.nanmean(unmasked_value)
    else:
        out = np.nan
    return out

def get_top_sources(x_in, y_in, sources_in, bins_x, bins_y, n=3):
    """
    gets the n top sources for data in each bin
    assumes bin edges are in increasing order
    """
    mapped_data = []
    out_vals = []
    out_pcts = []
    for i in range(0, len(bins_x)):
        mapped_data.append([])
        out_vals.append([])
        out_pcts.append([])
        for j in range(0, len(bins_y)):
            mapped_data[i].append([])
            out_vals[i].append([])
            out_pcts[i].append([])

    for idx in range(0, len(x_in)):
        bin_x = np.searchsorted(bins_x, x_in[idx])-1
        bin_y = np.searchsorted(bins_y, y_in[idx])-1
        mapped_data[bin_x][bin_y].append(sources_in[idx])

    for i in range(0, len(bins_x)):
        for j in range(0, len(bins_y)):
            bin_data = mapped_data[i][j]
            c = Counter(bin_data)
            common_vals = c.most_common(n)
            out_vals[i][j] = [this_tup[0] for this_tup in common_vals]
            counts = [this_tup[1] for this_tup in common_vals]
            out_pcts[i][j] = [100*this_count/len(bin_data) for this_count in counts]
    return out_vals, out_pcts

def make_format(bins_x, bins_y, top_sources, top_source_pcts):
    """
    make a cursor string format function with the top data sources
    """
    def format_coord(x, y):
        row = np.searchsorted(bins_x, x)-1
        col = np.searchsorted(bins_y, y)-1
        format_str = 'x={0:1.4f}, y={1:1.4f}'.format(x, y)

        if row >= 0 and row < len(top_sources) \
           and col >= 0 and col < len(top_sources[0]):
            ts = top_sources[row][col]
            tsp = top_source_pcts[row][col]
            if len(ts)!=0:
                format_str = '{0}, sources='.format(format_str)
                for idx in range(0, len(ts)):
                    format_str = '{0}{1}:{2:1.1f}%  '.format(format_str, ts[idx], tsp[idx])
        return format_str
    return format_coord

def scatter_density(x_in, y_in,
        uncert=None, bin_edges=100, source=None, cmap='jet', exclude_outliers=True):
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

    # aggregate the data sources if provided
    if source is not None:
        top_sources, top_source_pcts = get_top_sources(x, y, source,
                                                       binsx, binsy, n=3)

    extent = [np.min(binsx), np.max(binsx),
              np.min(binsy),np.max(binsy)]
    fig, ax = plt.subplots()
    im = plt.imshow(10*np.log10(h),
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
                unc.append(nanmean_masked(uncert[msk]))
            else:
                unc_med.append(np.nan)
                unc.append(np.nan)
    metrics = compute_metrics_from_error(y)
    p50 = np.array(p50)
    p68 = np.array(p68)
    unc = np.array(unc)
    unc_med = np.array(unc_med)

    binsx_cen = binsx[:-1] + (binsx[1]-binsx[0]) / 2.0
    ax.plot(binsx_cen, p50,'--k')
    ax.plot(binsx_cen, p68,'x-g')
    leg_text = ['50%-ile   '+'(Tot: %2.4f)'%(metrics['50_pct']),
        '|68|%-ile '+'(Tot: %2.4f)'%(metrics['|68_pct|']),]
    if uncert is not None:
        #ax.plot(binsx_cen, unc,'--b')
        ax.plot(binsx_cen, unc_med,'-b')
        ax.plot(binsx_cen, -unc_med,'-b')
        #ax.plot(binsx_cen, -unc,'--b')
        #leg_text.append('uncert, mean')
        leg_text.append('uncert, median')


    ax.plot(binsx_cen, -p68,'x-g')
    ax.set_xlim((extent[0],extent[1]))
    ax.set_ylim((extent[2],extent[3]))
    ax.legend(leg_text)
    ax.grid()

    # Add sources to cursor string format
    if source is not None:
        ax.format_coord = make_format(binsx, binsy, top_sources, top_source_pcts)
