'''
Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

'''

import numpy as np

def metrics_fit(x, y, poly=3, pts=25):
    z = np.polyfit(x, y, poly)
    f = np.poly1d(z)
    x_new = np.linspace(np.nanmin(x),
                        np.nanmax(x), pts)
    y_new = f(x_new)
    return x_new, y_new

def get_passfail():
    passfail = {
        '|wse_e_68_pct|': [0.10, 1],
        '|wse_e/wse_u_68_pct|': [1, 2],
        '|a_%e/a_%u_68_pct|': [1, 2]
    }
    return passfail

def std_mask(data, m=1):
    return abs(data - np.mean(data)) < m * np.std(data)

def weighted_mean(data, weights=None):
    return np.nansum(data * weights) / np.nansum(weights)

def weighted_percentile(data, pct, weights=None):
    if weights is None:
        weights = np.ones(len(data))
    sorter = np.argsort(data)
    values = data[sorter]
    weights = weights[sorter]
    weighted_pcts = np.nancumsum(weights) - 0.5 * weights
    weighted_pcts /= np.nansum(weights)
    return np.interp(pct/100, weighted_pcts, data)

def weighted_std(data, weights=None):
    mean = weighted_mean(data, weights)
    variance = weighted_mean((data-mean)**2, weights)
    return np.sqrt(variance)

def compute_metrics_from_error(err_array, weights=None, mask=None):
    error_metrics = {}
    if isinstance(err_array, np.ma.MaskedArray):
        err_array = err_array.filled(np.nan)

    if mask is not None:
        mask = np.logical_and(mask, ~np.isnan(err_array))
    else:
        mask = ~np.isnan(err_array)

    err_array = err_array[mask]
    if len(err_array) == 0:
        error_metrics['mean'] = np.nan
        error_metrics['std'] = np.nan
        error_metrics['|68_pct|'] = np.nan
        error_metrics['50_pct'] = np.nan
        return error_metrics

    if weights is None:
        error_metrics['mean'] = np.nanmean(err_array)
        error_metrics['std'] = np.nanstd(err_array)
        error_metrics['|68_pct|'] = np.nanpercentile(abs(err_array), 68)
        error_metrics['50_pct'] = np.nanpercentile(err_array, 50)
    else:
        weights = weights[mask]
        error_metrics['mean'] = weighted_mean(err_array, weights)
        error_metrics['std'] = weighted_std(err_array, weights)
        error_metrics['|68_pct|'] = weighted_percentile(abs(err_array), 68,
                                                        weights=weights)
        error_metrics['50_pct'] = weighted_percentile(err_array, 50,
                                                      weights=weights)

    return error_metrics

def nanmean_masked(value):
    # TODO: this is a hack to fix a bug in nanmean with fully masked arrays...
    unmasked_value = value[~value.mask]
    if len(unmasked_value)!=0:
        out = np.nanmean(unmasked_value)
    else:
        out = np.nan
    return out
