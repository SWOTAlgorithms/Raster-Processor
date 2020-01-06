#!/usr/bin/env python
'''
Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

'''

import os
import glob
import scipy
import argparse
import numpy as np
import raster_products
import matplotlib.pyplot as plt
import SWOTRiver.analysis.tabley

from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pixc_raster', help='raster made from pixel cloud', type=str)
    parser.add_argument('gdem_raster', help='raster made from truth gdem', type=str)
    parser.add_argument('--basedir', help='base directory to look for raster files to analyze', type=str, default=None)
    parser.add_argument('-d', '--dark_water_thresh', help='Dark water fraction threshold for extra metric', type=float, default=0.5)
    parser.add_argument('-p', '--min_pixels', help='Minimum number of pixels to use as valid data', type=int, default=None)
    parser.add_argument('-s', '--sort_key', help='Key to use when sorting output table', type=str, default=None)

    args = vars(parser.parse_args())

    basedir = args['basedir']
    metrics = []
    if  basedir is not None:
        # log all plots in subdirectory of wherever the data is
        raster_list = glob.glob(os.path.join(basedir, '**', args['pixc_raster']), recursive=True)
        for pixc_name in raster_list:
            # check that it has a corresponding gdem_raster
            p = Path(pixc_name)
            basepath = p.parents[1]
            scene = p.parts[-4]
            gdem_name = os.path.join(basepath, args['gdem_raster'])

            if os.path.isfile(gdem_name):
                # call the function to do the work
                tile_metrics = load_data(pixc_name, gdem_name,
                                         scene=scene, min_pixels=args['min_pixels'])
                metrics.append(tile_metrics)
    else:
        tile_metrics = load_data(args['pixc_raster'], args['gdem_raster'],
                                 min_pixels=args['min_pixels'])
        metrics.append(tile_metrics)
    
    print_metrics(metrics, dark_thresh=args['dark_water_thresh'], sort_key=args['sort_key'])


def load_data(
        pixc_file, gdem_file, scene='', min_pixels=None):
    '''
    load reaches from a particular tile, compute metrics,
    and accumulate the data, truth and metrics (if input)
    '''

    truth_tmp = raster_products.Raster.from_ncfile(gdem_file)
    data_tmp = raster_products.Raster.from_ncfile(pixc_file)

    tile_metrics = {}
    tile_metrics['scene'] = str(scene)
    tile_metrics['cycle'] = str(data_tmp.cycle_number)
    tile_metrics['pass'] = str(data_tmp.pass_number)
    tile_metrics['tiles'] = str(data_tmp.tile_numbers)
    print('Loading data for scene: {}, cycle: {}, pass: {}, tiles: {}'.format(
        tile_metrics['scene'],
        tile_metrics['cycle'],
        tile_metrics['pass'],
        tile_metrics['tiles']))

    # Handle potentially empty files
    if data_tmp['height'].size==0 or truth_tmp['height'].size==0:
        tile_metrics['height_err'] = np.array([np.nan])
        tile_metrics['area_perc_err'] = np.array([np.nan])
        tile_metrics['cross_track'] = np.array([np.nan])
        tile_metrics['dark_frac'] = np.array([np.nan])
        tile_metrics['num_pixc_px'] = np.array([np.nan])
        tile_metrics['total_px'] = truth_tmp['height'].count() + data_tmp['height'].count()
        tile_metrics['common_px'] = 0
        tile_metrics['uncommon_px_truth'] = truth_tmp['height'].count()
        tile_metrics['uncommon_px_data'] = data_tmp['height'].count()
    else:
        height_err = data_tmp['height'] - truth_tmp['height']
        area_perc_err = (data_tmp['water_area'] - truth_tmp['water_area']) \
                        / truth_tmp['water_area'] * 100
        total_mask = np.logical_or(~truth_tmp['height'].mask, ~data_tmp['height'].mask)
        common_mask = np.logical_and(~truth_tmp['height'].mask, ~data_tmp['height'].mask)
        truth_not_in_data_mask = np.logical_and(~truth_tmp['height'].mask, data_tmp['height'].mask)
        data_not_in_truth_mask = np.logical_and(~data_tmp['height'].mask, truth_tmp['height'].mask)
        
        if min_pixels is not None:
            common_mask = np.logical_and(common_mask,
                                         data_tmp['num_pixels'] >= min_pixels)
            truth_not_in_data_mask = np.logical_and(truth_not_in_data_mask,
                                                    data_tmp['num_pixels'] >= min_pixels)
            data_not_in_truth_mask = np.logical_and(data_not_in_truth_mask,
                                                    data_tmp['num_pixels'] >= min_pixels)

        tile_metrics['height_err'] = height_err[common_mask]
        tile_metrics['area_perc_err'] = area_perc_err[common_mask]
        tile_metrics['cross_track'] = data_tmp['cross_track'][common_mask]
        tile_metrics['dark_frac'] = data_tmp['dark_frac'][common_mask]
        tile_metrics['num_pixc_px'] = data_tmp['num_pixels'][common_mask]
        tile_metrics['total_px'] = np.count_nonzero(total_mask)
        tile_metrics['common_px'] = np.count_nonzero(common_mask)
        tile_metrics['uncommon_px_truth'] = np.count_nonzero(truth_not_in_data_mask)
        tile_metrics['uncommon_px_data'] = np.count_nonzero(data_not_in_truth_mask)
    
    return tile_metrics


def print_metrics(metrics, dark_thresh=None, sort_key=None):    
    # Tile by Tile metrics
    tile_table = {'h_e_mean':[],
                  'h_e_std':[],
                  'h_e_68_pct':[],
                  'h_e_50_pct':[],
                  'a_%e_mean':[],
                  'a_%e_std':[],
                  'a_%e_68_pct':[],
                  'a_%e_50_pct':[],
                  'total_px':[],
                  'common_px_%':[],
                  'uncommon_px_truth_%':[],
                  'uncommon_px_data_%':[],
                  'scene':[],
                  'cycle':[],
                  'pass':[],
                  'tiles':[],}

    for tile_metrics in metrics:
        height_err_metrics = compute_metrics_from_error(tile_metrics['height_err'])
        area_perc_err_metrics = compute_metrics_from_error(tile_metrics['area_perc_err'])
        
        # Add data to table
        tile_table['h_e_mean'].append(height_err_metrics['mean'])
        tile_table['h_e_std'].append(height_err_metrics['std'])
        tile_table['h_e_68_pct'].append(height_err_metrics['68_pct'])
        tile_table['h_e_50_pct'].append(height_err_metrics['50_pct'])
        tile_table['a_%e_mean'].append(area_perc_err_metrics['mean'])
        tile_table['a_%e_std'].append(area_perc_err_metrics['std'])
        tile_table['a_%e_68_pct'].append(area_perc_err_metrics['68_pct'])
        tile_table['a_%e_50_pct'].append(area_perc_err_metrics['50_pct'])
        tile_table['total_px'].append(tile_metrics['total_px'])
        if tile_metrics['total_px'] > 0:
            tile_table['common_px_%'].append(tile_metrics['common_px']/tile_metrics['total_px'] * 100)
            tile_table['uncommon_px_truth_%'].append(tile_metrics['uncommon_px_truth']/tile_metrics['total_px'] * 100)
            tile_table['uncommon_px_data_%'].append(tile_metrics['uncommon_px_data']/tile_metrics['total_px'] * 100)
        else:
            tile_table['common_px_%'].append(0)
            tile_table['uncommon_px_truth_%'].append(0)
            tile_table['uncommon_px_data_%'].append(0)
        tile_table['scene'].append(tile_metrics['scene'])
        tile_table['cycle'].append(tile_metrics['cycle'])
        tile_table['pass'].append(tile_metrics['pass'])
        tile_table['tiles'].append(tile_metrics['tiles'])
          
    # Sort the metrics by height error
    if sort_key is not None:
        sort_idx = np.argsort(tile_table[sort_key])
        for key in tile_table:
            tile_table[key] = np.array(tile_table[key])[sort_idx]

    all_dark_frac = np.concatenate([tile_metrics['dark_frac'] for tile_metrics in metrics])
    all_height_err = np.concatenate([tile_metrics['height_err'] for tile_metrics in metrics])
    all_area_perc_err = np.concatenate([tile_metrics['area_perc_err'] for tile_metrics in metrics])
    all_cross_track = np.concatenate([tile_metrics['cross_track'] for tile_metrics in metrics])
    all_pixc_px = np.concatenate([tile_metrics['num_pixc_px'] for tile_metrics in metrics])
    total_px = [tile_metrics['total_px'] for tile_metrics in metrics]
    common_px = [tile_metrics['common_px'] for tile_metrics in metrics]
    uncommon_px_truth = [tile_metrics['uncommon_px_truth'] for tile_metrics in metrics]
    uncommon_px_data = [tile_metrics['uncommon_px_data'] for tile_metrics in metrics]

    print('Tile metrics:')
    SWOTRiver.analysis.tabley.print_table(tile_table, precision=5)

    # Global metrics
    height_err_metrics = compute_metrics_from_error(all_height_err)
    area_perc_err_metrics = compute_metrics_from_error(all_area_perc_err)
    total_px_count = np.sum(total_px)
    common_px_pct = np.sum(common_px)/total_px_count * 100
    uncommon_px_truth_pct = np.sum(uncommon_px_truth)/total_px_count * 100
    uncommon_px_data_pct = np.sum(uncommon_px_data)/total_px_count * 100

    global_table = {}
    global_table['h_e_mean'] = [height_err_metrics['mean']]
    global_table['h_e_std'] = [height_err_metrics['std']]
    global_table['h_e_68_pct'] = [height_err_metrics['68_pct']]
    global_table['h_e_50_pct'] = [height_err_metrics['50_pct']]
    global_table['a_%e_mean'] = [area_perc_err_metrics['mean']]
    global_table['a_%e_std'] = [area_perc_err_metrics['std']]
    global_table['a_%e_68_pct'] = [area_perc_err_metrics['68_pct']]
    global_table['a_%e_50_pct'] = [area_perc_err_metrics['50_pct']]
    global_table['total_px'] = [total_px_count]
    global_table['common_px_%'] = [common_px_pct]
    global_table['uncommon_px_truth_%'] = [uncommon_px_truth_pct]    
    global_table['uncommon_px_data_%'] = [uncommon_px_data_pct]    

    print('Global metrics:')        
    SWOTRiver.analysis.tabley.print_table(global_table, precision=5)

    # Global metrics without darkwater
    if dark_thresh is not None:
        all_bright_mask = np.concatenate([tile_metrics['dark_frac'] < dark_thresh for tile_metrics in metrics])
        height_err_metrics = compute_metrics_from_error(all_height_err[all_bright_mask])
        area_perc_err_metrics = compute_metrics_from_error(all_area_perc_err[all_bright_mask])

        global_table_nodark = {}
        global_table_nodark['h_e_mean'] = [height_err_metrics['mean']]
        global_table_nodark['h_e_std'] = [height_err_metrics['std']]
        global_table_nodark['h_e_68_pct'] = [height_err_metrics['68_pct']]
        global_table_nodark['h_e_50_pct'] = [height_err_metrics['50_pct']]
        global_table_nodark['a_%e_mean'] = [area_perc_err_metrics['mean']]
        global_table_nodark['a_%e_std'] = [area_perc_err_metrics['std']]
        global_table_nodark['a_%e_68_pct'] = [area_perc_err_metrics['68_pct']]
        global_table_nodark['a_%e_50_pct'] = [area_perc_err_metrics['50_pct']]

        print('Global metrics (excluding pixels with dark water area over {}%):'.format(dark_thresh*100))
        SWOTRiver.analysis.tabley.print_table(global_table_nodark, precision=5)
 
    metrics_to_plot = {'Height Error (m)':all_height_err,
                       'Area Percent Area (%)':all_area_perc_err,
                       'Num Pixels':all_pixc_px}

    metrics_to_plot_against = {'Cross Track (m)':all_cross_track, 
                               'Num Pixels':all_pixc_px, 
                               'Dark Fraction (%)':all_dark_frac*100}   
    
    plot_metrics(metrics_to_plot, metrics_to_plot_against)


def plot_metrics(metrics_to_plot, metrics_to_plot_against, poly=2):
    for y_key in metrics_to_plot:
        for x_key in metrics_to_plot_against:
            if x_key != y_key:
                mask = np.logical_and(~np.isnan(metrics_to_plot[y_key]), 
                                      ~np.isnan(metrics_to_plot_against[x_key]))
                this_y_data = metrics_to_plot[y_key][mask]
                this_x_data = metrics_to_plot_against[x_key][mask] 
                plt.figure()
                plt.title('{} vs. {}'.format(y_key, x_key))
                plt.xlabel(x_key)
                plt.ylabel(y_key)
                plt.scatter(this_x_data, this_y_data,
                            marker='o', s=1)
                x_new, y_new = metrics_fit(this_x_data, this_y_data,
                                           poly=poly, pts=25)
                plt.plot(x_new, y_new, 'r--')
                sig_mask = std_mask(this_y_data, 1)
                x_new, y_new = metrics_fit(this_x_data[sig_mask],
                                           this_y_data[sig_mask],
                                           poly=poly, pts=25)
                plt.plot(x_new, y_new, 'g--')
                plt.legend(['fit', '1sig fit', 'data'])
                
    plt.show()

def metrics_fit(x, y, poly=3, pts=25):
    z = np.polyfit(x, y, poly)
    f = np.poly1d(z)
    x_new = np.linspace(np.nanmin(x),
                        np.nanmax(x), pts)
    y_new = f(x_new)
    return x_new, y_new

def std_mask(data, m=1):
    return abs(data - np.mean(data)) < m * np.std(data)

def compute_metrics_from_error(err_array):
    error_metrics = {}
    if isinstance(err_array, np.ma.MaskedArray):
        error_metrics['mean'] = np.nanmean(err_array.filled(np.nan))
        error_metrics['std'] = np.nanstd(err_array.filled(np.nan))
        error_metrics['68_pct'] = np.nanpercentile(abs(err_array.filled(np.nan)), 68)
        error_metrics['50_pct'] = np.nanpercentile(abs(err_array.filled(np.nan)), 50)
    else:
        error_metrics['mean'] = np.nanmean(err_array)
        error_metrics['std'] = np.nanstd(err_array)
        error_metrics['68_pct'] = np.nanpercentile(abs(err_array), 68)
        error_metrics['50_pct'] = np.nanpercentile(abs(err_array), 50)

    return error_metrics


if __name__ == "__main__":
    main()
