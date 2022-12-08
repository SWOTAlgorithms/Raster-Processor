#!/usr/bin/env python
'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

Plot raster statistics for single raster or bulk directory structure

Assumed bulk directory structure:
sim_scene_base_directory
└── tile_base_directory
    └── slc_base_directory
        ├── pixc_base_directory
        │   ├── pixc_systematic_errors_base_directory
        │   │   └── proc_raster_systematic_errors_base_directory
        │   └── proc_raster_base_directory
        └── truth_raster_base_directory
'''

import os
import glob
import argparse
import numpy as np
import SWOTRaster.products
import matplotlib.pyplot as plt
import SWOTRiver.analysis.tabley

from pathlib import Path
from SWOTRaster.analysis.metrics import *
from SWOTRaster.analysis.scatter_density import scatter_density
from SWOTWater.products.product import MutableProduct

import warnings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proc_raster', type=str, default=None,
                        help='processed raster file (or basename)')
    parser.add_argument('truth_raster', type=str, default=None,
                        help='truth raster file (or basename)')
    parser.add_argument('--basedir', type=str, default=None,
                        help='base directory of processing')
    parser.add_argument('-sb', '--slc_basename', type=str, default=None,
                        help='slc directory basename')
    parser.add_argument('-pb', '--pixc_basename', type=str, default=None,
                        help='pixc directory basename')
    parser.add_argument('-eb','--pixc_errors_basename', type=str, default=None,
                        help = "pixc systematic errors basename")
    parser.add_argument('-df', '--dark_frac_thresh', type=float, default=None,
                        help='dark water fraction threshold for extra metric')
    parser.add_argument('-wf', '--water_frac_thresh', type=float, default=None,
                        help='water fraction threshold to use as valid data')
    parser.add_argument('-hu', '--wse_uncert_thresh', type=float, default=None,
                        help='wse uncertainty threshold to use as valid data')
    parser.add_argument('-au', '--area_uncert_thresh', type=float, default=None,
                        help='area uncertainty threshold to use as valid data')
    parser.add_argument('-ct', '--cross_track_bounds', type=float, default=None,
                        help='crosstrack bounds to use as valid data', nargs=2)
    parser.add_argument('-mwp', '--min_wse_pixels', type=int, default=None,
                        help='minimum number of wse pixels to use as valid data')
    parser.add_argument('-map', '--min_area_pixels', type=int, default=None,
                        help='minimum number of area pixels to use as valid data')
    parser.add_argument('-w', '--weighted', action='store_true',
                        help='flag to enable weighting of wse and area ' \
                        'metrics by uncertainties')
    parser.add_argument('-e', '--exclude_scenes', default=[], nargs='+',
                        help='list of sim scenes to exclude')
    parser.add_argument('-o', '--outdir', type=str, default=None,
                        help='list of sim scenes to exclude')
    parser.add_argument('--scatter_plot', action='store_true',
                        help='flag for plotting old scatterplots')
    args = vars(parser.parse_args())

    metrics = []
    if args['basedir'] is not None:
        if args['slc_basename'] is None or args['pixc_basename'] is None:
            print('Must specify at least slc_basename and pixc_basename '
                  + 'if aggregating stats')
            return

        if args['pixc_errors_basename'] is not None:
            proc_raster_list = glob.glob(os.path.join(
                args['basedir'], '*', '*', args['slc_basename'], args['pixc_basename'],
                args['pixc_errors_basename'], args['proc_raster']))
        else:
            proc_raster_list = glob.glob(os.path.join(
                args['basedir'], '*', '*', args['slc_basename'], args['pixc_basename'],
                args['proc_raster']))

        # get only unique rasters
        proc_raster_list = np.unique([os.path.realpath(filename)
                                      for filename in proc_raster_list])

        # if proc_raster input is a basename, get the actual raster
        proc_raster_list = [os.path.join(proc_raster, 'raster_data', 'raster.nc')
                            if os.path.isdir(proc_raster) else proc_raster
                            for proc_raster in proc_raster_list]

        if args['pixc_errors_basename'] is not None:
            truth_raster_list = \
                [os.path.join(*Path(proc_raster).parts[:-5], args['truth_raster'])
                 for proc_raster in proc_raster_list]
        else:
            truth_raster_list = \
                [os.path.join(*Path(proc_raster).parts[:-4], args['truth_raster'])
                 for proc_raster in proc_raster_list]

        # if truth_raster input is a basename, get the actual raster
        truth_raster_list = [os.path.join(truth_raster, 'raster_data', 'raster.nc')
                             if os.path.isdir(truth_raster) else truth_raster
                             for truth_raster in truth_raster_list]

        for proc_raster, truth_raster in zip(proc_raster_list, truth_raster_list):
            if os.path.isfile(proc_raster) and os.path.isfile(truth_raster):
                if args['pixc_errors_basename'] is not None:
                    sim_scene = Path(proc_raster).parts[-8]
                else:
                    sim_scene = Path(proc_raster).parts[-7]

                if sim_scene in args['exclude_scenes']:
                    print('Not analyzing sim scene: {}'.format(sim_scene))
                    continue

                proc_raster_obj = SWOTRaster.products.RasterUTM.from_ncfile(proc_raster)
                sim_tile = proc_raster_obj.tile_names
                if isinstance(sim_tile, list):
                    sim_tile = '_'.join(sim_tile)
                print('Aggregating stats for {} - {}'.format(sim_scene, sim_tile))

                # call the function to do the work
                tile_metrics = load_data(
                    proc_raster, truth_raster, sim_scene=sim_scene,
                    dark_frac_thresh=args['dark_frac_thresh'],
                    water_frac_thresh=args['water_frac_thresh'],
                    wse_uncert_thresh=args['wse_uncert_thresh'],
                    area_uncert_thresh=args['area_uncert_thresh'],
                    cross_track_bounds=args['cross_track_bounds'],
                    min_wse_pixels=args['min_wse_pixels'],
                    min_area_pixels=args['min_area_pixels'])
                metrics.append(tile_metrics)
    else:
        # inputs can be either raster files, or basenames
        proc_raster = args['proc_raster']
        truth_raster = args['truth_raster']
        if os.path.isdir(proc_raster):
            proc_raster = os.path.join(proc_raster, 'raster_data', 'raster.nc')
        if os.path.isdir(truth_raster):
            truth_raster = os.path.join(truth_raster, 'raster_data', 'raster.nc')

        # call the function to do the work
        tile_metrics = load_data(
            proc_raster, truth_raster,
            dark_frac_thresh=args['dark_frac_thresh'],
            water_frac_thresh=args['water_frac_thresh'],
            wse_uncert_thresh=args['wse_uncert_thresh'],
            area_uncert_thresh=args['area_uncert_thresh'],
            cross_track_bounds=args['cross_track_bounds'],
            min_wse_pixels=args['min_wse_pixels'],
            min_area_pixels=args['min_area_pixels'])
        metrics.append(tile_metrics)

    preamble = '\033[93m' + 'Accumulating Metrics:' + '\033[00m'
    if args['dark_frac_thresh'] is not None:
        preamble += '\n  \033[93m' + 'Dark frac <= {}'.format(
            args['dark_frac_thresh']) + '\033[00m'
    if args['water_frac_thresh'] is not None:
        preamble += '\n  \033[93m' + 'Water frac >= {}'.format(
            args['water_frac_thresh']) + '\033[00m'
    if args['wse_uncert_thresh'] is not None:
        preamble += '\n  \033[93m' + 'WSE uncert <= {}'.format(
            args['wse_uncert_thresh']) + '\033[00m'
    if args['area_uncert_thresh'] is not None:
        preamble += '\n  \033[93m' + 'Water area uncert <= {}'.format(
            args['area_uncert_thresh']) + '\033[00m'
    if args['cross_track_bounds'] is not None:
        preamble += '\n  \033[93m' + 'Cross track bounds = {}'.format(
            args['cross_track_bounds']) + '\033[00m'
    if args['min_wse_pixels'] is not None:
        preamble += '\n  \033[93m' + 'Min WSE pixels = {}'.format(
            args['min_wse_pixels']) + '\033[00m'
    if args['min_area_pixels'] is not None:
        preamble += '\n  \033[93m' + 'Min water area pixels = {}'.format(
            args['min_area_pixels']) + '\033[00m'

    print(preamble)
    # check if outdir exists, if not create it
    if args['outdir'] is not None:
        if not os.path.exists(args['outdir']):
            os.makedirs(args['outdir'])
    print_metrics(metrics,
                  weighted=args['weighted'],
                  scatter_plot=args['scatter_plot'],
                  outdir=args['outdir'], preamble=preamble)

def load_data(
        proc_raster_file, truth_raster_file, sim_scene='', dark_frac_thresh=None,
        water_frac_thresh=None, wse_uncert_thresh=None, area_uncert_thresh=None,
        cross_track_bounds=None, min_wse_pixels=None, min_area_pixels=None):
    '''
    load reaches from a particular tile, compute metrics,
    and accumulate the data, truth and metrics (if input)
    '''

    truth_tmp = SWOTRaster.products.RasterUTM.from_ncfile(truth_raster_file)
    data_tmp = SWOTRaster.products.RasterUTM.from_ncfile(proc_raster_file)

    tile_metrics = {}
    tile_metrics['sim_scene'] = str(sim_scene)
    tile_metrics['cycle'] = str(data_tmp.cycle_number)
    tile_metrics['tile_names'] = data_tmp.tile_names

    # handle potentially empty files
    if data_tmp['wse'].size==0 or truth_tmp['wse'].size==0:
        tile_metrics['wse_err'] = np.array([np.nan])
        tile_metrics['wse_uncert'] = np.array([np.nan])
        tile_metrics['area_perc_err'] = np.array([np.nan])
        tile_metrics['area_perc_uncert'] = np.array([np.nan])
        tile_metrics['cross_track'] = np.array([np.nan])
        tile_metrics['dark_frac'] = np.array([np.nan])
        tile_metrics['dark_frac_err'] = np.array([np.nan])
        tile_metrics['water_frac'] = np.array([np.nan])
        tile_metrics['water_frac_err'] = np.array([np.nan])
        tile_metrics['n_wse_pix'] = np.array([np.nan])
        tile_metrics['n_water_area_pix'] = np.array([np.nan])
        tile_metrics['total_wse_pix'] = truth_tmp['wse'].count() + data_tmp['wse'].count()
        tile_metrics['common_wse_pix'] = 0
        tile_metrics['uncommon_wse_pix_truth'] = truth_tmp['wse'].count()
        tile_metrics['uncommon_wse_pix_data'] = data_tmp['wse'].count()
        tile_metrics['total_area_pix'] = truth_tmp['water_area'].count() + data_tmp['water_area'].count()
        tile_metrics['common_area_pix'] = 0
        tile_metrics['uncommon_area_pix_truth'] = truth_tmp['water_area'].count()
        tile_metrics['uncommon_area_pix_data'] = data_tmp['water_area'].count()
    else:
        wse_err = data_tmp['wse'] - truth_tmp['wse']
        area_err = data_tmp['water_area'] - truth_tmp['water_area']
        area_perc_err = area_err / truth_tmp['water_area'] * 100
        area_perc_unc = data_tmp['water_area_uncert'] / truth_tmp['water_area'] * 100
        water_frac_err = data_tmp['water_frac'] - truth_tmp['water_frac']
        dark_frac_err = data_tmp['dark_frac'] - truth_tmp['dark_frac']
        wse_total_mask = np.logical_or(~truth_tmp['wse'].mask, ~data_tmp['wse'].mask)
        wse_common_mask = np.logical_and(~truth_tmp['wse'].mask, ~data_tmp['wse'].mask)
        wse_truth_not_in_data_mask = np.logical_and(
            ~truth_tmp['wse'].mask, data_tmp['wse'].mask)
        wse_data_not_in_truth_mask = np.logical_and(
            ~data_tmp['wse'].mask, truth_tmp['wse'].mask)
        area_total_mask = np.logical_or(
            ~truth_tmp['water_area'].mask, ~data_tmp['water_area'].mask)
        area_common_mask = np.logical_and(
            ~truth_tmp['water_area'].mask, ~data_tmp['water_area'].mask)
        area_truth_not_in_data_mask = np.logical_and(
            ~truth_tmp['water_area'].mask, data_tmp['water_area'].mask)
        area_data_not_in_truth_mask = np.logical_and(
            ~data_tmp['water_area'].mask, truth_tmp['water_area'].mask)

        # use additional filters to update masks
        tmp_mask = np.ones_like(wse_common_mask)
        if dark_frac_thresh is not None:
            tmp_mask = np.logical_and(
                tmp_mask, truth_tmp['dark_frac'] <= dark_frac_thresh)

        if water_frac_thresh is not None:
            tmp_mask = np.logical_and(
                tmp_mask, truth_tmp['water_frac'] >= water_frac_thresh)

        if wse_uncert_thresh is not None:
            tmp_mask = np.logical_and(
                tmp_mask, data_tmp['wse_uncert'] <= wse_uncert_thresh)

        if area_uncert_thresh is not None:
            tmp_mask = np.logical_and(
                tmp_mask, data_tmp['area_uncert'] <= area_uncert_thresh)

        if cross_track_bounds is not None:
            tmp_mask = np.logical_and.reduce(
                (tmp_mask,
                 np.abs(truth_tmp['cross_track']) >= min(cross_track_bounds),
                 np.abs(truth_tmp['cross_track']) <= max(cross_track_bounds)))

        if min_wse_pixels is not None:
            tmp_mask = np.logical_and(
                tmp_mask, data_tmp['n_wse_pix'] >= min_wse_pixels)

        if min_area_pixels is not None:
            tmp_mask = np.logical_and(
                tmp_mask, data_tmp['n_water_area_pix'] >= min_area_pixels)

        wse_total_mask = np.logical_and(wse_total_mask, tmp_mask)
        wse_common_mask = np.logical_and(wse_common_mask, tmp_mask)
        wse_truth_not_in_data_mask = np.logical_and(wse_truth_not_in_data_mask,
                                                    tmp_mask)
        wse_data_not_in_truth_mask = np.logical_and(wse_data_not_in_truth_mask,
                                                    tmp_mask)

        area_total_mask = np.logical_and(area_total_mask, tmp_mask)
        area_common_mask = np.logical_and(area_common_mask, tmp_mask)
        area_truth_not_in_data_mask = np.logical_and(area_truth_not_in_data_mask,
                                                     tmp_mask)
        area_data_not_in_truth_mask = np.logical_and(area_data_not_in_truth_mask,
                                                     tmp_mask)

        common_mask = np.logical_or(wse_common_mask, area_common_mask)
        tile_metrics['wse_err'] = wse_err[common_mask]
        tile_metrics['wse_uncert'] = data_tmp['wse_uncert'][common_mask]
        tile_metrics['area_perc_err'] = area_perc_err[common_mask]
        tile_metrics['area_perc_uncert'] = area_perc_unc[common_mask]
        tile_metrics['cross_track'] = truth_tmp['cross_track'][common_mask]
        tile_metrics['dark_frac'] = truth_tmp['dark_frac'][common_mask]
        tile_metrics['dark_frac_err'] = dark_frac_err[common_mask]
        tile_metrics['water_frac'] = truth_tmp['water_frac'][common_mask]
        tile_metrics['water_frac_err'] = water_frac_err[common_mask]
        tile_metrics['n_wse_pix'] = data_tmp['n_wse_pix'][common_mask]
        tile_metrics['n_water_area_pix'] = data_tmp['n_water_area_pix'][common_mask]
        tile_metrics['total_wse_pix'] = np.count_nonzero(wse_total_mask)
        tile_metrics['common_wse_pix'] = np.count_nonzero(wse_common_mask)
        tile_metrics['uncommon_wse_pix_truth'] = np.count_nonzero(
            wse_truth_not_in_data_mask)
        tile_metrics['uncommon_wse_pix_data'] = np.count_nonzero(
            wse_data_not_in_truth_mask)
        tile_metrics['total_area_pix'] = np.count_nonzero(area_total_mask)
        tile_metrics['common_area_pix'] = np.count_nonzero(area_common_mask)
        tile_metrics['uncommon_area_pix_truth'] = np.count_nonzero(
            area_truth_not_in_data_mask)
        tile_metrics['uncommon_area_pix_data'] = np.count_nonzero(
            area_data_not_in_truth_mask)

    return tile_metrics

def print_metrics(metrics, weighted=False, scatter_plot=False,
                  outdir=None, preamble=None):
    # setup output fnames
    table_wse_fname = None
    table_area_fname = None
    table_wse_norm_fname = None
    table_area_norm_fname = None
    table_wse_g_fname = None
    table_area_g_fname = None
    table_wse_g_norm_fname = None
    table_area_g_norm_fname = None
    if outdir is not None:
        # create the outdir if it doesnt exist
        table_wse_fname = os.path.join(outdir,'table_wse.txt')
        table_area_fname = os.path.join(outdir,'table_area.txt')
        table_wse_norm_fname = os.path.join(outdir,'table_wse_norm.txt')
        table_area_norm_fname = os.path.join(outdir,'table_area_norm.txt')
        table_wse_g_fname = os.path.join(outdir,'table_wse_global.txt')
        table_area_g_fname = os.path.join(outdir,'table_area_global.txt')
        table_wse_g_norm_fname = os.path.join(outdir,'table_wse_global_norm.txt')
        table_area_g_norm_fname = os.path.join(outdir,'table_area_global_norm.txt')
    # get pass/fail bounds
    passfail = get_passfail()

    # tile-by-tile metrics
    tile_table = {}
    tile_table_normalized = {}
    for tile_metrics in metrics:
        tile_table = append_tile_table(tile_metrics, tile_table,
                                       inverse_variance_weight=weighted)
        tile_table_normalized = append_tile_table(tile_metrics, tile_table_normalized,
                                                  wse_prefix='wse_e/wse_u_',
                                                  area_prefix='a_%e/a_%u_',
                                                  normalize_by_uncert=True)

    if weighted:
        weight_desc = 'inverse variance weight'
    else:
        weight_desc = 'unweighted'

    ttl = 'Tile metrics (' + weight_desc + ', wse in m):'
    wse_keys = ['wse_e_mean', 'wse_e_std', '|wse_e_68_pct|', 'wse_e_50_pct',
                'total_wse_pix', 'common_wse_pix_%',
                'uncommon_wse_pix_truth_%', 'uncommon_wse_pix_data_%',
                'sim_scene', 'cycle', 'tile_names']
    area_keys = ['a_%e_mean', 'a_%e_std', '|a_%e_68_pct|', 'a_%e_50_pct',
                 'total_area_pix', 'common_area_pix_%',
                 'uncommon_area_pix_truth_%', 'uncommon_area_pix_data_%',
                 'sim_scene', 'cycle', 'tile_names']
    tile_table_wse = {key:tile_table[key] for key in wse_keys}
    tile_table_area = {key:tile_table[key] for key in area_keys}
    sort_table(tile_table_wse, '|wse_e_68_pct|')
    sort_table(tile_table_area, '|a_%e_68_pct|')

    SWOTRiver.analysis.tabley.print_table(tile_table_wse, precision=5,
                                          passfail=passfail, fname=table_wse_fname,
                                          preamble=preamble+'\n'+ttl)
    SWOTRiver.analysis.tabley.print_table(tile_table_area, precision=5,
                                          passfail=passfail, fname=table_area_fname,
                                          preamble=preamble+'\n'+ttl)

    ttl = 'Tile metrics (normalized by uncertainties):'
    wse_keys = ['wse_e/wse_u_mean', 'wse_e/wse_u_std', '|wse_e/wse_u_68_pct|',
                'wse_e/wse_u_50_pct', 'total_wse_pix', 'common_wse_pix_%',
                'uncommon_wse_pix_truth_%', 'uncommon_wse_pix_data_%',
                'sim_scene', 'cycle', 'tile_names']
    area_keys = ['a_%e/a_%u_mean', 'a_%e/a_%u_std', '|a_%e/a_%u_68_pct|',
                 'a_%e/a_%u_50_pct', 'total_area_pix', 'common_area_pix_%',
                 'uncommon_area_pix_truth_%', 'uncommon_area_pix_data_%',
                 'sim_scene', 'cycle', 'tile_names']

    tile_table_wse = {key:tile_table_normalized[key] for key in wse_keys}
    tile_table_area = {key:tile_table_normalized[key] for key in area_keys}
    sort_table(tile_table_wse, '|wse_e/wse_u_68_pct|')
    sort_table(tile_table_area, '|a_%e/a_%u_68_pct|')
    SWOTRiver.analysis.tabley.print_table(tile_table_wse, precision=5,
                                          passfail=passfail, fname=table_wse_norm_fname,
                                          preamble=preamble+'\n'+ttl)
    SWOTRiver.analysis.tabley.print_table(tile_table_area, precision=5,
                                          passfail=passfail, fname=table_area_norm_fname,
                                          preamble=preamble+'\n'+ttl)

    # concatenate tiles for global metrics
    all_dark_frac = np.ma.concatenate(tuple(tile_metrics['dark_frac'] for tile_metrics in metrics))
    all_dark_frac_err = np.ma.concatenate(tuple(tile_metrics['dark_frac_err'] for tile_metrics in metrics))
    all_water_frac = np.ma.concatenate(tuple(tile_metrics['water_frac'] for tile_metrics in metrics))
    all_water_frac_err = np.ma.concatenate(tuple(tile_metrics['water_frac_err'] for tile_metrics in metrics))
    all_wse_err = np.ma.concatenate(tuple(tile_metrics['wse_err'] for tile_metrics in metrics))
    all_wse_uncert = np.ma.concatenate(tuple(tile_metrics['wse_uncert'] for tile_metrics in metrics))
    all_area_perc_err = np.ma.concatenate(tuple(tile_metrics['area_perc_err'] for tile_metrics in metrics))
    all_area_perc_uncert = np.ma.concatenate(tuple(tile_metrics['area_perc_uncert'] for tile_metrics in metrics))
    all_cross_track = np.ma.concatenate(tuple(tile_metrics['cross_track'] for tile_metrics in metrics))
    all_n_wse_pix = np.ma.concatenate(tuple(tile_metrics['n_wse_pix'] for tile_metrics in metrics))
    total_wse_pix = [tile_metrics['total_wse_pix'] for tile_metrics in metrics]
    common_wse_pix = [tile_metrics['common_wse_pix'] for tile_metrics in metrics]
    uncommon_wse_pix_truth = [tile_metrics['uncommon_wse_pix_truth'] for tile_metrics in metrics]
    uncommon_wse_pix_data = [tile_metrics['uncommon_wse_pix_data'] for tile_metrics in metrics]
    total_area_pix = [tile_metrics['total_area_pix'] for tile_metrics in metrics]
    common_area_pix = [tile_metrics['common_area_pix'] for tile_metrics in metrics]
    uncommon_area_pix_truth = [tile_metrics['uncommon_area_pix_truth'] for tile_metrics in metrics]
    uncommon_area_pix_data = [tile_metrics['uncommon_area_pix_data'] for tile_metrics in metrics]
    all_sim_scenes = np.array([tile_metrics['sim_scene'] + '_' + tile_metrics['tile_names']
                           for tile_metrics in metrics
                           for i in range(len(tile_metrics['cross_track']))])
    # global metrics
    total_wse_pix_count = np.sum(total_wse_pix)
    common_wse_pix_pct = np.sum(common_wse_pix)/total_wse_pix_count * 100
    uncommon_wse_pix_truth_pct = np.sum(uncommon_wse_pix_truth)/total_wse_pix_count * 100
    uncommon_wse_pix_data_pct = np.sum(uncommon_wse_pix_data)/total_wse_pix_count * 100
    total_area_pix_count = np.sum(total_area_pix)
    common_area_pix_pct = np.sum(common_area_pix)/total_area_pix_count * 100
    uncommon_area_pix_truth_pct = np.sum(uncommon_area_pix_truth)/total_area_pix_count * 100
    uncommon_area_pix_data_pct = np.sum(uncommon_area_pix_data)/total_area_pix_count * 100

    if weighted:
        global_table = make_global_table(all_wse_err, all_area_perc_err,
                                         wse_weight=1/np.square(all_wse_uncert),
                                         area_weight=1/np.square(all_area_perc_uncert))
    else:
        global_table = make_global_table(all_wse_err, all_area_perc_err)

    global_table['total_wse_pix'] = [total_wse_pix_count]
    global_table['common_wse_pix_%'] = [common_wse_pix_pct]
    global_table['uncommon_wse_pix_truth_%'] = [uncommon_wse_pix_truth_pct]
    global_table['uncommon_wse_pix_data_%'] = [uncommon_wse_pix_data_pct]
    global_table['total_area_pix'] = [total_area_pix_count]
    global_table['common_area_pix_%'] = [common_area_pix_pct]
    global_table['uncommon_area_pix_truth_%'] = [uncommon_area_pix_truth_pct]
    global_table['uncommon_area_pix_data_%'] = [uncommon_area_pix_data_pct]

    ttl = 'Global metrics (' + weight_desc + ', wse in m):'
    wse_keys = ['wse_e_mean', 'wse_e_std', '|wse_e_68_pct|', 'wse_e_50_pct',
                'total_wse_pix', 'common_wse_pix_%',
                'uncommon_wse_pix_truth_%', 'uncommon_wse_pix_data_%']
    area_keys = ['a_%e_mean', 'a_%e_std', '|a_%e_68_pct|', 'a_%e_50_pct',
                 'total_area_pix', 'common_area_pix_%',
                 'uncommon_area_pix_truth_%', 'uncommon_area_pix_data_%']
    global_table_wse = {key:global_table[key] for key in wse_keys}
    global_table_area = {key:global_table[key] for key in area_keys}
    sort_table(global_table_wse, '|wse_e_68_pct|')
    sort_table(global_table_area, '|a_%e_68_pct|')
    SWOTRiver.analysis.tabley.print_table(global_table_wse, precision=5,
                                          passfail=passfail, fname=table_wse_g_fname,
                                          preamble=preamble+'\n'+ttl)
    SWOTRiver.analysis.tabley.print_table(global_table_area, precision=5,
                                          passfail=passfail, fname=table_area_g_fname,
                                          preamble=preamble+'\n'+ttl)

    global_table_weighted = make_global_table(all_wse_err/all_wse_uncert,
                                              all_area_perc_err/all_area_perc_uncert,
                                              wse_prefix='wse_e/wse_u_',
                                              area_prefix='a_%e/a_%u_')

    global_table_weighted['total_wse_pix'] = [total_wse_pix_count]
    global_table_weighted['common_wse_pix_%'] = [common_wse_pix_pct]
    global_table_weighted['uncommon_wse_pix_truth_%'] = [uncommon_wse_pix_truth_pct]
    global_table_weighted['uncommon_wse_pix_data_%'] = [uncommon_wse_pix_data_pct]
    global_table_weighted['total_area_pix'] = [total_area_pix_count]
    global_table_weighted['common_area_pix_%'] = [common_area_pix_pct]
    global_table_weighted['uncommon_area_pix_truth_%'] = [uncommon_area_pix_truth_pct]
    global_table_weighted['uncommon_area_pix_data_%'] = [uncommon_area_pix_data_pct]

    ttl = 'Global metrics (normalized by uncertainties):'
    wse_keys = ['wse_e/wse_u_mean', 'wse_e/wse_u_std', '|wse_e/wse_u_68_pct|',
                'wse_e/wse_u_50_pct', 'total_wse_pix', 'common_wse_pix_%',
                'uncommon_wse_pix_truth_%', 'uncommon_wse_pix_data_%']
    area_keys = ['a_%e/a_%u_mean', 'a_%e/a_%u_std', '|a_%e/a_%u_68_pct|',
                 'a_%e/a_%u_50_pct', 'total_area_pix', 'common_area_pix_%',
                 'uncommon_area_pix_truth_%', 'uncommon_area_pix_data_%']
    global_table_wse_weighted = {key:global_table_weighted[key] for key in wse_keys}
    global_table_area_weighted = {key:global_table_weighted[key] for key in area_keys}
    sort_table(global_table_wse_weighted, '|wse_e/wse_u_68_pct|')
    sort_table(global_table_area_weighted, '|a_%e/a_%u_68_pct|')
    SWOTRiver.analysis.tabley.print_table(global_table_wse_weighted, precision=5,
                                          passfail=passfail, fname=table_wse_g_norm_fname,
                                          preamble=preamble+'\n'+ttl)
    SWOTRiver.analysis.tabley.print_table(global_table_area_weighted, precision=5,
                                          passfail=passfail, fname=table_area_g_norm_fname,
                                          preamble=preamble+'\n'+ttl)

    metrics_to_plot = {'WSE Error (m)':all_wse_err,
                       'Area Percent Error (%)':all_area_perc_err,
                       'Water Fraction Error (%)':all_water_frac_err*100,
                       'Dark Fraction Error (%)':all_dark_frac_err*100}

    uncert_to_plot = {'WSE Error (m)':all_wse_uncert,
                       'Area Percent Error (%)':all_area_perc_uncert,
                       'Water Fraction Error (%)':None,
                       'Dark Fraction Error (%)':None}

    metrics_to_plot_against = {'Cross Track (m)':all_cross_track,
                               'Num WSE Pixels':all_n_wse_pix,
                               'Dark Fraction (%)':all_dark_frac*100,
                               'Water Fraction (%)':all_water_frac*100}

    plot_metrics(metrics_to_plot, metrics_to_plot_against,
        uncert_to_plot=uncert_to_plot, sources=all_sim_scenes, scatter_plot=scatter_plot,
        outdir=outdir)

def append_tile_table(tile_metrics, tile_table={},
                      wse_prefix='wse_e_', area_prefix='a_%e_',
                      inverse_variance_weight=False, normalize_by_uncert=False):

    if not tile_table:
        tile_table = {wse_prefix + 'mean':[],
                      wse_prefix + 'std':[],
                      '|' + wse_prefix + '68_pct|':[],
                      wse_prefix + '50_pct':[],
                      area_prefix + 'mean':[],
                      area_prefix + 'std':[],
                      '|' + area_prefix + '68_pct|':[],
                      area_prefix + '50_pct':[],
                      'total_wse_pix':[],
                      'common_wse_pix_%':[],
                      'uncommon_wse_pix_truth_%':[],
                      'uncommon_wse_pix_data_%':[],
                      'total_area_pix':[],
                      'common_area_pix_%':[],
                      'uncommon_area_pix_truth_%':[],
                      'uncommon_area_pix_data_%':[],
                      'sim_scene':[],
                      'cycle':[],
                      'tile_names':[],}

    wse_err = tile_metrics['wse_err']
    area_perc_err = tile_metrics['area_perc_err']
    if normalize_by_uncert:
        wse_err = wse_err/tile_metrics['wse_uncert']
        area_perc_err = area_perc_err/tile_metrics['area_perc_uncert']

    wse_weight = None
    area_weight = None
    if inverse_variance_weight:
        wse_weight = 1/np.square(tile_metrics['wse_uncert'])
        area_weight = 1/np.square(tile_metrics['area_perc_uncert'])

    wse_err_metrics = compute_metrics_from_error(
        wse_err, weights=wse_weight)
    area_err_metrics = compute_metrics_from_error(
        area_perc_err, weights=area_weight)

    # add data to table
    tile_table[wse_prefix + 'mean'].append(wse_err_metrics['mean'])
    tile_table[wse_prefix + 'std'].append(wse_err_metrics['std'])
    tile_table['|' + wse_prefix + '68_pct|'].append(wse_err_metrics['|68_pct|'])
    tile_table[wse_prefix + '50_pct'].append(wse_err_metrics['50_pct'])
    tile_table[area_prefix + 'mean'].append(area_err_metrics['mean'])
    tile_table[area_prefix + 'std'].append(area_err_metrics['std'])
    tile_table['|' + area_prefix + '68_pct|'].append(area_err_metrics['|68_pct|'])
    tile_table[area_prefix + '50_pct'].append(area_err_metrics['50_pct'])
    tile_table['total_wse_pix'].append(tile_metrics['total_wse_pix'])
    tile_table['total_area_pix'].append(tile_metrics['total_area_pix'])

    if tile_metrics['total_wse_pix'] > 0:
        tile_table['common_wse_pix_%'].append(
            tile_metrics['common_wse_pix']/tile_metrics['total_wse_pix'] * 100)
        tile_table['uncommon_wse_pix_truth_%'].append(
            tile_metrics['uncommon_wse_pix_truth']/tile_metrics['total_wse_pix'] * 100)
        tile_table['uncommon_wse_pix_data_%'].append(
            tile_metrics['uncommon_wse_pix_data']/tile_metrics['total_wse_pix'] * 100)
    else:
        tile_table['common_wse_pix_%'].append(0)
        tile_table['uncommon_wse_pix_truth_%'].append(0)
        tile_table['uncommon_wse_pix_data_%'].append(0)

    if tile_metrics['total_area_pix'] > 0:
        tile_table['common_area_pix_%'].append(
            tile_metrics['common_area_pix']/tile_metrics['total_area_pix'] * 100)
        tile_table['uncommon_area_pix_truth_%'].append(
            tile_metrics['uncommon_area_pix_truth']/tile_metrics['total_area_pix'] * 100)
        tile_table['uncommon_area_pix_data_%'].append(
            tile_metrics['uncommon_area_pix_data']/tile_metrics['total_area_pix'] * 100)
    else:
        tile_table['common_area_pix_%'].append(0)
        tile_table['uncommon_area_pix_truth_%'].append(0)
        tile_table['uncommon_area_pix_data_%'].append(0)

    tile_table['sim_scene'].append(tile_metrics['sim_scene'])
    tile_table['cycle'].append(tile_metrics['cycle'])
    tile_table['tile_names'].append(tile_metrics['tile_names'])
    return tile_table

def make_global_table(all_wse_err, all_area_perc_err,
                      wse_weight=None, area_weight=None,
                      wse_prefix='wse_e_', area_prefix='a_%e_', mask=None):
    wse_err_metrics = compute_metrics_from_error(all_wse_err,
                                                    weights=wse_weight,
                                                    mask=mask)
    area_err_metrics = compute_metrics_from_error(all_area_perc_err,
                                                  weights=area_weight,
                                                  mask=mask)

    global_table = {}
    global_table[wse_prefix + 'mean'] = [wse_err_metrics['mean']]
    global_table[wse_prefix + 'std'] = [wse_err_metrics['std']]
    global_table['|' + wse_prefix + '68_pct|'] = [wse_err_metrics['|68_pct|']]
    global_table[wse_prefix + '50_pct'] = [wse_err_metrics['50_pct']]
    global_table[area_prefix + 'mean'] = [area_err_metrics['mean']]
    global_table[area_prefix + 'std'] = [area_err_metrics['std']]
    global_table['|' + area_prefix + '68_pct|'] = [area_err_metrics['|68_pct|']]
    global_table[area_prefix + '50_pct'] = [area_err_metrics['50_pct']]

    return global_table

def plot_metrics(metrics_to_plot, metrics_to_plot_against,
                 uncert_to_plot=None, poly=2, sources=None, scatter_plot=False,
                 outdir=None):
    warnings.simplefilter("ignore")
    for y_key in metrics_to_plot:
        for x_key in metrics_to_plot_against:
            # create output names
            outname = None
            if outdir is not None:
                y_str = y_key.split('(')[0].strip().replace(' ','_')
                x_str = x_key.split('(')[0].strip().replace(' ','_')
                outname = os.path.join(outdir, '{}_vs_{}.png'.format(y_str, x_str))
            if x_key != y_key:
                mask = np.logical_and(~np.isnan(metrics_to_plot[y_key]),
                                      ~np.isnan(metrics_to_plot_against[x_key]))
                this_y_data = metrics_to_plot[y_key][mask]
                this_x_data = metrics_to_plot_against[x_key][mask]
                this_uncert = None
                if uncert_to_plot is not None:
                    if uncert_to_plot[y_key] is not None:
                        this_uncert = uncert_to_plot[y_key][mask]

                if scatter_plot:
                    plt.figure()
                    plt.title('{} vs. {}'.format(y_key, x_key))
                    plt.xlabel(x_key)
                    plt.ylabel(y_key)
                    plt.scatter(this_x_data, this_y_data,
                                marker='o', s=1)
                    try:
                        x_new, y_new = metrics_fit(this_x_data, this_y_data,
                                                   poly=poly, pts=25)
                        plt.plot(x_new, y_new, 'r--')
                        sig_mask = std_mask(this_y_data, 1)
                        x_new, y_new = metrics_fit(this_x_data[sig_mask],
                                                   this_y_data[sig_mask],
                                                   poly=poly, pts=25)
                        plt.plot(x_new, y_new, 'g--')
                        plt.legend(['fit', '68pct fit', 'data'])
                        if outname is not None:
                            plt.savefig(outname, dpi=300)
                    except Exception as E:
                        print('Plotting Exception: {}'.format(E))
                else:
                    # set bin_edges full extent for possibly filtered
                    # water_frac and dark_frac
                    bin_edges = 100
                    if ('Water Fraction (%)' in x_key) or (
                            'Dark Fraction (%)' in x_key):
                        bin_edges = (100, np.linspace(0,100,100))
                    if ('Cross Track (m)' in x_key):
                        # don't do signed cross-track in the plot to
                        # make it easier to read
                        this_x_data = np.abs(this_x_data)
                    try:
                        scatter_density(this_x_data, this_y_data,
                            uncert=this_uncert, source=sources, bin_edges=bin_edges)
                        plt.title('{} vs. {}'.format(y_key, x_key))
                        plt.xlabel(x_key)
                        plt.ylabel(y_key)
                        if outname is not None:
                            plt.savefig(outname, dpi=300)
                    except Exception as E:
                        print('Plotting Exception: {}'.format(E))


    if outdir is None:
        plt.show()
    warnings.resetwarnings()

def sort_table(table, sort_key):
    sort_idx = np.argsort(table[sort_key])
    for key in table:
        table[key] = np.array(table[key])[sort_idx]

if __name__ == "__main__":
    main()
