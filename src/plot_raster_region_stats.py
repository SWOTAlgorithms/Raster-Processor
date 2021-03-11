#!/usr/bin/env python
'''
Copyright (c) 2020-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''

import os
import glob
import argparse
import numpy as np
import raster_products
import SWOTRiver.analysis.tabley

from metrics import *
from netCDF4 import Dataset
from pathlib import Path

METRICS_LAYER_KEYS = ['river_width', 'lake_area']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proc_raster', type=str,
                        help='input processed raster file (or basename)')
    parser.add_argument('truth_raster', type=str,
                        help='input truth raster file (or basename)')
    parser.add_argument('--basedir', type=str, default=None,
                        help='base directory of processing')
    parser.add_argument('-sb', '--slc_basename', type=str, default=None,
                        help='slc directory basename')
    parser.add_argument('-pb', '--pixc_basename', type=str, default=None,
                        help='pixc directory basename')
    parser.add_argument('-eb', '--pixc_errors_basename', type=str, default=None,
                        help='pixc systematic errors basename')
    parser.add_argument('-df', '--dark_frac_thresh', type=float, default=None,
                        help='Dark water fraction threshold for extra metric')
    parser.add_argument('-wf', '--water_frac_thresh', type=float, default=None,
                        help='Water fraction threshold to use as valid data')
    parser.add_argument('-hu', '--wse_uncert_thresh', type=float, default=None,
                        help='WSE uncertainty threshold to use as valid data')
    parser.add_argument('-au', '--area_uncert_thresh', type=float, default=None,
                        help='Area uncertainty threshold to use as valid data')
    parser.add_argument('-ct', '--cross_track_bounds', type=float, default=None,
                        help='Crosstrack bounds to use as valid data', nargs=2)
    parser.add_argument('-mwp', '--min_wse_pixels', type=int, default=None,
                        help='Minimum number of wse pixels to use as valid data')
    parser.add_argument('-map', '--min_area_pixels', type=int, default=None,
                        help='Minimum number of area pixels to use as valid data')
    parser.add_argument('-e', '--exclude_scenes', default=[], nargs='+',
                        help='list of sim scenes to exclude')
    parser.add_argument('-o', '--outdir', type=str, default=None,
                        help='output directory for tables and plots')
    args = parser.parse_args()

    river_metrics = []
    lake_metrics = []
    if args.basedir is not None:
        if args.slc_basename is None or args.pixc_basename is None:
            print('Must specify at least slc_basename and pixc_basename '
                  + 'if aggregating stats')
            return

        # TODO: Right now it's hardcoded that the truth data lives under the slc
        # base directory, and the proc data lives under the pixc base directory
        if args.pixc_errors_basename is not None:
            proc_raster_base_list = glob.glob(os.path.join(
                args.basedir, '*', '*', args.slc_basename, args.pixc_basename,
                args.pixc_errors_basename, args.proc_raster))
        else:
            proc_raster_base_list = glob.glob(os.path.join(
                args.basedir, '*', '*', args.slc_basename, args.pixc_basename,
                args.proc_raster))

        # If proc_raster input is the file itself, get the basename
        proc_raster_base_list = [Path(*Path(proc_raster).parts[:-2])
                                 if os.path.isfile(proc_raster) else proc_raster
                                 for proc_raster in proc_raster_base_list]

        if args.pixc_errors_basename is not None:
            truth_raster_base_list = \
                [os.path.join(*Path(proc_raster).parts[:-3], args.truth_raster)
                 for proc_raster in proc_raster_base_list]
        else:
            truth_raster_base_list = \
                [os.path.join(*Path(proc_raster).parts[:-2], args.truth_raster)
                 for proc_raster in proc_raster_base_list]

        # If truth_raster input is the file itself, get the basename
        truth_raster_base_list = [Path(*Path(truth_raster).parts[:-2])
                                  if os.path.isfile(truth_raster) else truth_raster
                                  for truth_raster in truth_raster_base_list]

        for proc_raster_base, truth_raster_base in \
            zip(proc_raster_base_list, truth_raster_base_list):
            proc_raster_path = os.path.join(proc_raster_base, 'raster_data')
            truth_raster_path = os.path.join(truth_raster_base, 'raster_data')
            proc_raster = os.path.join(proc_raster_path, 'raster.nc')
            truth_raster = os.path.join(truth_raster_path, 'raster.nc')
            proc_raster_regionmaps = os.path.join(proc_raster_path,
                                                  'raster_region_maps.nc')
            truth_raster_regionmaps = os.path.join(truth_raster_path,
                                                   'raster_region_maps.nc')

            if args.pixc_errors_basename is not None:
                sim_scene = Path(proc_raster).parts[-8]
            else:
                sim_scene = Path(proc_raster).parts[-7]

            if sim_scene in args.exclude_scenes:
                print('Not analyzing sim scene: {}'.format(sim_scene))
                continue

            proc_raster_obj = raster_products.RasterUTM.from_ncfile(proc_raster)
            sim_tile = proc_raster_obj.tile_names
            if isinstance(sim_tile, list):
                sim_tile = '_'.join(sim_tile)
            print('Aggregating stats for {} {}'.format(sim_scene, sim_tile))

            tile_river_metrics, tile_lake_metrics = single_tile_stats(
                proc_raster, truth_raster,
                proc_raster_regionmaps, truth_raster_regionmaps,
                sim_scene=sim_scene,
                dark_frac_thresh=args.dark_frac_thresh,
                water_frac_thresh=args.water_frac_thresh,
                wse_uncert_thresh=args.wse_uncert_thresh,
                area_uncert_thresh=args.area_uncert_thresh,
                cross_track_bounds=args.cross_track_bounds,
                min_wse_pixels=args.min_wse_pixels,
                min_area_pixels=args.min_area_pixels)
            river_metrics.append(tile_river_metrics)
            lake_metrics.append(tile_lake_metrics)

    else:
        # Inputs can be either raster files, or basenames
        if os.path.isfile(args.proc_raster):
            proc_raster_base = Path(*Path(args.proc_raster).parts[:-2])
        else:
            proc_raster_base = args.proc_raster

        if os.path.isfile(args.truth_raster):
            truth_raster_base = Path(*Path(args.truth_raster).parts[:-2])
        else:
            truth_raster_base = args.truth_raster

        proc_raster_path = os.path.join(proc_raster_base, 'raster_data')
        truth_raster_path = os.path.join(truth_raster_base, 'raster_data')
        proc_raster = os.path.join(proc_raster_path, 'raster.nc')
        truth_raster = os.path.join(truth_raster_path, 'raster.nc')
        proc_raster_regionmaps = os.path.join(proc_raster_path,
                                              'raster_region_maps.nc')
        truth_raster_regionmaps = os.path.join(truth_raster_path,
                                               'raster_region_maps.nc')

        tile_river_metrics, tile_lake_metrics = single_tile_stats(
            proc_raster, truth_raster,
            proc_raster_regionmaps, truth_raster_regionmaps,
            dark_frac_thresh=args.dark_frac_thresh,
            water_frac_thresh=args.water_frac_thresh,
            wse_uncert_thresh=args.wse_uncert_thresh,
            area_uncert_thresh=args.area_uncert_thresh,
            cross_track_bounds=args.cross_track_bounds,
            min_wse_pixels=args.min_wse_pixels,
            min_area_pixels=args.min_area_pixels)
        river_metrics.append(tile_river_metrics)
        lake_metrics.append(tile_lake_metrics)

    preamble = '\033[93m' + 'Accumulating Metrics:' + '\033[00m'
    if args.dark_frac_thresh is not None:
        preamble += '\n  \033[93m' + 'Dark frac <= {}'.format(
            args.dark_frac_thresh) + '\033[00m'
    if args.water_frac_thresh is not None:
        preamble += '\n  \033[93m' + 'Water frac >= {}'.format(
            args.water_frac_thresh) + '\033[00m'
    if args.wse_uncert_thresh is not None:
        preamble += '\n  \033[93m' + 'WSE uncert <= {}'.format(
            args.wse_uncert_thresh) + '\033[00m'
    if args.area_uncert_thresh is not None:
        preamble += '\n  \033[93m' + 'Water area uncert <= {}'.format(
            args.area_uncert_thresh) + '\033[00m'
    if args.cross_track_bounds is not None:
        preamble += '\n  \033[93m' + 'Cross track bounds = {}'.format(
            args.cross_track_bounds) + '\033[00m'
    if args.min_wse_pixels is not None:
        preamble += '\n  \033[93m' + 'Min WSE pixels = {}'.format(
            args.min_wse_pixels) + '\033[00m'
    if args.min_area_pixels is not None:
        preamble += '\n  \033[93m' + 'Min water area pixels = {}'.format(
            args.min_area_pixels) + '\033[00m'

    print(preamble)
    # check if outdir exists, if not create it
    if args.outdir is not None:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    print_metrics(river_metrics, wb_type='river', outdir=args.outdir,
                  preamble=preamble)
    print_metrics(lake_metrics, wb_type='lake', outdir=args.outdir,
                  preamble=preamble)
    print_global_metrics(river_metrics, lake_metrics, outdir=args.outdir,
                         preamble=preamble)

def single_tile_stats(proc_raster, truth_raster,
                      proc_raster_regionmaps, truth_raster_regionmaps,
                      sim_scene=None, dark_frac_thresh=None,
                      water_frac_thresh=None, wse_uncert_thresh=None,
                      area_uncert_thresh=None, cross_track_bounds=None,
                      min_wse_pixels=None, min_area_pixels=None):

    tile_river_metrics = load_data(proc_raster, truth_raster,
                                   proc_raster_regionmaps,
                                   truth_raster_regionmaps,
                                   wb_type='river',
                                   sim_scene=sim_scene,
                                   dark_frac_thresh=dark_frac_thresh,
                                   water_frac_thresh=water_frac_thresh,
                                   wse_uncert_thresh=wse_uncert_thresh,
                                   area_uncert_thresh=area_uncert_thresh,
                                   cross_track_bounds=cross_track_bounds,
                                   min_wse_pixels=min_wse_pixels,
                                   min_area_pixels=min_area_pixels)
    tile_lake_metrics = load_data(proc_raster, truth_raster,
                                  proc_raster_regionmaps,
                                  truth_raster_regionmaps,
                                  wb_type='lake',
                                  sim_scene=sim_scene,
                                  dark_frac_thresh=dark_frac_thresh,
                                  water_frac_thresh=water_frac_thresh,
                                  wse_uncert_thresh=wse_uncert_thresh,
                                  area_uncert_thresh=area_uncert_thresh,
                                  cross_track_bounds=cross_track_bounds,
                                  min_wse_pixels=min_wse_pixels,
                                  min_area_pixels=min_area_pixels)

    return tile_river_metrics, tile_lake_metrics

def load_data(proc_raster_filename, truth_raster_filename,
              proc_raster_regionmaps_filename, truth_raster_regionmaps_filename,
              sim_scene='', wb_type='river', dark_frac_thresh=None,
              water_frac_thresh=None, wse_uncert_thresh=None,
              area_uncert_thresh=None, cross_track_bounds=None,
              min_wse_pixels=None, min_area_pixels=None):
    proc_raster = raster_products.RasterUTM.from_ncfile(proc_raster_filename)
    truth_raster = raster_products.RasterUTM.from_ncfile(truth_raster_filename)
    with Dataset(proc_raster_regionmaps_filename, 'r') as fin_proc, \
         Dataset(truth_raster_regionmaps_filename, 'r') as fin_truth:
        # When loading, set unregioned area to -1 and width/area to nan
        if wb_type=='river':
            region_map_raster_truth = fin_truth['region_map_river'][:].filled(-1)
            region_map_raster_proc = fin_proc['region_map_river'][:].filled(-1)
            river_width = np.ma.append(fin_truth['river_width'][:], np.nan)
        elif wb_type=='lake':
            region_map_raster_truth = fin_truth['region_map_lake'][:].filled(-1)
            region_map_raster_proc = fin_proc['region_map_lake'][:].filled(-1)
            lake_area = np.ma.append(fin_truth['lake_area'][:], np.nan)

    # 1. get list of unique region ids
    region_list = np.unique(region_map_raster_proc)

    tile_metrics = []

    for region in region_list:
        truth_mask = region_map_raster_truth==region
        proc_mask = region_map_raster_proc==region
        wse_truth_mask = np.logical_and(~truth_raster['wse'].mask,
                                        truth_mask)
        wse_proc_mask = np.logical_and(~proc_raster['wse'].mask,
                                        proc_mask)
        area_truth_mask = np.logical_and(~truth_raster['water_area'].mask,
                                         truth_mask)
        area_proc_mask = np.logical_and(~proc_raster['water_area'].mask,
                                        proc_mask)

        wse_total_mask = np.logical_or(wse_truth_mask, wse_proc_mask)
        wse_common_mask = np.logical_and(wse_truth_mask, wse_proc_mask)
        wse_truth_not_in_data_mask = np.logical_and(wse_truth_mask, ~wse_proc_mask)
        wse_data_not_in_truth_mask = np.logical_and(~wse_truth_mask, wse_proc_mask)

        area_total_mask = np.logical_or(area_truth_mask, area_proc_mask)
        area_common_mask = np.logical_and(area_truth_mask, area_proc_mask)
        area_truth_not_in_data_mask = np.logical_and(area_truth_mask, ~area_proc_mask)
        area_data_not_in_truth_mask = np.logical_and(~area_truth_mask, area_proc_mask)

        # Use additional filters to update masks
        tmp_mask = np.ones_like(wse_common_mask)
        if dark_frac_thresh is not None:
            tmp_mask = np.logical_and(
                tmp_mask, truth_raster['dark_frac'] <= dark_frac_thresh)

        if water_frac_thresh is not None:
            tmp_mask = np.logical_and(
                tmp_mask, truth_raster['water_frac'] >= water_frac_thresh)

        if wse_uncert_thresh is not None:
            tmp_mask = np.logical_and(
                tmp_mask, proc_raster['wse_uncert'] <= wse_uncert_thresh)

        if area_uncert_thresh is not None:
            tmp_mask = np.logical_and(
                tmp_mask, proc_raster['area_uncert'] <= area_uncert_thresh)

        if cross_track_bounds is not None:
            tmp_mask = np.logical_and.reduce(
                (tmp_mask,
                 np.abs(truth_raster['cross_track']) >= min(cross_track_bounds),
                 np.abs(truth_raster['cross_track']) <= max(cross_track_bounds)))

        if min_wse_pixels is not None:
            tmp_mask = np.logical_and(
                tmp_mask, proc_raster['n_wse_pix'] >= min_wse_pixels)

        if min_area_pixels is not None:
            tmp_mask = np.logical_and(
                tmp_mask, proc_raster['n_area_pix'] >= min_area_pixels)

        wse_truth_mask = np.logical_and(wse_truth_mask, tmp_mask)
        wse_proc_mask = np.logical_and(wse_proc_mask, tmp_mask)
        area_truth_mask = np.logical_and(area_truth_mask, tmp_mask)
        area_proc_mask = np.logical_and(area_proc_mask, tmp_mask)

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

        region_metrics = {}
        region_metrics['sim_scene'] = str(sim_scene)
        region_metrics['cycle'] = str(proc_raster.cycle_number)
        region_metrics['tile_names'] = proc_raster.tile_names

        if wb_type=='river':
            region_metrics['river_idx'] = region
            region_metrics['river_width'] = river_width[region]
        elif wb_type=='lake':
            region_metrics['lake_idx'] = region
            region_metrics['lake_area'] = lake_area[region]

        region_metrics['true_ct_mean'] = \
            nanmean_masked(truth_raster['cross_track'][area_truth_mask])
        region_metrics['meas_wse_mean'] = \
            nanmean_masked(proc_raster['wse'][wse_proc_mask])
        region_metrics['meas_area_total'] = \
            np.nansum(proc_raster['water_area'][area_proc_mask])
        region_metrics['true_wse_mean'] = \
            nanmean_masked(truth_raster['wse'][wse_truth_mask])
        region_metrics['true_area_total'] = \
            np.nansum(truth_raster['water_area'][area_truth_mask])

        wse_err = proc_raster['wse'][wse_common_mask] \
                  - truth_raster['wse'][wse_common_mask]
        wse_unc = proc_raster['wse_uncert'][wse_common_mask]
        area_err = proc_raster['water_area'][area_common_mask] \
                   - truth_raster['water_area'][area_common_mask]
        area_perc_err = area_err / truth_raster['water_area'][area_common_mask] * 100
        area_perc_unc = proc_raster['water_area_uncert'][area_common_mask] \
                        / truth_raster['water_area'][area_common_mask] * 100

        region_metrics['wse_err'] = wse_err
        region_metrics['wse_uncert'] = wse_unc
        region_metrics['area_perc_err'] = area_perc_err
        region_metrics['area_perc_uncert'] = area_perc_unc
        region_metrics['total_wse_pix'] = \
            np.count_nonzero(wse_total_mask)
        region_metrics['common_wse_pix'] = \
            np.count_nonzero(wse_common_mask)
        region_metrics['uncommon_wse_pix_truth'] = \
            np.count_nonzero(wse_truth_not_in_data_mask)
        region_metrics['uncommon_wse_pix_data'] = \
            np.count_nonzero(wse_data_not_in_truth_mask)
        region_metrics['total_area_pix'] = \
            np.count_nonzero(area_total_mask)
        region_metrics['common_area_pix'] = \
            np.count_nonzero(area_common_mask)
        region_metrics['uncommon_area_pix_truth'] = \
            np.count_nonzero(area_truth_not_in_data_mask)
        region_metrics['uncommon_area_pix_data'] = \
            np.count_nonzero(area_data_not_in_truth_mask)

        tile_metrics.append(region_metrics)

    return tile_metrics

def append_tile_table(tile_metrics, tile_table={}, wb_type='river',
                      wse_prefix='wse_e_', area_prefix='a_%e_',
                      normalize_by_uncert=False):
    if not tile_table:
        tile_table = {'sim_scene':[],
                      'cycle':[],
                      'tile_names':[]}

        if wb_type=='river':
            tile_table.update({'river_idx':[], 'river_width':[]})
        elif wb_type=='lake':
            tile_table.update({'lake_idx':[], 'lake_area':[]})

        # TODO: meas/true wse and area info may already basically be captured in the 50_pct
        # may be able to remove them...
        tile_table.update({'true_ct_mean':[],
                           'meas_wse_mean':[],
                           'true_wse_mean':[],
                           'meas_area_total':[],
                           'true_area_total':[],
                           wse_prefix + 'mean':[],
                           wse_prefix + 'std':[],
                           '|' + wse_prefix + '68_pct|':[],
                           wse_prefix + '50_pct':[],
                           area_prefix + 'mean':[],
                           area_prefix + 'std':[],
                           '|' + area_prefix + '68_pct|':[],
                           area_prefix + '50_pct':[],
                           'total_wse_pix':[],
                           'common_wse_pix':[],
                           'uncommon_wse_pix_truth':[],
                           'uncommon_wse_pix_data':[],
                           'total_area_pix':[],
                           'common_area_pix':[],
                           'uncommon_area_pix_truth':[],
                           'uncommon_area_pix_data':[]})

    if isinstance(tile_metrics, dict):
        tile_metrics = [tile_metrics]

    for region_data in tile_metrics:
        tile_table['sim_scene'].append(region_data['sim_scene'])
        tile_table['cycle'].append(region_data['cycle'])
        tile_table['tile_names'].append(region_data['tile_names'])
        if wb_type=='river':
            tile_table['river_idx'].append(region_data['river_idx'])
            tile_table['river_width'].append(region_data['river_width'])
        elif wb_type=='lake':
            tile_table['lake_idx'].append(region_data['lake_idx'])
            tile_table['lake_area'].append(region_data['lake_area'])
        tile_table['true_ct_mean'].append(region_data['true_ct_mean'])
        tile_table['meas_wse_mean'].append(region_data['meas_wse_mean'])
        tile_table['true_wse_mean'].append(region_data['true_wse_mean'])
        tile_table['meas_area_total'].append(region_data['meas_area_total'])
        tile_table['true_area_total'].append(region_data['true_area_total'])

        wse_err = region_data['wse_err']
        area_perc_err = region_data['area_perc_err']
        if normalize_by_uncert:
            wse_err = wse_err/region_data['wse_uncert']
            area_perc_err = area_perc_err/region_data['area_perc_uncert']

        wse_err_metrics = compute_metrics_from_error(wse_err)
        area_err_metrics = compute_metrics_from_error(area_perc_err)

        tile_table[wse_prefix + 'mean'].append(wse_err_metrics['mean'])
        tile_table[wse_prefix + 'std'].append(wse_err_metrics['std'])
        tile_table['|' + wse_prefix + '68_pct|'].append(wse_err_metrics['|68_pct|'])
        tile_table[wse_prefix + '50_pct'].append(wse_err_metrics['50_pct'])
        tile_table[area_prefix + 'mean'].append(area_err_metrics['mean'])
        tile_table[area_prefix + 'std'].append(area_err_metrics['std'])
        tile_table['|' + area_prefix + '68_pct|'].append(area_err_metrics['|68_pct|'])
        tile_table[area_prefix + '50_pct'].append(area_err_metrics['50_pct'])

        tile_table['total_wse_pix'].append(region_data['total_wse_pix'])
        tile_table['common_wse_pix'].append(region_data['common_wse_pix'])
        tile_table['uncommon_wse_pix_truth'].append(region_data['uncommon_wse_pix_truth'])
        tile_table['uncommon_wse_pix_data'].append(region_data['uncommon_wse_pix_data'])

        tile_table['total_area_pix'].append(region_data['total_area_pix'])
        tile_table['common_area_pix'].append(region_data['common_area_pix'])
        tile_table['uncommon_area_pix_truth'].append(region_data['uncommon_area_pix_truth'])
        tile_table['uncommon_area_pix_data'].append(region_data['uncommon_area_pix_data'])

    return tile_table

def print_global_metrics(river_metrics, lake_metrics, outdir=None, preamble=None):
    # setup output fnames
    global_table_river_fname = None
    global_table_lake_fname = None
    if outdir is not None:
        global_table_river_fname = os.path.join(outdir, 'global_table_river.txt')
        global_table_lake_fname = os.path.join(outdir, 'global_table_lake.txt')

    river_width_groups = [1, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, \
                          2000, 3000, 4000, 5000, np.inf]
    lake_area_groups = [0, 0.25**2, 0.5**2, 1**2, 2**2, 5**2, 10**2, 20**2, \
                        50**2, np.inf]

    global_table_keys = ['total_wse_pix', 'common_wse_pix',
                         'uncommon_wse_pix_truth', 'uncommon_wse_pix_data',
                         'total_area_pix', 'common_area_pix',
                         'uncommon_area_pix_truth', 'uncommon_area_pix_data']

    global_table_river = {key:[0 for idx in range(len(river_width_groups)+1)] \
                          for key in ['river_width'] + global_table_keys}
    global_table_river['river_width'] = ['{}'.format(x) for x in river_width_groups[:-1]] + ['none', 'all']

    global_table_lake = {key:[0 for idx in range(len(lake_area_groups)+1)] \
                          for key in ['lake_area'] + global_table_keys}
    global_table_lake['lake_area'] = ['{}'.format(x) for x in lake_area_groups[:-1]] + ['none', 'all']

    for tile_metrics in river_metrics:
        for region_metrics in tile_metrics:
            # Aggregate stats for unregioned area
            if region_metrics['river_idx'] == -1:
                for field in global_table_river:
                    if field != 'river_width':
                        global_table_river[field][-2] += region_metrics[field]
                continue # Skip unregioned area for now, TODO: something special here
            # Aggregate stats per width
            for river_width_idx in range(len(river_width_groups)-1):
                this_width = river_width_groups[river_width_idx]
                next_width = river_width_groups[river_width_idx+1]
                is_valid = region_metrics['river_width'] >= this_width \
                           and region_metrics['river_width'] < next_width
                for field in global_table_river:
                    if field != 'river_width' and is_valid:
                        global_table_river[field][river_width_idx] += region_metrics[field]
            # Aggregate stats for all widths
            for field in global_table_river:
                if field != 'river_width':
                    global_table_river[field][-1] += region_metrics[field]


    for tile_metrics in lake_metrics:
        for region_metrics in tile_metrics:
            # Aggregate stats for unregioned area
            if region_metrics['lake_idx'] == -1:
                for field in global_table_lake:
                    if field != 'lake_area':
                        global_table_lake[field][-2] += region_metrics[field]
                continue # Skip unregioned area for now, TODO: something special here
            # Aggregate stats per area
            for lake_area_idx in range(len(lake_area_groups)-1):
                this_area = lake_area_groups[lake_area_idx] * 1000**2
                next_area = lake_area_groups[lake_area_idx+1] * 1000**2
                is_valid = region_metrics['lake_area'] >= this_area \
                           and region_metrics['lake_area'] < next_area
                for field in global_table_lake:
                    if field != 'lake_area' and is_valid:
                        global_table_lake[field][lake_area_idx] += region_metrics[field]
            # Aggregate stats for all areas
            for field in global_table_lake:
                if field != 'lake_area':
                    global_table_lake[field][-1] += region_metrics[field]

    ttl = 'Global metrics - river (width in meters):'
    SWOTRiver.analysis.tabley.print_table(global_table_river, precision=5,
                                          fname=global_table_river_fname,
                                          preamble=preamble+'\n'+ttl)

    ttl = 'Global metrics - lake (area in km^2):'
    SWOTRiver.analysis.tabley.print_table(global_table_lake, precision=5,
                                          fname=global_table_lake_fname,
                                          preamble=preamble+'\n'+ttl)


def print_metrics(metrics, resolution=100, wb_type='river', outdir=None,
                  preamble=None):
    # setup output fnames
    table_wse_fname = None
    table_area_fname = None
    table_wse_norm_fname = None
    table_area_norm_fname = None
    if outdir is not None:
        # create the outdir if it doesnt exist
        table_wse_fname = os.path.join(outdir, 'table_wse_' + wb_type + '.txt')
        table_area_fname = os.path.join(outdir, 'table_area_' + wb_type + '.txt')
        table_wse_norm_fname = os.path.join(outdir, 'table_wse_norm_' + wb_type + '.txt')
        table_area_norm_fname = os.path.join(outdir, 'table_area_norm_' + wb_type + '.txt')

    passfail = get_passfail()
    data_not_in_truth_fail = (100**2)/resolution # TODO: tweak this
    passfail['uncommon_wse_pix_truth'] = [0, data_not_in_truth_fail]
    passfail['uncommon_area_pix_truth'] = [0, data_not_in_truth_fail]

    region_table = {}
    region_table_normalized = {}
    for tile_metrics in metrics:
        region_table = append_tile_table(tile_metrics, region_table,
                                         wb_type=wb_type)
        region_table_normalized = append_tile_table(tile_metrics,
                                                    region_table_normalized,
                                                    wb_type=wb_type,
                                                    wse_prefix='wse_e/wse_u_',
                                                    area_prefix='a_%e/a_%u_',
                                                    normalize_by_uncert=True)

    if wb_type=='river':
        region_idx_key = 'river_idx'
        region_size_key = 'river_width'
    elif wb_type=='lake':
        region_idx_key = 'lake_idx'
        region_size_key = 'lake_area'

    ttl = 'Tile metrics - ' + wb_type + ' - (wse in m):'
    wse_keys = ['sim_scene', 'cycle', 'tile_names', region_idx_key,
                region_size_key, 'true_ct_mean', 'meas_wse_mean', 'true_wse_mean',
                'wse_e_mean', 'wse_e_std', '|wse_e_68_pct|', 'wse_e_50_pct',
                'total_wse_pix', 'common_wse_pix', 'uncommon_wse_pix_truth',
                'uncommon_wse_pix_data']
    area_keys = ['sim_scene', 'cycle', 'tile_names', region_idx_key,
                 region_size_key, 'true_ct_mean', 'meas_area_total', 'true_area_total',
                 'a_%e_mean', 'a_%e_std', '|a_%e_68_pct|', 'a_%e_50_pct',
                 'total_area_pix', 'common_area_pix', 'uncommon_area_pix_truth',
                 'uncommon_area_pix_data']
    region_table_wse = {key:region_table[key] for key in wse_keys}
    region_table_area = {key:region_table[key] for key in area_keys}

    SWOTRiver.analysis.tabley.print_table(region_table_wse, precision=5,
                                          passfail=passfail, fname=table_wse_fname,
                                          preamble=preamble+'\n'+ttl)
    SWOTRiver.analysis.tabley.print_table(region_table_area, precision=5,
                                          passfail=passfail, fname=table_area_fname,
                                          preamble=preamble+'\n'+ttl)

    ttl = 'Tile metrics - ' + wb_type + ' - (normalized by uncertainties):'
    wse_keys = ['sim_scene', 'cycle', 'tile_names', region_idx_key,
                region_size_key, 'meas_wse_mean', 'true_wse_mean',
                'wse_e/wse_u_mean', 'wse_e/wse_u_std', '|wse_e/wse_u_68_pct|',
                'wse_e/wse_u_50_pct', 'total_wse_pix', 'common_wse_pix',
                'uncommon_wse_pix_truth', 'uncommon_wse_pix_data']
    area_keys = ['sim_scene', 'cycle', 'tile_names', region_idx_key,
                 region_size_key, 'meas_area_total', 'true_area_total',
                 'a_%e/a_%u_mean', 'a_%e/a_%u_std', '|a_%e/a_%u_68_pct|',
                 'a_%e/a_%u_50_pct', 'total_area_pix', 'common_area_pix',
                 'uncommon_area_pix_truth', 'uncommon_area_pix_data']

    region_table_wse = {key:region_table_normalized[key] for key in wse_keys}
    region_table_area = {key:region_table_normalized[key] for key in area_keys}

    SWOTRiver.analysis.tabley.print_table(region_table_wse, precision=5,
                                          passfail=passfail, fname=table_wse_norm_fname,
                                          preamble=preamble+'\n'+ttl)
    SWOTRiver.analysis.tabley.print_table(region_table_area, precision=5,
                                          passfail=passfail, fname=table_area_norm_fname,
                                          preamble=preamble+'\n'+ttl)

if __name__ == '__main__':
    main()
