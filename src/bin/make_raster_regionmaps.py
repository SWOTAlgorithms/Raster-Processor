#!/usr/bin/env python
'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''

import os
import glob
import argparse
import numpy as np
import multiprocessing
import SWOTRaster.products
import SWOTWater.aggregate as ag

from netCDF4 import Dataset
from swot_pixc2raster import load_raster_configs
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

METRICS_LAYER_KEYS = ['river_width', 'lake_area']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proc_raster', type=str,
                        help='input processed raster file (or basename)')
    parser.add_argument('truth_raster', type=str,
                        help='input truth raster file (or basename)')
    parser.add_argument('slant_region_map', type=str,
                        help='slant plane region map file (or basename)')
    parser.add_argument('ground_region_map', type=str,
                        help='ground plane region map file (or dir)')
    parser.add_argument('proc_alg_config', type=str,
                        help='processed raster algorithmic config')
    parser.add_argument('truth_alg_config', type=str,
                        help='truth raster algorithmic config')
    parser.add_argument('runtime_config', type=str,
                        help='raster runtime config')
    parser.add_argument('--basedir', type=str, default=None,
                        help='base directory of processing')
    parser.add_argument('-sb', '--slc_basename', type=str, default=None,
                        help='slc directory basename')
    parser.add_argument('-pb', '--pixc_basename', type=str, default=None,
                        help='pixc directory basename')
    parser.add_argument('-eb', '--pixc_errors_basename', type=str, default=None,
                        help='pixc systematic errors basename')
    parser.add_argument('-e', '--exclude_scenes', default=[], nargs='+',
                        help='List of sim scenes to exclude')
    parser.add_argument('-x', '--max_proc', type=int, default=8,
                        help='Max number of processes')
    args = parser.parse_args()

    if args.basedir is not None:
        if args.slc_basename is None or args.pixc_basename is None:
            print('Must specify at least slc_basename and pixc_basename '
                  + 'if running on a directory structure.')
            return

        print('Getting input files...')
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

        slc_base_list = \
            [os.path.join(*Path(truth_raster_base).parts[:-1])
             for truth_raster_base in truth_raster_base_list]

        slant_region_map_base_list = \
            [os.path.join(*Path(truth_raster_base).parts[:-1], args.slant_region_map)
             for truth_raster_base in truth_raster_base_list]

        # If slant_region_map input is the file itself, get the basename
        slant_region_map_base_list = [Path(*Path(slant_region_map_base).parts[:-2])
                                      if os.path.isfile(slant_region_map_base) else slant_region_map_base
                                      for slant_region_map_base in slant_region_map_base_list]

        tup_list = []
        tup_key_list = []
        for proc_raster_base, truth_raster_base, slc_base, slant_region_map_base in \
            zip(proc_raster_base_list, truth_raster_base_list, slc_base_list,
                slant_region_map_base_list):
            proc_raster_path = os.path.join(proc_raster_base, 'raster_data')
            truth_raster_path = os.path.join(truth_raster_base, 'raster_data')
            proc_raster = os.path.join(proc_raster_path, 'raster.nc')
            truth_raster = os.path.join(truth_raster_path, 'raster.nc')
            proc_raster_regionmaps = os.path.join(proc_raster_path,
                                                  'raster_region_maps.nc')
            truth_raster_regionmaps = os.path.join(truth_raster_path,
                                                   'raster_region_maps.nc')
            proc_int_pixc = os.path.join(proc_raster_path, 'internal_scene_pixc.nc')
            truth_int_pixc = os.path.join(truth_raster_path, 'internal_scene_pixc.nc')
            gdem = os.path.join(slc_base, 'slc_data', 'gdem_truth.RightSwath.nc')
            slant_region_map = os.path.join(slant_region_map_base, 'region_map_data',
                                            'region_map.nc')
            if not os.path.isfile(gdem):
                gdem = os.path.join(slc_base, 'slc_data', 'gdem_truth.LeftSwath.nc')

            if args.pixc_errors_basename is not None:
                sim_scene = Path(proc_raster).parts[-8]
            else:
                sim_scene = Path(proc_raster).parts[-7]

            if sim_scene in args.exclude_scenes:
                print('Not analyzing sim scene: {}'.format(sim_scene))
                continue

            ground_region_map = os.path.join(args.ground_region_map,
                                             sim_scene + '_regionmap.nc')

            proc_raster_obj = SWOTRaster.products.RasterUTM.from_ncfile(proc_raster)
            sim_tile = proc_raster_obj.tile_names
            if isinstance(sim_tile, list):
                sim_tile = '_'.join(sim_tile)

            tup_list.append((proc_raster, truth_raster, proc_raster_regionmaps,
                             truth_raster_regionmaps, proc_int_pixc,
                             truth_int_pixc, slant_region_map, ground_region_map,
                             gdem, args.proc_alg_config, args.truth_alg_config,
                             args.runtime_config, sim_scene, sim_tile))
            tup_key_list.append(sim_scene + '/' + sim_tile)

        # Run it
        print('Creating region maps...')
        failed_procs = mp_proc_async_keyed(proc_catcher, tup_list,
                                           tup_key_list,
                                           processes=args.max_proc)
        print([key for key, val in failed_procs])

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
        proc_int_pixc = os.path.join(proc_raster_path, 'internal_scene_pixc.nc')
        truth_int_pixc = os.path.join(truth_raster_path, 'internal_scene_pixc.nc')

        if os.path.isdir(args.slant_region_map):
            slant_region_map = os.path.join(args.slant_region_map,
                                            'region_map_data', 'region_map.nc')
        else:
            slant_region_map = args.slant_region_map

        # Ground range region map must be a file, as we don't know the scene num
        ground_region_map = args.ground_region_map

        if os.path.isdir(args.slc_basename):
            gdem = os.path.join(args.slc_basename, 'slc_data',
                                'gdem_truth.RightSwath.nc')
            if not os.path.isfile(gdem):
                gdem = os.path.join(args.slc_basename, 'slc_data',
                                    'gdem_truth.LeftSwath.nc')
        else:
            # This is a little messy, but we're reusing the slc basename as the gdem
            gdem = args.slc_basename

        single_tile_regionmaps(proc_raster, truth_raster,
                               proc_raster_regionmaps, truth_raster_regionmaps,
                               proc_int_pixc, truth_int_pixc,
                               slant_region_map, ground_region_map,
                               gdem, args.proc_alg_config, args.truth_alg_config,
                               args.runtime_config)


def mp_proc_async_keyed(func, func_args_tuple, proc_keys_tuple, processes=1):
    pool = multiprocessing.Pool(processes=processes)
    res = []
    for tup, key in zip(func_args_tuple, proc_keys_tuple):
        res.append((key, pool.apply_async(func, tup)))
    pool.close()
    failed_list = []
    for r in res:
        try:
            r[1].get(2592000)
        except Exception as e:
            failed_list.append((r[0], e))

    return failed_list


def proc_catcher(proc_raster, truth_raster, proc_raster_regionmaps,
                 truth_raster_regionmaps, proc_int_pixc,
                 truth_int_pixc, slant_region_map, ground_region_map,
                 gdem, proc_alg_config, truth_alg_config,
                 runtime_config, sim_scene, sim_tile):
    try:
        single_tile_regionmaps(proc_raster, truth_raster,
                               proc_raster_regionmaps, truth_raster_regionmaps,
                               proc_int_pixc, truth_int_pixc,
                               slant_region_map, ground_region_map,
                               gdem, proc_alg_config, truth_alg_config,
                               runtime_config, sim_scene=sim_scene,
                               sim_tile=sim_tile)
    except Exception as e:
        print('Unable to create region maps for {} {}'.format(
            sim_scene, sim_tile))
        print(e)
        raise e


def single_tile_regionmaps(proc_raster, truth_raster,
                           proc_raster_regionmaps, truth_raster_regionmaps,
                           proc_int_pixc, truth_int_pixc,
                           slant_region_map, ground_region_map,
                           gdem, proc_alg_config, truth_alg_config,
                           runtime_config, gdem_subsample_factor=2,
                           sim_scene=None, sim_tile=None):
    # Create proc raster regionmaps
    if os.path.isfile(proc_raster_regionmaps):
        print('Proc raster regionmaps file exists for {}/{}, skipping...'.format(
            sim_scene, sim_tile))
    else:
        print('Making proc raster regionmaps for {}/{}'.format(
            sim_scene, sim_tile))
        proc_alg_cfg, rt_cfg = load_raster_configs(proc_alg_config,
                                                   runtime_config)

        with Dataset(slant_region_map, 'r') as fin:
            sr_region_map_river = fin['region_map_river'][:]
            sr_region_map_lake = fin['region_map_lake'][:]

        region_map_river_raster_proc, region_map_lake_raster_proc = \
            rasterize_region_maps(proc_raster, proc_int_pixc,
                                  sr_region_map_river, sr_region_map_lake,
                                  proc_alg_cfg, rt_cfg)

        make_raster_regionmaps_file(proc_raster, ground_region_map,
                                    proc_raster_regionmaps,
                                    region_map_river_raster_proc,
                                    region_map_lake_raster_proc)

    # Create truth raster regionmaps
    if os.path.isfile(truth_raster_regionmaps):
        print('Truth raster regionmaps file exists for {}/{}, skipping...'.format(
            sim_scene, sim_tile))
    else:
        print('Making truth raster regionmaps for {}/{}'.format(
            sim_scene, sim_tile))
        truth_alg_cfg, rt_cfg = load_raster_configs(truth_alg_config,
                                                    runtime_config)

        with Dataset(ground_region_map, 'r') as fin:
            gr_lat = fin['latitude'][:]
            gr_lon = fin['longitude'][:]
            gr_region_map_river = fin['region_map_river'][:]
            gr_region_map_lake = fin['region_map_lake'][:]

        with Dataset(gdem, 'r') as fin:
            gdem_lat = fin['latitude'][:]
            gdem_lon = fin['longitude'][:]

        # Handle possibly latitude-flipped data
        if gr_lat[-1] < gr_lat[0]:
            gr_lat = gr_lat[::-1]
            gr_region_map_river = gr_region_map_river[::-1]
            gr_region_map_lake = gr_region_map_lake[::-1]

        f_river = RegularGridInterpolator((gr_lat, gr_lon), gr_region_map_river,
                                          method='nearest', bounds_error=False)
        f_lake = RegularGridInterpolator((gr_lat, gr_lon), gr_region_map_lake,
                                         method='nearest', bounds_error=False)
        gdem_region_map_river = f_river((gdem_lat, gdem_lon))
        gdem_region_map_lake = f_lake((gdem_lat, gdem_lon))

        # subsample
        gdem_region_map_river = gdem_region_map_river[::gdem_subsample_factor]
        gdem_region_map_lake = gdem_region_map_lake[::gdem_subsample_factor]

        region_map_river_raster_truth, region_map_lake_raster_truth = \
            rasterize_region_maps(truth_raster, truth_int_pixc,
                                  gdem_region_map_river, gdem_region_map_lake,
                                  truth_alg_cfg, rt_cfg)

        make_raster_regionmaps_file(truth_raster, ground_region_map,
                                    truth_raster_regionmaps,
                                    region_map_river_raster_truth,
                                    region_map_lake_raster_truth)


def rasterize_region_maps(raster_filename, int_pixc_filename,
                          region_map_river, region_map_lake,
                          alg_cfg, rt_cfg):
    raster_in = SWOTRaster.products.RasterUTM.from_ncfile(raster_filename)
    pixc = SWOTRaster.products.ScenePixc.from_ncfile(int_pixc_filename)

    # use improved geolocation if specified in config
    use_improved_geoloc=False
    if alg_cfg['height_constrained_geoloc_source'].lower() == 'lowres_raster' or \
       alg_cfg['height_constrained_geoloc_source'].lower() == 'pixcvec':
        use_improved_geoloc=True

    pixc_mask = pixc.get_valid_mask(use_improved_geoloc=use_improved_geoloc)
    pixc_mask = np.logical_and(
        pixc_mask,
        np.isin(pixc['pixel_cloud']['classification'],
                np.concatenate((alg_cfg['interior_water_classes'],
                                alg_cfg['water_edge_classes'],
                                alg_cfg['land_edge_classes'],
                                alg_cfg['dark_water_classes']))))
    raster_mapping = raster_in.get_raster_mapping(
        pixc, pixc_mask, use_improved_geoloc=use_improved_geoloc)
    size_y = raster_in.dimensions['y']
    size_x = raster_in.dimensions['x']

    region_map_river_pixc = region_map_river[pixc['pixel_cloud']['azimuth_index'],
                                             pixc['pixel_cloud']['range_index']]
    region_map_lake_pixc = region_map_lake[pixc['pixel_cloud']['azimuth_index'],
                                           pixc['pixel_cloud']['range_index']]

    pixc_mask_river_regions = np.logical_and(
        pixc_mask, region_map_river_pixc != -1)

    pixc_mask_lake_regions = np.logical_and(
        pixc_mask, region_map_lake_pixc != -1)

    region_map_river_raster = np.ma.masked_all((size_y, size_x), dtype=np.int32)
    region_map_lake_raster = np.ma.masked_all((size_y, size_x), dtype=np.int32)
    for i in range(0, size_y):
        for j in range(0, size_x):
            if len(raster_mapping[i][j]) != 0:
                good_river = pixc_mask_river_regions[raster_mapping[i][j]]
                good_lake = pixc_mask_lake_regions[raster_mapping[i][j]]

                river_tmp = ag.simple(
                    region_map_river_pixc[raster_mapping[i][j]][good_river],
                    metric='mode')
                lake_tmp = ag.simple(
                    region_map_lake_pixc[raster_mapping[i][j]][good_lake],
                    metric='mode')

                if river_tmp.size > 0 and not np.isnan(river_tmp):
                    region_map_river_raster[i][j] = river_tmp

                if lake_tmp.size > 0 and not np.isnan(lake_tmp):
                    region_map_lake_raster[i][j] = lake_tmp

    return region_map_river_raster, region_map_lake_raster


def make_raster_regionmaps_file(raster_filename, ground_region_map_filename,
                                raster_regionmaps_filename,
                                region_map_river, region_map_lake):
    raster_in = SWOTRaster.products.RasterUTM.from_ncfile(raster_filename)

    with Dataset(ground_region_map_filename, 'r') as fin, \
         Dataset(raster_regionmaps_filename, 'w') as fout:
        fout.createDimension('x', raster_in.dimensions['x'])
        fout.createDimension('y', raster_in.dimensions['y'])
        for key in METRICS_LAYER_KEYS:
            for dim in fin[key].dimensions:
                try:
                    fout.createDimension(dim, fin.dimensions[dim].size)
                except RuntimeError:
                    pass

        region_map_river_var = fout.createVariable('region_map_river',
                                                   region_map_river.dtype,
                                                   ('y', 'x'), fill_value=999999)
        region_map_river_var[:] = region_map_river

        region_map_lake_var = fout.createVariable('region_map_lake',
                                                  region_map_lake.dtype,
                                                  ('y', 'x'), fill_value=999999)
        region_map_lake_var[:] = region_map_lake

        for key in METRICS_LAYER_KEYS:
            this_var = fout.createVariable(key, fin[key].dtype, fin[key].dimensions,
                                           fill_value=fin[key][:].fill_value)
            this_var[:] = fin[key][:]


if __name__ == '__main__':
    main()
