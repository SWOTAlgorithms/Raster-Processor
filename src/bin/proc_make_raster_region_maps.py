#!/usr/bin/env python
'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

Make raster region maps for bulk directory structure

Assumed bulk directory structure:
sim_scene_base_directory
└── tile_base_directory
    └── slc_base_directory
        ├── pixc_base_directory
        │   ├── pixc_systematic_errors_base_directory
        │   │   └── proc_raster_systematic_errors_base_directory
        │   └── proc_raster_base_directory
        ├── slant_plane_region_map_base_directory
        └── truth_raster_base_directory
'''

import os
import RDF
import glob
import argparse
import numpy as np
import multiprocessing
import SWOTRaster.products

from pathlib import Path
from make_raster_region_maps import (RASTER_FLAVORS, make_truth_region_maps,
                                     make_proc_region_maps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=str,
                        help='base directory of processing')
    parser.add_argument('slc_basename', type=str,
                        help='slc directory basename')
    parser.add_argument('pixc_basename', type=str,
                        help='pixc directory basename')
    parser.add_argument('raster_basename', type=str,
                        help='raster directory basename')
    parser.add_argument('slant_region_map_basename', type=str,
                        help='slant plane region map basename')
    parser.add_argument('ground_region_map_dir', type=str,
                        help='ground plane region map dir')
    parser.add_argument('flavor', type=str.lower,
                        help='flavor in {}'.format(RASTER_FLAVORS))
    parser.add_argument('alg_config', type=str,
                        help='raster algorithmic config')
    parser.add_argument('runtime_config', type=str,
                        help='raster runtime config')
    parser.add_argument('-eb', '--pixc_errors_basename', type=str, default=None,
                        help='pixc systematic errors basename')
    parser.add_argument('-e', '--exclude_scenes', default=[], nargs='+',
                        help='list of sim scenes to exclude')
    parser.add_argument('-x', '--max_proc', type=int, default=8,
                        help='max number of processes')
    args = parser.parse_args()

    flavor_is_truth = args.flavor == 'truth'

    if flavor_is_truth:
        raster_base_list = glob.glob(os.path.join(
            args.basedir, '*', '*', args.slc_basename, args.raster_basename))
    elif args.pixc_errors_basename is None:
        raster_base_list = glob.glob(os.path.join(
            args.basedir, '*', '*', args.slc_basename, args.pixc_basename,
            args.raster_basename))
    else:
        raster_base_list = glob.glob(os.path.join(
            args.basedir, '*', '*', args.slc_basename, args.pixc_basename,
            args.pixc_errors_basename, args.raster_basename))

    raster_base_list = np.unique([os.path.realpath(filename)
                                  for filename in raster_base_list])

    tup_list = []
    tup_key_list = []
    for raster_base in raster_base_list:
        raster_ann_file = os.path.join(raster_base, 'raster_data',
                                       'raster-annotation.rdf')

        if not os.path.isfile(raster_ann_file):
            print('Raster annotation file does not exist: {}'.format(
                raster_ann_file))
            continue

        raster_ann = RDF.RDF()
        raster_ann.rdfParse(raster_ann_file)
        raster_ann = dict(raster_ann)

        raster_file = raster_ann[args.flavor + ' raster file']
        pixc_files = raster_ann[args.flavor + ' pixc file'].split()

        slant_region_map_files = []
        gdem_files = []
        for pixc_file in pixc_files:
            if flavor_is_truth or Path(pixc_file).parts[-2] == 'pixc_data':
                dir_idx = -3
            elif args.pixc_errors_basename is None:
                dir_idx = -4
            else:
                dir_idx = -5

            slant_region_map_file = os.path.join(
                *Path(pixc_file).parts[:dir_idx], args.slant_region_map_basename,
                'region_map_data', 'region_map.nc')
            slc_dir = os.path.join(
                *Path(pixc_file).parts[:dir_idx], args.slc_basename, 'slc_data')

            gdem_file = os.path.join(slc_dir, 'gdem_truth.RightSwath.nc')
            if not os.path.isfile(gdem_file):
                gdem_file = os.path.join(slc_dir, 'gdem_truth.LeftSwath.nc')

            slant_region_map_files.append(slant_region_map_file)
            gdem_files.append(gdem_file)

        if args.pixc_errors_basename is None:
            sim_scene = Path(raster_file).parts[-7]
        else:
            sim_scene = Path(raster_file).parts[-8]

        if sim_scene in args.exclude_scenes:
            print('Not making region maps for sim scene: {}'.format(sim_scene))
            continue

        ground_region_map_file = os.path.join(
            args.ground_region_map_dir, sim_scene + '_regionmap.nc')

        raster_obj = SWOTRaster.products.RasterUTM.from_ncfile(raster_file)
        sim_tile = raster_obj.tile_names
        if isinstance(sim_tile, list):
            sim_tile = ' '.join(sim_tile)

        raster_dir = os.path.dirname(raster_file)
        output_raster_region_maps_file = \
            os.path.join(raster_dir, 'raster_region_maps.nc')

        if flavor_is_truth:
            tup_list.append((raster_file, output_raster_region_maps_file,
                             pixc_files, ground_region_map_file, gdem_files,
                             args.alg_config, args.runtime_config,
                             sim_scene, sim_tile))
        else:
            tup_list.append((raster_file, output_raster_region_maps_file,
                             pixc_files, slant_region_map_files,
                             ground_region_map_file, args.alg_config,
                             args.runtime_config, sim_scene, sim_tile))

        tup_key_list.append(sim_scene + '/' + sim_tile)

    if flavor_is_truth:
        failed_procs = mp_proc_async_keyed(
            proc_catcher_truth, tup_list, tup_key_list, processes=args.max_proc)
    else:
        failed_procs = mp_proc_async_keyed(
            proc_catcher_proc, tup_list, tup_key_list, processes=args.max_proc)
    print([key for key, val in failed_procs])

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

def proc_catcher_proc(raster_file, output_raster_region_maps_file, pixc_files,
                      slant_region_map_files, ground_region_map_file,
                      alg_config, runtime_config, sim_scene, sim_tile):
    try:
        print('Making proc region maps for {} - {}'.format(
            sim_scene, sim_tile))
        print('make_proc_region_maps({},{},{},{},{},{},{})'.format(
            raster_file, output_raster_region_maps_file,
            pixc_files, slant_region_map_files,
            ground_region_map_file, alg_config, runtime_config))
        input()
        #make_proc_region_maps(raster_file, output_raster_region_maps_file,
        #                      pixc_files, slant_region_map_files,
        #                      ground_region_map_file, alg_config, runtime_config)
    except Exception as e:
        print('Unable to make proc region maps for {} - {}'.format(
            sim_scene, sim_tile))
        print(e)
        raise e

def proc_catcher_truth(raster_file, output_raster_region_maps_file, pixc_files,
                       ground_region_map_file, gdem_files, alg_config,
                       runtime_config, gdem_subsample_factor, sim_scene,
                       sim_tile):
    try:
        print('Making truth region maps for {} - {}'.format(
            sim_scene, sim_tile))
        #make_truth_region_maps(raster_file, output_raster_region_maps_file,
        #                       pixc_files, ground_region_map_file,
        #                       gdem_diles, alg_config, runtime_config,
        #                       gdem_subsample_factor)
    except Exception as e:
        print('Unable to make truth region maps for {} - {}'.format(
            sim_scene, sim_tile))
        print(e)
        raise e

if __name__ == '__main__':
    main()
