#!/usr/bin/env python
'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

Makes a raster region map

Note that slant_region_map_files, pixc_files and gdem_files must have
1-to-1 correspondence
'''

import os
import argparse
import numpy as np
import SWOTRaster.products
import SWOTWater.aggregate as ag

from netCDF4 import Dataset
from swot_pixc2raster import load_raster_configs
from scipy.interpolate import RegularGridInterpolator
from SWOTWater.products.product import MutableProduct

METRICS_LAYER_KEYS = ['river_width', 'lake_area']
RASTER_FLAVORS = ['nominal', 'ideal', 'truth']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raster_file', type=str,
                        help='input raster file')
    parser.add_argument('alg_config', type=str,
                        help='raster algorithmic config')
    parser.add_argument('runtime_config', type=str,
                        help='raster runtime config')
    parser.add_argument('ground_region_map_file', type=str,
                        help='ground plane region map file')
    parser.add_argument('-o', '--output_raster_region_maps_file', type=str,
                        help='output raster region maps file',
                        default='raster_region_maps.nc')
    parser.add_argument('-s', '--slant_region_map_files', default=[], nargs='+',
                        help='slant plane region map files')
    parser.add_argument('-p', '--pixc_files', default=[], nargs='+',
                        help='pixc files')
    parser.add_argument('-g', '--gdem_files', default=[], nargs='+',
                        help='gdem files')
    parser.add_argument('flavor', type=str.lower,
                        help='flavor in {}'.format(RASTER_FLAVORS))
    parser.add_argument('-gs', '--gdem_subsample_factor', type=int, default=2,
                        help='gdem subsample factor')
    args = parser.parse_args()

    if args.flavor == 'truth':
        make_truth_region_maps(args.raster_file, args.output_raster_region_maps_file,
                               args.pixc_files, args.ground_region_map_file,
                               args.gdem_files, args.alg_config, args.runtime_config,
                               gdem_subsample_factor=args.gdem_subsample_factor)
    else:
        make_proc_region_maps(args.raster_file, args.output_raster_region_maps_file,
                              args.pixc_files, args.slant_region_map_files,
                              args.ground_region_map_file, args.alg_config,
                              args.runtime_config)

def make_proc_region_maps(raster_file, output_raster_region_maps_file, pixc_files,
                          slant_region_map_files, ground_region_map_file,
                          alg_config, runtime_config):
    # Create proc raster region_map
    if os.path.isfile(output_raster_region_maps_file):
        print('Output file {} exists, skipping...'.format(
            output_raster_region_maps_file))
    else:
        alg_cfg, rt_cfg = load_raster_configs(alg_config, runtime_config)

        sr_region_maps_river = []
        sr_region_maps_lake = []
        for slant_region_map in slant_region_map_files:
            with Dataset(slant_region_map, 'r') as fin:
                sr_region_maps_river.append(fin['region_map_river'][:])
                sr_region_maps_lake.append(fin['region_map_lake'][:])

        region_map_river_raster, region_map_lake_raster = \
            rasterize_region_maps(raster_file, pixc_files, sr_region_maps_river,
                                  sr_region_maps_lake, alg_cfg)

        make_raster_region_maps_file(
            raster_file, ground_region_map_file, output_raster_region_maps_file,
            region_map_river_raster, region_map_lake_raster)

def make_truth_region_maps(raster_file, output_raster_region_maps_file, pixc_files,
                           ground_region_map_file, gdem_files, alg_config,
                           runtime_config, gdem_subsample_factor=2):
    # Create truth raster region_map
    if os.path.isfile(output_raster_region_maps_file):
        print('Output file {} exists, skipping...'.format(
            output_raster_region_maps_file))
    else:
        alg_cfg, rt_cfg = load_raster_configs(alg_config, runtime_config)

        with Dataset(ground_region_map_file, 'r') as fin:
            gr_lat = fin['latitude'][:]
            gr_lon = fin['longitude'][:]
            gr_region_map_river = fin['region_map_river'][:]
            gr_region_map_lake = fin['region_map_lake'][:]

        for gdem_file in gdem_files:
            with Dataset(gdem_file, 'r') as fin:
                gdem_lat = fin['latitude'][:]
                gdem_lon = fin['longitude'][:]

            # Handle possibly latitude-flipped data
            if gr_lat[-1] < gr_lat[0]:
                gr_lat = gr_lat[::-1]
                gr_region_map_river = gr_region_map_river[::-1]
                gr_region_map_lake = gr_region_map_lake[::-1]

            f_river = RegularGridInterpolator(
                (gr_lat, gr_lon), gr_region_map_river, method='nearest',
                bounds_error=False)
            f_lake = RegularGridInterpolator(
                (gr_lat, gr_lon), gr_region_map_lake, method='nearest',
                bounds_error=False)
            gdem_region_map_river = f_river((gdem_lat, gdem_lon))
            gdem_region_map_lake = f_lake((gdem_lat, gdem_lon))

            # subsample
            gdem_region_maps_river.append(
                gdem_region_map_river[::gdem_subsample_factor])
            gdem_region_maps_lake.append(
                gdem_region_map_lake[::gdem_subsample_factor])

        region_map_river_raster, region_map_lake_raster = \
            rasterize_region_maps(raster_file, pixc_files,
                                  gdem_region_maps_river,
                                  gdem_region_maps_lake, alg_cfg)

        make_raster_region_maps_file(
            raster_file, ground_region_map_file, output_raster_region_maps_file,
            region_map_river_raster, region_map_lake_raster)

def rasterize_region_maps(raster_file, pixc_files, region_maps_river,
                          region_maps_lake, alg_cfg):
    raster_in = SWOTRaster.products.RasterUTM.from_ncfile(raster_file)

    # use improved geolocation if specified in config
    use_improved_geoloc=False
    if alg_cfg['height_constrained_geoloc_source'].lower() == 'lowres_raster' or \
       alg_cfg['height_constrained_geoloc_source'].lower() == 'pixcvec':
        use_improved_geoloc=True

    pixc_classes = np.concatenate((alg_cfg['interior_water_classes'],
                                   alg_cfg['water_edge_classes'],
                                   alg_cfg['land_edge_classes'],
                                   alg_cfg['dark_water_classes']))
    size_y = raster_in.dimensions['y']
    size_x = raster_in.dimensions['x']

    river_raster_mapping = []
    for i in range(0, size_y):
        river_raster_mapping.append([])
        for j in range(0, size_x):
            river_raster_mapping[i].append([])

    lake_raster_mapping = []
    for i in range(0, size_y):
        lake_raster_mapping.append([])
        for j in range(0, size_x):
            lake_raster_mapping[i].append([])

    for pixc_file, region_map_river, region_map_lake \
        in zip(pixc_files, region_maps_river, region_maps_lake):
        pixc = SWOTRaster.products.ScenePixc.from_tile(
            MutableProduct.from_ncfile(pixc_file))
        pixc_mask = pixc.get_mask(
            pixc_classes, use_improved_geoloc=use_improved_geoloc)
        raster_mapping = raster_in.get_raster_mapping(
            pixc, pixc_mask, use_improved_geoloc=use_improved_geoloc)

        region_map_river_pixc = region_map_river[
            pixc['pixel_cloud']['azimuth_index'],
            pixc['pixel_cloud']['range_index']]
        region_map_lake_pixc = region_map_lake[
            pixc['pixel_cloud']['azimuth_index'],
            pixc['pixel_cloud']['range_index']]

        pixc_mask_river_regions = np.logical_and(
            pixc_mask, region_map_river_pixc != -1)
        pixc_mask_lake_regions = np.logical_and(
            pixc_mask, region_map_lake_pixc != -1)

        for i in range(0, size_y):
            for j in range(0, size_x):
                if len(raster_mapping[i][j]) != 0:
                    good_river = pixc_mask_river_regions[raster_mapping[i][j]]
                    good_lake = pixc_mask_lake_regions[raster_mapping[i][j]]

                    river_tmp = region_map_river_pixc[raster_mapping[i][j]][good_river]
                    lake_tmp = region_map_lake_pixc[raster_mapping[i][j]][good_lake]

                    river_tmp = river_tmp[np.logical_not(np.isnan(river_tmp))]
                    lake_tmp = lake_tmp[np.logical_not(np.isnan(lake_tmp))]

                    if river_tmp.size > 0:
                        river_raster_mapping[i][j].extend(river_tmp)

                    if lake_tmp.size > 0:
                        lake_raster_mapping[i][j].extend(lake_tmp)

    region_map_river_raster = np.ma.masked_all((size_y, size_x), dtype=np.int32)
    region_map_lake_raster = np.ma.masked_all((size_y, size_x), dtype=np.int32)
    for i in range(0, size_y):
        for j in range(0, size_x):
            if len(river_raster_mapping[i][j]) != 0:
                region_map_river_raster[i][j] = ag.simple(
                    river_raster_mapping[i][j], metric='mode')
            if len(lake_raster_mapping[i][j]) != 0:
                region_map_lake_raster[i][j] = ag.simple(
                    lake_raster_mapping[i][j], metric='mode')

    return region_map_river_raster, region_map_lake_raster

def make_raster_region_maps_file(raster_file, ground_region_map_file,
                                 output_raster_region_maps_file,
                                 region_map_river, region_map_lake):
    raster_in = SWOTRaster.products.RasterUTM.from_ncfile(raster_file)

    with Dataset(ground_region_map_file, 'r') as fin, \
         Dataset(output_raster_region_maps_file, 'w') as fout:
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
