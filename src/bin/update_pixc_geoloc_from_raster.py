#!/usr/bin/env python
'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

Updates pixc lat, lon and height from a raster file
'''

import logging
import argparse

import SWOTRaster.products
from SWOTRaster.geoloc_raster import GeolocRaster
from SWOTRaster.errors import RasterUsageException
from SWOTWater.products.product import MutableProduct

from swot_pixc2raster import load_raster_configs

LOGGER = logging.getLogger('update_pixc_geoloc_from_raster')


def main():
    """ Updates pixc lat, lon and height from a raster file """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pixc_file', type=str,
                        help='input pixc file')
    parser.add_argument('raster_file', type=str,
                        help='input raster file')
    parser.add_argument('alg_config_file', type=str,
                        help='raster algorithmic config')
    parser.add_argument('runtime_config_file', type=str,
                        help='raster runtime config')
    parser.add_argument('-o', '--output_pixc_file', type=str,
                        help='output pixc file', default='pixel_cloud.nc')
    parser.add_argument('-l', '--log-level', type=str,
                        help="logging level, one of: debug info warning error",
                        default="info")
    args = parser.parse_args()

    level = {'debug': logging.DEBUG, 'info': logging.INFO,
             'warning': logging.WARNING,
             'error': logging.ERROR}[args.log_level]
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=format_str)

    alg_cfg, rt_cfg = load_raster_configs(args.alg_config_file,
                                          args.runtime_config_file)

    pixc_tile = MutableProduct.from_ncfile(args.pixc_file)
    pixc_prod = SWOTRaster.products.ScenePixc.from_tile(pixc_tile, None)

    projection_type = rt_cfg['output_sampling_grid_type']
    if projection_type == 'utm':
        if alg_cfg['debug_flag']:
            raster_prod = SWOTRaster.products.RasterUTMDebug.from_ncfile(
                args.raster_file)
        else:
            raster_prod = SWOTRaster.products.RasterUTM.from_ncfile(
                args.raster_file)
    elif projection_type == 'geo':
        if alg_cfg['debug_flag']:
            raster_prod = SWOTRaster.products.RasterGeoDebug.from_ncfile(
                args.raster_file)
        else:
            raster_prod = SWOTRaster.products.RasterGeo.from_ncfile(
                args.raster_file)
    else:
        raise RasterUsageException(
            'Unknown projection type: {}'.format(projection_type))

    geolocator = GeolocRaster(pixc_prod, alg_cfg)
    geolocator.set_new_height_from_raster(raster_prod)
    out_lat, out_lon, out_height = geolocator.process()

    pixc_tile['pixel_cloud']['height'][:] = out_height
    pixc_tile['pixel_cloud']['latitude'][:] = out_lat
    pixc_tile['pixel_cloud']['longitude'][:] = out_lon
    pixc_tile.to_ncfile(args.output_pixc_file)


if __name__ == "__main__":
    main()
