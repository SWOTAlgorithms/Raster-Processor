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

from SWOTRaster.products import ScenePixc
from swot_pixc2raster import load_raster_configs
from SWOTWater.products.product import MutableProduct

LOGGER = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pixc_file')
    parser.add_argument('raster_file')
    parser.add_argument('alg_config_file')
    parser.add_argument('runtime_config_file')
    parser.add_argument('out_pixc_file')
    args = parser.parse_args()

    alg_cfg, rt_cfg = load_raster_configs(args.alg_config_file,
                                          args.runtime_config_file)

    pixc_tile = MutableProduct.from_ncfile(args.pixc_file)
    pixc_prod = ScenePixc.from_tile(pixc_tile, None)

    if rt_cfg['output_sampling_grid_type'] == 'utm':
        if alg_cfg['debug_flag']:
            raster_prod = \
                SWOTRaster.products.RasterUTMDebug.from_ncfile(args.raster_file)
        else:
            raster_prod = \
                SWOTRaster.products.RasterUTM.from_ncfile(args.raster_file)
    elif rt_cfg['output_sampling_grid_type'] == 'geo':
        if alg_cfg['debug_flag']:
            raster_prod = \
                SWOTRaster.products.RasterGeoDebug.from_ncfile(args.raster_file)
        else:
            raster_prod = \
                SWOTRaster.products.RasterGeo.from_ncfile(args.raster_file)

    geolocator = Geoloc_raster(pixc_prod, raster_prod, alg_cfg)
    out_lat, out_lon, out_height = geolocator.process()

    pixc_prod['pixel_cloud']['height'][:] = out_height
    pixc_prod['pixel_cloud']['latitude'][:] = out_lat
    pixc_prod['pixel_cloud']['longitude'][:] = out_lon
    pixc_prod.to_ncfile(args.out_pixc_file)


if __name__ == "__main__":
    main()
