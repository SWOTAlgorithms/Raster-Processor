#!/usr/bin/env python
'''
Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''

import os
import rdf
import raster
import logging
import argparse

from raster_products import RasterPixc
from SWOTWater.products.product import MutableProduct

description = """
description:
    pixc_to_raster.py rasterizes a given pixelcloud using configuration
    parameters in algorithmic and runtime config files

example algorithmic config parameters:
    buffer_size                         (-) = 0
    interior_water_classes              (-) = [4, 24]
    water_edge_classes                  (-) = [3, 23]
    land_edge_classes                   (-) = [2, 22]
    dark_water_classes                  (-) = [22, 23, 24]
    height_agg_method                   (-) = weight
    area_agg_method                     (-) = composite
    do_improved_geolocation             (-) = True
    improved_geolocation_method         (-) = taylor
    improved_geolocation_smooth_factor  (-) = 5
    debug_flag                          (-) = False

example runtime config parameters:
    output_granule_exent_flag   (-) = 0
    raster_resolution           (-) = 100
    output_sampling_grid_type   (-) = utm
    utm_zone_adjust             (-) = 0
    latitude_band_adjust        (-) = 0

"""

def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = description)
    parser.add_argument("pixc_file", type=str,
                        help='input pixelcloud file')
    parser.add_argument("alg_config_file", type=str,
                        help='algorithmic config file')
    parser.add_argument("runtime_config_file", type=str,
                        help='runtime config file')
    parser.add_argument("out_file", type=str,
                        help='output raster file')
    parser.add_argument("-ig", "--improved_geoloc_pixc_file", type=str,
                        help='improved geolocation pixc input file', default=None)
    args = parser.parse_args()

    alg_cfg = rdf.parse(os.path.abspath(args.alg_config_file), comment='!')
    rt_cfg = rdf.parse(os.path.abspath(args.runtime_config_file), comment='!')

    pixc_tile = MutableProduct.from_ncfile(args.pixc_file)
    pixc_data = RasterPixc.from_tile(pixc_tile)
    if args.improved_geoloc_pixc_file is not None:
        improved_geoloc_pixc_data = MutableProduct.from_ncfile(
            args.improved_geoloc_pixc_file)
    else:
        improved_geoloc_pixc_data = None
    proc = raster.L2PixcToRaster(pixc_data, improved_geoloc_pixc_data,
                                 alg_cfg, rt_cfg)
    product = proc.process()
    product.to_ncfile(args.out_file)

if __name__ == '__main__':
    main()
