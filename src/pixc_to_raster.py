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
from cnes.common.lib_lake.proc_pixc_vec import PixelCloudVec

description = """
description:
    pixc_to_raster.py rasterizes a given pixelcloud using configuration
    parameters in algorithmic and runtime config files

example algorithmic config parameters:
    padding                                         (-) = 0
    interior_water_classes                          (-) = [4, 24]
    water_edge_classes                              (-) = [3, 23]
    land_edge_classes                               (-) = [2, 22]
    dark_water_classes                              (-) = [22, 23, 24]
    height_agg_method                               (-) = weight
    area_agg_method                                 (-) = composite
    height_constrained_geoloc_source                (-) = lowres_raster
    lowres_raster_height_constrained_geoloc_method  (-) = taylor
    lowres_raster_scale_factor                      (-) = 0.2
    debug_flag                                      (-) = False

example runtime config parameters:
    raster_resolution           (-) = 100
    output_sampling_grid_type   (-) = utm
    utm_zone_adjust             (-) = 0
    mgrs_band_adjust            (-) = 0

"""

def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = description)
    parser.add_argument("pixc_file", type=str,
                        help='input pixelcloud file')
    parser.add_argument("alg_config_file", type=str,
                        help='raster algorithmic config file')
    parser.add_argument("runtime_config_file", type=str,
                        help='raster runtime config file')
    parser.add_argument("out_file", type=str,
                        help='output raster file')
    parser.add_argument("-pv", "--pixcvec_file", type=str,
                        help='pixcvec input file', default=None)
    args = parser.parse_args()

    alg_cfg = rdf.parse(os.path.abspath(args.alg_config_file), comment='!')
    rt_cfg = rdf.parse(os.path.abspath(args.runtime_config_file), comment='!')

    pixc_tile = MutableProduct.from_ncfile(args.pixc_file)
    if args.pixcvec_file is not None:
        pixcvec_tile = PixelCloudVec("SP").set_from_pixcvec_file(
            args.pixcvec_file)
    else:
        pixcvec_tile = None

    pixc_data = RasterPixc.from_tile(pixc_tile, pixcvec_tile)

    proc = raster.L2PixcToRaster(pixc_data, alg_cfg, rt_cfg)
    product = proc.process()
    product.to_ncfile(args.out_file)

if __name__ == '__main__':
    main()
