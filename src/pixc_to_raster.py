#!/usr/bin/env python
'''
Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''

import os
import ast
import RDF
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
    parser.add_argument("-id", "--intermediate_files_dir", type=str,
                        help='directory to write out intermediate files',
                        default=None)
    args = parser.parse_args()

    alg_cfg, rt_cfg = load_raster_configs(args.alg_config_file,
                                          args.runtime_config_file)

    pixc_tile = MutableProduct.from_ncfile(args.pixc_file)
    if args.pixcvec_file is not None:
        pixcvec_tile = PixelCloudVec("SP")
        pixcvec_tile.set_from_pixcvec_file(args.pixcvec_file)
    else:
        pixcvec_tile = None

    pixc_data = RasterPixc.from_tile(pixc_tile, pixcvec_tile)

    proc = raster.L2PixcToRaster(pixc=pixc_data, algorithmic_config=alg_cfg,
                                 runtime_config=rt_cfg)
    product = proc.process()

    if args.intermediate_files_dir is not None:
        proc.pixc.to_ncfile(os.path.join(args.intermediate_files_dir,
                                         'intermediate_raster_pixc.nc'))

    product.to_ncfile(args.out_file)

def load_raster_configs(alg_config_file, runtime_config_file):
    alg_cfg = RDF.RDF()
    alg_cfg.rdfParse(os.path.abspath(alg_config_file))
    alg_cfg = dict(alg_cfg)

    # Typecast most config values with eval (except strings)
    for key in alg_cfg.keys():
        if key in ['height_agg_method', 'area_agg_method',
                   'height_constrained_geoloc_source',
                   'lowres_raster_height_constrained_geoloc_method']:
            continue
        alg_cfg[key] = ast.literal_eval(alg_cfg[key])

    rt_cfg = RDF.RDF()
    rt_cfg.rdfParse(os.path.abspath(runtime_config_file))
    rt_cfg = dict(rt_cfg)

    # Typecast most config values with eval (except strings)
    for key in rt_cfg.keys():
        if key in ['output_sampling_grid_type']:
            continue
        rt_cfg[key] = ast.literal_eval(rt_cfg[key])

    return alg_cfg, rt_cfg

if __name__ == '__main__':
    main()
