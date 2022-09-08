#!/usr/bin/env python
'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

Rasterizes a given pixelcloud using configuration parameters in algorithmic
and runtime config files

example algorithmic config parameters:
    max_cross_track_distance                        (-) = 64e3
    padding                                         (-) = 0
    interior_water_classes                          (-) = [4, 7]
    water_edge_classes                              (-) = [3, 6]
    land_edge_classes                               (-) = [2]
    dark_water_classes                              (-) = [5, 23, 24]
    height_agg_method                               (-) = weight
    area_agg_method                                 (-) = composite
    height_constrained_geoloc_source                (-) = lowres_raster
    lowres_raster_height_constrained_geoloc_method  (-) = taylor
    lowres_raster_scale_factor                      (-) = 0.2
    wse_uncert_qual_thresh                          (-) = 5
    water_frac_uncert_qual_thresh                   (-) = 0.5
    sig0_uncert_qual_thresh                         (-) = 20
    num_pixels_qual_thresh                          (-) = 5
    debug_flag                                      (-) = False
    write_internal_files                            (-) = False

example runtime config parameters:
    raster_resolution           (-) = 100
    output_sampling_grid_type   (-) = utm
    utm_zone_adjust             (-) = 0
    mgrs_band_adjust            (-) = 0
'''

import os
import ast
import RDF
import logging
import argparse
import SWOTRaster.l2pixc_to_raster

from SWOTRaster.products import ScenePixc
from SWOTWater.products.product import MutableProduct
from cnes.common.lib_lake.proc_pixc_vec import PixelCloudVec

LOGGER = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
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
    parser.add_argument("-id", "--internal_files_dir", type=str,
                        help='directory to write out internal files',
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

    pixc_data = ScenePixc.from_tile(pixc_tile, pixcvec_tile)

    proc = SWOTRaster.l2pixc_to_raster.L2PixcToRaster(
        pixc=pixc_data, algorithmic_config=alg_cfg, runtime_config=rt_cfg)

    product = proc.process()

    if args.internal_files_dir is not None:
        proc.pixc.to_ncfile(os.path.join(args.internal_files_dir,
                                         'internal_scene_pixc.nc'))

    product.to_ncfile(args.out_file)

def load_raster_configs(alg_config_file, runtime_config_file):
    alg_cfg = RDF.RDF()
    alg_cfg.rdfParse(alg_config_file)
    alg_cfg = dict(alg_cfg)

    # Typecast most config values with eval (except strings)
    for key in alg_cfg.keys():
        if key in ['height_agg_method', 'area_agg_method',
                   'height_constrained_geoloc_source',
                   'lowres_raster_height_constrained_geoloc_method']:
            continue
        alg_cfg[key] = ast.literal_eval(alg_cfg[key])

    rt_cfg = RDF.RDF()
    rt_cfg.rdfParse(runtime_config_file)
    rt_cfg = dict(rt_cfg)

    # Typecast most config values with eval (except strings)
    for key in rt_cfg.keys():
        if key in ['output_sampling_grid_type']:
            continue
        rt_cfg[key] = ast.literal_eval(rt_cfg[key])

    return alg_cfg, rt_cfg

if __name__ == '__main__':
    main()