#!/usr/bin/env python
import os
import rdf
import raster
import logging
import argparse

from SWOTWater.products.product import MutableProduct

description = """
description:
    pixc_to_raster.py rasterizes a given pixelcloud using configuration 
    paramaters in a given rdf file
"""

def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = description)
    parser.add_argument("pixc_file", type=str)
    parser.add_argument("rdf_file", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()
    
    cfg = rdf.parse(os.path.abspath(args.rdf_file), comment='!')
    pixc_data = product.MutableProduct.from_ncfile(args.pixc_file)
    processor = raster.Worker(cfg, pixc_data)
    raster_data = processor.rasterize()
    raster_data.to_ncfile(args.out_file)

if __name__ == '__main__':
    main()
