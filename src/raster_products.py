'''
Product description for the raster processor

Author (s): Alexander Corben

'''

from collections import OrderedDict as odict
from netCDF4 import Dataset
import numpy as np

from SWOTWater.products.product import Product

class Raster(Product):
    UID = "raster"
    DIMENSIONS = odict([
        ['xdim', 0],
        ['ydim', 0]
    ])
    ATTRIBUTES = odict([
        ['description', {'dtype': 'str',
                         'value': 'Preliminary Raster Product'}],
        ['proj_type', odict()],
        ['proj_res', odict()],
        ['utm_num', odict()],
        ['x_min', odict()],
        ['x_max', odict()],
        ['y_min', odict()],
        ['y_max', odict()],
    ])
    ATTRIBUTES['description']['docstr'] = ATTRIBUTES['description']['value']
    VARIABLES = odict([
        ['x',
         odict([['dtype', 'f4']])],
        ['y',
         odict([['dtype', 'f4']])],
        ['num_pixels',
         odict([['dtype', 'i4']])],
        ['sigma0',
         odict([['dtype', 'f4']])],
        ['sigma0_uncert',
         odict([['dtype', 'f4']])],
        ['dark_frac',
         odict([['dtype', 'f4']])],
        ['height',
         odict([['dtype', 'f4']])],
        ['height_uncert',
         odict([['dtype', 'f4']])],
        ['water_area',
         odict([['dtype', 'f4']])],
        ['water_area_uncert',
         odict([['dtype', 'f4']])],
        ['water_frac',
         odict([['dtype', 'f4']])],
        ['cross_track',
         odict([['dtype', 'f4']])],
        ['quality_flag',
         odict([['dtype', 'i1']])],
        ['surface_type_flag',
         odict([['dtype', 'i1']])],
        ['rain_flag',
         odict([['dtype', 'i1']])],
        ['frozen_flag',
         odict([['dtype', 'i1']])],
        ['layover_flag',
         odict([['dtype', 'i1']])],
        ['geoid_height',
         odict([['dtype', 'f4']])],
        ['geoid_slope',
         odict([['dtype', 'f4']])],
        ['solid_earth_tide',
         odict([['dtype', 'f4']])],
        ['pole_tide',
         odict([['dtype', 'f4']])],
        ['iono_corr',
         odict([['dtype', 'f4']])],
        ['dry_tropo_corr',
         odict([['dtype', 'f4']])],
        ['wet_tropo_corr',
         odict([['dtype', 'f4']])],
    ])
    for name, reference in VARIABLES.items():
        reference['dimensions'] = odict([['ydim', 0], ['xdim', 0]])
    VARIABLES['x']['dimensions'] = odict([['xdim', 0]])
    VARIABLES['y']['dimensions'] = odict([['ydim', 0]])

class RasterDebug(Raster):
    ATTRIBUTES = odict({key:Raster.ATTRIBUTES[key].copy()
                        for key in Raster.ATTRIBUTES})
    DIMENSIONS = odict({key:Raster.DIMENSIONS[key]
                        for key in Raster.DIMENSIONS})
    VARIABLES = odict({key:Raster.VARIABLES[key].copy()
                        for key in Raster.VARIABLES})
    VARIABLES.update(odict([
        ['classification',
         odict([['dtype', 'i1']])],
    ]))    
    for name, reference in VARIABLES.items():
        reference['dimensions'] = odict([['ydim', 0], ['xdim', 0]])
    VARIABLES['x']['dimensions'] = odict([['xdim', 0]])
    VARIABLES['y']['dimensions'] = odict([['ydim', 0]])
    
