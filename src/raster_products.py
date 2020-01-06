'''
Product description for the raster processor

Author (s): Alexander Corben

'''
import textwrap
import numpy as np

from netCDF4 import Dataset
from collections import OrderedDict as odict
from SWOTWater.products.product import Product

def textjoin(text):
    """Dedent join and strip text"""
    text = textwrap.dedent(text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

class Raster(Product):
    UID = "raster"
    DIMENSIONS = odict([
        ['xdim', 0],
        ['ydim', 0]
    ])
    ATTRIBUTES = odict([
        ['Conventions', {'dtype': 'str' ,'value': 'CF-1.7',
                         'docstr':textjoin("""
                         NetCDF-4 conventions adopted in this group. This
                         attribute should be set to CF-1.7 to indicate that the group is
                         compliant with the Climate and Forecast NetCDF conventions.""") }],
        ['title', {'dtype': 'str', 'value':'Level 2 KaRIn High Rate Raster Data Product',
                   'docstr': 'Level 2 KaRIn High Rate Raster Data Product'}],
        ['institution', {'dtype': 'str', 'value': 'JPL',
                         'docstr': 'Name of producing agency.'}],
        ['source', {'dtype': 'str', 'value': 'Ka-band radar interferometer',
                    'docstr': textjoin("""
                    The method of production of the original data.
                    If it was model-generated, source should name the model and its
                    version, as specifically as could be useful. If it is observational,
                    source should characterize it (e.g., 'Ka-band radar interferometer').""")}],
        ['history', {'dtype': 'str',
                     'docstr': textjoin("""
                     UTC time when file generated. Format is:
                     'YYYY-MM-DD hh:mm:ss : Creation'""")}],
        ['mission_name', {'dtype': 'str' ,'value':'SWOT','docstr': 'SWOT'}],
        ['references', {'dtype': 'str',
                        'docstr': textjoin("""
                        Published or web-based references that describe
                        the data or methods used to product it. Provides version number of
                        software generating product.""")}],
        ['reference_document', {'dtype': 'str',
                                'docstr': textjoin("""
                                Name and version of Product Description Document
                                to use as reference for product.""")}],
        ['contact', {'dtype': 'str',
                     'docstr': textjoin("""
                     Contact information for producer of product.
                     (e.g., 'ops@jpl.nasa.gov').""")}],
        ['cycle_number', {'dtype': 'i2',
                          'docstr': 'Cycle number of the product granule.'}],
        ['pass_number', {'dtype': 'i2',
                         'docstr': 'Pass number of the product granule.'}],
        ['tile_numbers', {'dtype': 'i2',
                          'docstr': 'Tile numbers in the pass of the product granule.'}],
        ['proj_type', {'dtype': 'str'}],
        ['proj_res', {'dtype': 'f4'}],
        ['utm_num', {'dtype': 'i2'}],
        ['x_min', {'dtype': 'f4'}],
        ['x_max', {'dtype': 'f4'}],
        ['y_min', {'dtype': 'f4'}],
        ['y_max', {'dtype': 'f4'}],
    ])
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
    
