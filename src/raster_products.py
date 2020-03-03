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
        ['scene_number', {'dtype': 'i2',
                          'docstr': 'Scene number of the product granule.'}],
        ['tile_numbers', {'dtype': 'i2',
                          'docstr': 'Pixelcloud tile numbers used to assemble the product granule.'}],
        ['tile_names', {'dtype': 'str',
                        'docstr': textjoin("""
                        Pixelcloud tile names using format PPP_TTTS, where PPP is a 3 digit
                        pass number with leading zeros, TTT is a 3 digit tile number within the pass,
                        and S is a character 'L' or 'R' for the left and right swath, respectively.""")}],
        ['proj_type', {'dtype': 'str',
                       'docstr': 'Raster projection type: utm or geo.'}],
        ['proj_res', {'dtype': 'f4',
                      'docstr': 'Raster projection resolution.'}],
        ['utm_num', {'dtype': 'i2',
                     'docstr': 'UTM zone number. Valid only if proj_type is utm.'}],
        ['utm_band', {'dtype': 'str',
                     'docstr': textjoin("""
                     UTM latitude band from Military Grid Reference System (MGRS).
                     Valid only if proj_type is utm. """)}],
        ['x_min', {'dtype': 'f4',
                   'docstr': 'Projection minimum x coordinate.'}],
        ['x_max', {'dtype': 'f4',
                   'docstr': 'Projection maximum x coordinate.'}],
        ['y_min', {'dtype': 'f4',
                   'docstr': 'Projection minimum y coordinate.'}],
        ['y_max', {'dtype': 'f4',
                   'docstr': 'Projection maximum y coordinate.'}],
    ])
    VARIABLES = odict([
        ['x',
         odict([['dtype', 'f4']])],
        ['y',
         odict([['dtype', 'f4']])],
        ['num_pixels',
         odict([['dtype', 'i4'],
                ['long_name', 'number_of_pixels'],
                ['units', 'l'],
                ['valid_min', 1],
                ['valid_max', 999999],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Number of contributing pixelcloud pixels""")],
                ])],
        ['inc',
         odict([['dtype', 'f4'],
                ['long_name', 'incidence angle'],
                ['units', 'degrees'],
                ['valid_min', 0],
                ['valid_max', 999999],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Incidence angle.""")],
                ])],
        ['sig0',
         odict([['dtype', 'f4'],
                ['long_name', 'sigma0'],
                ['units', '1'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Normalized radar cross section, or backscatter
                    brightness.""")],
                ])],
        ['sig0_uncert',
         odict([['dtype', 'f4'],
                ['long_name', 'uncertainty in sigma0'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 1000],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Uncertainty of sigma0. The value is provided in linear units.
                    This value is a one-sigma additive (not multiplicative)
                    uncertainty term, which can be added to or subtracted from
                    sigma0.""")],
                ])],
        ['dark_frac',
         odict([['dtype', 'f4'],
                ['long_name', 'fractional area of dark water'],
                ['units', 'l'],
                ['valid_min', 0],
                ['valid_max', 1],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Fraction of pixel water area covered by dark water.""")],
                ])],
        ['height',
         odict([['dtype', 'f4'],
                ['long_name', 'height above reference ellipsoid'],
                ['units', 'm'],
                ['valid_min', -1500],
                ['valid_max', 15000],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    'Height of the pixel above the reference ellipsoid.""")],
                ])],
        ['height_uncert',
         odict([['dtype', 'f4'],
                ['long_name',
                 'total uncertainty in the height above reference ellipsoid'],
                ['units', 'm'],
                ['valid_min', 0],
                ['valid_max', 100],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Total one-sigma uncertainty in the height above reference
                    ellipsoid including uncertainties of corrections.""")],
                ])],
        ['water_area',
         odict([['dtype', 'f4'],
                ['long_name',
                 'Surface area of detected water'],
                ['units', 'm^2'],
                ['valid_min', 0],
                ['valid_max', 2000000000],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Surface area of the detected water pixels.""")],
                ])],
        ['water_area_uncert',
         odict([['dtype', 'f4'],
                ['long_name', textjoin("""
                    Uncertainty estimate of the surface area of
                    detected water""")],
                ['units', 'm^2'],
                ['valid_min', 0],
                ['valid_max', 2000000000],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Total one-sigma uncertainty in the surface area of the
                    detected water pixels.""")],
                ])],
        ['water_frac',
         odict([['dtype', 'f4'],
                ['long_name', 'water fraction'],
                ['units', '1'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Noisy estimate of the fraction of the pixel that is
                    water.""")],
                ])],
        ['water_frac_uncert',
         odict([['dtype', 'f4'],
                ['long_name', 'water fraction uncertainty'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 999999],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Uncertainty estimate of the water fraction estimate
                    (width of noisy water frac estimate distribution).""")],
                ])],
        ['cross_track',
         odict([['dtype', 'f4'],
                ['long_name', 'approximate cross-track location'],
                ['units', 'm'],
                ['valid_min', -75000],
                ['valid_max', 75000],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Approximate cross-track location of the pixel.""")],
                ])],
        ['qual_flag',
         odict([['dtype', 'i1'],
                ['standard_name', 'quality_flag'],
                ['flag_meanings', 'good bad'],
                ['flag_values', np.array([0, 1]).astype('i1')],
                ['valid_min', 0],
                ['valid_max', 1],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Quality flag for raster data.""")],
                ])],
        ['ice_clim_flag',
         odict([['dtype', 'u1'],
                ['long_name', 'climatological ice cover flag'],
                ['source', 'UNC'],
                ['flag_meanings', textjoin("""
                    no_ice_cover partial_ice_cover full_ice_cover
                    not_available""")],
                ['flag_values', np.array([0, 1, 2, 255]).astype('i2')],
                ['valid_min', 0],
                ['valid_max', 255],
                ['comment', textjoin("""
                    Climatological ice cover flag indicating whether the pixel
                    is ice-covered on the day of the observation based on
                    external climatological information (not the SWOT
                    measurement).  Values of 0, 1, and 2 indicate that the
                    pixel is not ice covered, partially ice covered, and fully
                    ice covered, respectively. A value of 255 indicates that
                    this flag is not available.""")],
                ])],
        ['ice_dyn_flag',
         odict([['dtype', 'u1'],
                ['long_name', 'dynamic ice cover flag'],
                ['source', 'UNC'],
                ['flag_meanings', textjoin("""
                    no_ice_cover partial_ice_cover full_ice_cover
                    not_available""")],
                ['flag_values', np.array([0, 1, 2, 255]).astype('u1')],
                ['valid_min', 0],
                ['valid_max', 255],
                ['comment', textjoin("""
                    Dynamic ice cover flag indicating whether the surface is
                    ice-covered on the day of the observation based on
                    analysis of external satellite optical data.  Values of
                    0, 1, and 2 indicate that the pixel is not ice covered,
                    partially ice covered, and fully ice covered, respectively.
                    A value of 255 indicates that this flag is not available.
                    """)],
                ])],
        ['layover_impact',
         odict([['dtype', 'f4'],
                ['long_name', 'layover impact'],
                ['units', 'm'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Estimate of the height error caused by layover.""")],
                ])],
        ['geoid_height',
         odict([['dtype', 'f4'],
                ['long_name', 'geoid height'],
                ['standard_name','geoid_height_above_reference_ellipsoid'],
                ['source', 'EGM2008 (Pavlis et al., 2012)'],
                ['units', 'm'],
                ['valid_min', -150],
                ['valid_max', 150],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Geoid height above the reference ellipsoid with a
                    correction to refer the value to the mean tide system,
                    i.e. includes the permanent tide (zero frequency).""")],
                ])],
        ['geoid_slope',
         odict([['dtype', 'f4'],
                ['long_name', 'geoid slope'],
                ['units', 'm/m'],
                ['valid_min', -150],
                ['valid_max', 150],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Geoid slope calculated based on the
                    EGM2008 (Pavlis et al., 2012) geoid.""")],
                ])],
        ['solid_earth_tide',
         odict([['dtype', 'f4'],
                ['long_name', 'solid Earth tide height'],
                ['source', textjoin("""
                    Cartwright and Taylor (1971) and Cartwright and Edden
                    (1973)""")],
                ['units', 'm'],
                ['valid_min', -1],
                ['valid_max', 1],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Solid-Earth (body) tide height. The zero-frequency
                    permanent tide component is not included.""")],
                ])],
        ['load_tide_sol1',
         odict([['dtype', 'f4'],
                ['long_name', 'geocentric load tide height from model 1'],
                ['source', 'FES2014b (Carrere et al., 2016)'],
                ['institution', 'LEGOS/CNES'],
                ['units', 'm'],
                ['valid_min', -0.2],
                ['valid_max', 0.2],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Geocentric load tide height. The effect of the ocean tide
                    loading of the Earth’s crust. This value is reported for
                    reference but is not applied to the reported height.""")],
                ])],
        ['load_tide_sol2',
         odict([['dtype', 'f4'],
                ['long_name', 'geocentric load tide height from model 2'],
                ['source', 'GOT4.10c (Ray, 2013)'],
                ['institution', 'GSFC'],
                ['units', 'm'],
                ['valid_min', -0.2],
                ['valid_max', 0.2],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Geocentric load tide height. The effect of the ocean tide
                    loading of the Earth’s crust. This value is reported for
                    reference but is not applied to the reported height.""")],
                ])],
        ['pole_tide',
         odict([['dtype', 'f4'],
                ['long_name', 'geocentric pole tide height'],
                ['source', 'Wahr (1985) and Desai et al. (2015)'],
                ['units', 'm'],
                ['valid_min', -0.2],
                ['valid_max', 0.2],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Geocentric pole tide height. The total of the contribution
                    from the solid-Earth (body) pole tide height and the load
                    pole tide height (i.e., the effect of the ocean pole tide
                    loading of the Earth’s crust).""")],
                ])],
        ['iono_cor_gim_ka',
         odict([['dtype', 'f4'],
                ['long_name', 'ionosphere vertical correction'],
                ['source', 'Global Ionosphere Maps'],
                ['institution', 'JPL'],
                ['units', 'm'],
                ['valid_min', -0.5],
                ['valid_max', 0],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Equivalent vertical correction due to ionosphere delay.
                    The reported pixel height, latitude and longitude are
                    computed after adding negative media corrections to
                    uncorrected range along slant-range paths, accounting for
                    the differential delay between the two KaRIn antennas. The
                    equivalent vertical correction is computed by applying
                    obliquity factors to the slant-path correction. Adding the
                    reported correction to the reported pixel height results
                    in the uncorrected pixel height.""")],
                ])],
        ['model_dry_tropo_cor',
         odict([['dtype', 'f4'],
                ['long_name', 'dry troposphere vertical correction'],
                ['source', 'European Centre for Medium-Range Weather Forecasts'],
                ['institution', 'ECMWF'],
                ['units', 'm'],
                ['valid_min', -3],
                ['valid_max', -1.5],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Equivalent vertical correction due to dry troposphere delay.
                    The reported pixel height, latitude and longitude are
                    computed after adding negative media corrections to
                    uncorrected range along slant-range paths, accounting for
                    the differential delay between the two KaRIn antennas. The
                    equivalent vertical correction is computed by applying
                    obliquity factors to the slant-path correction. Adding the
                    reported correction to the reported pixel height results
                    in the uncorrected pixel height.""")],
                ])],
        ['model_wet_tropo_cor',
         odict([['dtype', 'f4'],
                ['long_name', 'wet troposphere vertical correction'],
                ['source', 'European Centre for Medium-Range Weather Forecasts'],
                ['institution', 'ECMWF'],
                ['units', 'm'],
                ['valid_min', -1],
                ['valid_max', 0],
                ['coordinates', 'x y'],
                ['comment', textjoin("""
                    Equivalent vertical correction due to wet troposphere delay.
                    The reported pixel height, latitude and longitude are
                    computed after adding negative media corrections to
                    uncorrected range along slant-range paths, accounting for
                    the differential delay between the two KaRIn antennas. The
                    equivalent vertical correction is computed by applying
                    obliquity factors to the slant-path correction. Adding the
                    reported correction to the reported pixel height results
                    in the uncorrected pixel height.""")],
                ])],
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
