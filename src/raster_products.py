'''
Product description for the raster processor

Author (s): Alexander Corben

'''
import utm
import logging
import textwrap
import numpy as np

from netCDF4 import Dataset
from collections import OrderedDict as odict
from SWOTWater.products.product import Product

LOGGER = logging.getLogger(__name__)

def textjoin(text):
    """Dedent join and strip text"""
    text = textwrap.dedent(text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

COMMON_ATTRIBUTES = odict([
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
    ['projection', {'dtype': 'str',
                    'docstr': 'Projection type.'}],
])

COMMON_VARIABLES = odict([
        ['wse',
         odict([['dtype', 'f4'],
                ['long_name', 'water surface elevation above geoid'],
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -1500],
                ['valid_max', 15000],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    'Water surface elevation of the pixel above the geoid.""")],
                ])],
        ['wse_uncert',
         odict([['dtype', 'f4'],
                ['long_name',
                 'total uncertainty in the water surface elevation above geoid'],
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', 0],
                ['valid_max', 100],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Total one-sigma uncertainty in the water surface elevation
                    above geoid including uncertainties of corrections.""")],
                ])],
        ['water_area',
         odict([['dtype', 'f4'],
                ['long_name',
                 'Surface area of detected water'],
                ['grid_mapping', 'crs'],
                ['units', 'm^2'],
                ['valid_min', 0],
                ['valid_max', 2000000000],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Surface area of the detected water pixels.""")],
                ])],
        ['water_area_uncert',
         odict([['dtype', 'f4'],
                ['long_name', textjoin("""
                    Uncertainty estimate of the surface area of
                    detected water""")],
                ['grid_mapping', 'crs'],
                ['units', 'm^2'],
                ['valid_min', 0],
                ['valid_max', 2000000000],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Total one-sigma uncertainty in the surface area of the
                    detected water pixels.""")],
                ])],
        ['water_frac',
         odict([['dtype', 'f4'],
                ['long_name', 'water fraction'],
                ['grid_mapping', 'crs'],
                ['units', '1'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Noisy estimate of the fraction of the pixel that is
                    water.""")],
                ])],
        ['water_frac_uncert',
         odict([['dtype', 'f4'],
                ['long_name', 'water fraction uncertainty'],
                ['grid_mapping', 'crs'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 999999],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Uncertainty estimate of the water fraction estimate
                    (width of noisy water frac estimate distribution).""")],
                ])],
        ['sig0',
         odict([['dtype', 'f4'],
                ['long_name', 'sigma0'],
                ['grid_mapping', 'crs'],
                ['units', '1'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Normalized radar cross section, or backscatter
                    brightness.""")],
                ])],
        ['sig0_uncert',
         odict([['dtype', 'f4'],
                ['long_name', 'uncertainty in sigma0'],
                ['grid_mapping', 'crs'],
                ['units', '1'],
                ['valid_min', 0],
                ['valid_max', 1000],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Uncertainty of sigma0. The value is provided in linear units.
                    This value is a one-sigma additive (not multiplicative)
                    uncertainty term, which can be added to or subtracted from
                    sigma0.""")],
                ])],
        ['inc',
         odict([['dtype', 'f4'],
                ['long_name', 'incidence angle'],
                ['grid_mapping', 'crs'],
                ['units', 'degrees'],
                ['valid_min', 0],
                ['valid_max', 999999],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Incidence angle.""")],
                ])],
        ['cross_track',
         odict([['dtype', 'f4'],
                ['long_name', 'approximate cross-track location'],
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -75000],
                ['valid_max', 75000],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Approximate cross-track location of the pixel.""")],
                ])],
        ['num_pixels',
         odict([['dtype', 'i4'],
                ['long_name', 'number_of_pixels'],
                ['grid_mapping', 'crs'],
                ['units', 'l'],
                ['valid_min', 1],
                ['valid_max', 999999],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Number of contributing pixelcloud pixels""")],
                ])],
        ['qual_flag',
         odict([['dtype', 'i1'],
                ['standard_name', 'quality_flag'],
                ['grid_mapping', 'crs'],
                ['flag_meanings', 'good bad'],
                ['flag_values', np.array([0, 1]).astype('i1')],
                ['valid_min', 0],
                ['valid_max', 1],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Quality flag for raster data.""")],
                ])],
        ['dark_frac',
         odict([['dtype', 'f4'],
                ['long_name', 'fractional area of dark water'],
                ['grid_mapping', 'crs'],
                ['units', 'l'],
                ['valid_min', 0],
                ['valid_max', 1],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Fraction of pixel water area covered by dark water.""")],
                ])],
        ['ice_clim_flag',
         odict([['dtype', 'u1'],
                ['long_name', 'climatological ice cover flag'],
                ['source', 'UNC'],
                ['grid_mapping', 'crs'],
                ['flag_meanings', textjoin("""
                    no_ice_cover partial_ice_cover full_ice_cover
                    not_available""")],
                ['flag_values', np.array([0, 1, 2, 255]).astype('i2')],
                ['valid_min', 0],
                ['valid_max', 255],
                ['coordinates', '[Raster coordinates]'],
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
                ['grid_mapping', 'crs'],
                ['flag_meanings', textjoin("""
                    no_ice_cover partial_ice_cover full_ice_cover
                    not_available""")],
                ['flag_values', np.array([0, 1, 2, 255]).astype('u1')],
                ['valid_min', 0],
                ['valid_max', 255],
                ['coordinates', '[Raster coordinates]'],
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
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -999999],
                ['valid_max', 999999],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Estimate of the height error caused by layover.""")],
                ])],
        ['geoid',
         odict([['dtype', 'f4'],
                ['long_name', 'geoid height'],
                ['standard_name','geoid_height_above_reference_ellipsoid'],
                ['source', 'EGM2008 (Pavlis et al., 2012)'],
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -150],
                ['valid_max', 150],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Geoid height above the reference ellipsoid with a
                    correction to refer the value to the mean tide system,
                    i.e. includes the permanent tide (zero frequency).""")],
                ])],
        ['solid_earth_tide',
         odict([['dtype', 'f4'],
                ['long_name', 'solid Earth tide height'],
                ['source', textjoin("""
                    Cartwright and Taylor (1971) and Cartwright and Edden
                    (1973)""")],
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -1],
                ['valid_max', 1],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Solid-Earth (body) tide height. The zero-frequency
                    permanent tide component is not included.""")],
                ])],
        ['load_tide_sol1',
         odict([['dtype', 'f4'],
                ['long_name', 'geocentric load tide height from model 1'],
                ['source', 'FES2014b (Carrere et al., 2016)'],
                ['institution', 'LEGOS/CNES'],
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -0.2],
                ['valid_max', 0.2],
                ['coordinates', '[Raster coordinates]'],
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
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -0.2],
                ['valid_max', 0.2],
                ['coordinates', '[Raster coordinates]'],
                ['comment', textjoin("""
                    Geocentric load tide height. The effect of the ocean tide
                    loading of the Earth’s crust. This value is reported for
                    reference but is not applied to the reported height.""")],
                ])],
        ['pole_tide',
         odict([['dtype', 'f4'],
                ['long_name', 'geocentric pole tide height'],
                ['source', 'Wahr (1985) and Desai et al. (2015)'],
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -0.2],
                ['valid_max', 0.2],
                ['coordinates', '[Raster coordinates]'],
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
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -0.5],
                ['valid_max', 0],
                ['coordinates', '[Raster coordinates]'],
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
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -3],
                ['valid_max', -1.5],
                ['coordinates', '[Raster coordinates]'],
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
                ['grid_mapping', 'crs'],
                ['units', 'm'],
                ['valid_min', -1],
                ['valid_max', 0],
                ['coordinates', '[Raster coordinates]'],
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


class RasterUTM(Product):
    UID = "raster"
    DIMENSIONS = odict([
        ['x', 0],
        ['y', 0]
    ])
    ATTRIBUTES = odict([
        ['Conventions', COMMON_ATTRIBUTES['Conventions']],
        ['title', COMMON_ATTRIBUTES['title']],
        ['institution', COMMON_ATTRIBUTES['institution']],
        ['source', COMMON_ATTRIBUTES['source']],
        ['history', COMMON_ATTRIBUTES['history']],
        ['mission_name', COMMON_ATTRIBUTES['mission_name']],
        ['references', COMMON_ATTRIBUTES['references']],
        ['reference_document', COMMON_ATTRIBUTES['reference_document']],
        ['contact', COMMON_ATTRIBUTES['contact']],
        ['cycle_number', COMMON_ATTRIBUTES['cycle_number']],
        ['pass_number', COMMON_ATTRIBUTES['pass_number']],
        ['scene_number', COMMON_ATTRIBUTES['scene_number']],
        ['tile_numbers', COMMON_ATTRIBUTES['tile_numbers']],
        ['tile_names', COMMON_ATTRIBUTES['tile_names']],
        ['projection', {'dtype': COMMON_ATTRIBUTES['projection']['dtype'],
                        'value':'Universal Transverse Mercator',
                        'docstr': COMMON_ATTRIBUTES['projection']['docstr']}],
        ['utm_zone_num', {'dtype': 'i2',
                          'docstr': 'UTM zone number.'}],
        ['mgrs_latitude_band', {'dtype': 'str',
                                'docstr': 'MGRS latitude band.'}],
        ['resolution', {'dtype': 'f4',
                        'docstr': 'Raster projection resolution.'}],
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
        ['crs',
         odict([['dtype', 'S1'],
                ['long_name', 'CRS Definition'],
                ['grid_mapping_name', 'transverse_mercator'],
                ['projected_crs_name', '[OGS projected crs name]'],
                ['geographic_crs_name', 'WGS 84'],
                ['reference_ellipsoid_name', 'WGS 84'],
                ['horizontal_datum_name', 'WGS_1984'],
                ['prime_meridian_name', 'Greenwich'],
                ['false_easting', 500000.],
                ['false_northing', '[Projection false northing value]'],
                ['longitude_of_central_meridian', '[Projection longitude of central meridian]'],
                ['longitude_of_prime_meridian', 0.],
                ['latitude_of_projection_origin', 0.],
                ['scale_factor_at_central_meridian', 0.9996],
                ['semi_major_axis', 6378137.],
                ['inverse_flattening', 298.257223563],
                ['crs_wkt', '[OGS Well-Known Text string]'],
                ['comment', 'UTM zone coordinate reference system'],
         ])],
        ['x', 
         odict([['dtype', 'f8'],
                ['long_name', 'x coordinate of projection'],
                ['standard_name', 'projection_x_coordinate'],
                ['units', 'm'],
                ['valid_min', -9999999], # TODO: Figure out valid min/max for utm easting
                ['valid_max', 9999999],
                ['comment', textjoin("""
                    UTM easting coordinate of the pixel""")],
         ])],
        ['y',
         odict([['dtype', 'f8'],
                ['long_name', 'y coordinate of projection'],
                ['standard_name', 'projection_y_coordinate'],
                ['units', 'm'],
                ['valid_min', -9999999], # TODO: Figure out valid min/max for utm northing
                ['valid_max', 9999999],
                ['comment', textjoin("""
                    UTM northing coordinate of the pixel""")],
         ])],
        ['wse', COMMON_VARIABLES['wse'].copy()],
        ['wse_uncert', COMMON_VARIABLES['wse_uncert'].copy()],
        ['water_area', COMMON_VARIABLES['water_area'].copy()],
        ['water_area_uncert', COMMON_VARIABLES['water_area_uncert'].copy()],
        ['water_frac', COMMON_VARIABLES['water_frac'].copy()],
        ['water_frac_uncert', COMMON_VARIABLES['water_frac_uncert'].copy()],
        ['dark_frac', COMMON_VARIABLES['dark_frac'].copy()],
        ['sig0', COMMON_VARIABLES['sig0'].copy()],
        ['sig0_uncert', COMMON_VARIABLES['sig0_uncert'].copy()],
        ['inc', COMMON_VARIABLES['inc'].copy()],
        ['cross_track', COMMON_VARIABLES['cross_track'].copy()],
        ['num_pixels', COMMON_VARIABLES['num_pixels'].copy()],
        ['qual_flag', COMMON_VARIABLES['qual_flag'].copy()],
        ['ice_clim_flag', COMMON_VARIABLES['ice_clim_flag'].copy()],
        ['ice_dyn_flag', COMMON_VARIABLES['ice_dyn_flag'].copy()],
        ['layover_impact', COMMON_VARIABLES['layover_impact'].copy()],
        ['geoid', COMMON_VARIABLES['geoid'].copy()],
        ['solid_earth_tide', COMMON_VARIABLES['solid_earth_tide'].copy()],
        ['load_tide_sol1', COMMON_VARIABLES['load_tide_sol1'].copy()],
        ['load_tide_sol2', COMMON_VARIABLES['load_tide_sol2'].copy()],
        ['pole_tide', COMMON_VARIABLES['pole_tide'].copy()],
        ['iono_cor_gim_ka', COMMON_VARIABLES['iono_cor_gim_ka'].copy()],
        ['model_dry_tropo_cor', COMMON_VARIABLES['model_dry_tropo_cor'].copy()],
        ['model_wet_tropo_cor', COMMON_VARIABLES['model_wet_tropo_cor'].copy()],
    ])
    
    for key in COMMON_VARIABLES:
        VARIABLES[key]['coordinates'] = 'x y'
        VARIABLES[key]['dimensions'] = odict([['y', 0], ['x', 0]])
        
    VARIABLES['x']['dimensions'] = odict([['x', 0]])
    VARIABLES['y']['dimensions'] = odict([['y', 0]])
    VARIABLES['crs']['dimensions'] = odict([])
    
    def get_raster_mapping(self, pixc, mask):
        LOGGER.info('Getting raster mapping')
        pixc_lats = pixc['pixel_cloud']['latitude']
        pixc_lons = np.mod(pixc['pixel_cloud']['longitude'] + 180, 360) - 180

        x_tmp=[]
        y_tmp=[]
        for x in range(0,len(pixc_lats)):
            if mask[x]:
                u_x, u_y, u_num, u_zone = utm.from_latlon(
                    pixc_lats[x], pixc_lons[x], force_zone_number=self.utm_zone_num)
                x_tmp.append(u_x)
                y_tmp.append(u_y)
            else:
                x_tmp.append(0)
                y_tmp.append(0)

        mapping_tmp = []
        for i in range(0, self.dimensions['y']):
            mapping_tmp.append([])
            for j in range(0, self.dimensions['x']):
                mapping_tmp[i].append([])

        for x in range(0,len(pixc_lats)):
            if mask[x]:
                i = round((y_tmp[x] - self.y_min) / self.resolution).astype(int)
                j = round((x_tmp[x] - self.x_min) / self.resolution).astype(int)
                # check bounds
                if (i >= 0 and i < self.dimensions['y'] and
                    j >= 0 and j < self.dimensions['x']):
                    mapping_tmp[i][j].append(x)

        return mapping_tmp

    
class RasterGeo(Product):
    UID = "raster"

    DIMENSIONS = odict([
        ['longitude', 0],
        ['latitude', 0]
    ])

    ATTRIBUTES = odict([
        ['Conventions', COMMON_ATTRIBUTES['Conventions']],
        ['title', COMMON_ATTRIBUTES['title']],
        ['institution', COMMON_ATTRIBUTES['institution']],
        ['source', COMMON_ATTRIBUTES['source']],
        ['history', COMMON_ATTRIBUTES['history']],
        ['mission_name', COMMON_ATTRIBUTES['mission_name']],
        ['references', COMMON_ATTRIBUTES['references']],
        ['reference_document', COMMON_ATTRIBUTES['reference_document']],
        ['contact', COMMON_ATTRIBUTES['contact']],
        ['cycle_number', COMMON_ATTRIBUTES['cycle_number']],
        ['pass_number', COMMON_ATTRIBUTES['pass_number']],
        ['scene_number', COMMON_ATTRIBUTES['scene_number']],
        ['tile_numbers', COMMON_ATTRIBUTES['tile_numbers']],
        ['tile_names', COMMON_ATTRIBUTES['tile_names']],
        ['projection', {'dtype': COMMON_ATTRIBUTES['projection']['dtype'],
                        'value':'Geodetic Latitude/Longitude',
                        'docstr': COMMON_ATTRIBUTES['projection']['docstr']}],
        ['resolution', {'dtype': 'f4',
                        'docstr': 'Raster projection resolution.'}],
        ['longitude_min', {'dtype': 'f4',
                           'docstr': 'Minimum longitude coordinate.'}],
        ['longitude_max', {'dtype': 'f4',
                           'docstr': 'Maximum longitude coordinate.'}],
        ['latitude_min', {'dtype': 'f4',
                          'docstr': 'Minimum latitude coordinate.'}],
        ['latitude_max', {'dtype': 'f4',
                          'docstr': 'Maximum latitude coordinate.'}],
    ])    

    VARIABLES = odict([
        ['crs',
         odict([['dtype', 'S1'],
                ['long_name', 'CRS Definition'],
                ['grid_mapping_name', 'latitude_longitude'],
                ['geographic_crs_name', 'WGS 84'],
                ['reference_ellipsoid_name', 'WGS 84'],
                ['horizontal_datum_name', 'WGS_1984'],
                ['prime_meridian_name', 'Greenwich'],
                ['longitude_of_prime_meridian', 0.],
                ['semi_major_axis', 6378137.],
                ['inverse_flattening', 298.257223563],
                ['crs_wkt', '[OGS Well-Known Text string]'],
                ['comment', 'WGS84 geodetic lat/lon coordinate reference system'],
         ])],
        ['latitude',
         odict([['dtype', 'f8'],
                ['long_name', 'latitude (positive N, negative S)'],
                ['standard_name', 'latitude'],
                ['units', 'degrees_north'],
                ['valid_min', -80],
                ['valid_max', 80],
                ['comment', textjoin("""
                    Geodetic latitude [-80,80] (degrees north of equator) of
                    the pixel.""")],
         ])],
        ['longitude',
         odict([['dtype', 'f8'],
                ['long_name', 'longitude (degrees East)'],
                ['standard_name', 'longitude'],
                ['units', 'degrees_east'],
                ['valid_min', -180],
                ['valid_max', 180],
                ['comment', textjoin("""
                    Geodetic longitude [-180,180] (east of the Greenwich meridian)
                    of the pixel.""")],
         ])],
        ['wse', COMMON_VARIABLES['wse'].copy()],
        ['wse_uncert', COMMON_VARIABLES['wse_uncert'].copy()],
        ['water_area', COMMON_VARIABLES['water_area'].copy()],
        ['water_area_uncert', COMMON_VARIABLES['water_area_uncert'].copy()],
        ['water_frac', COMMON_VARIABLES['water_frac'].copy()],
        ['water_frac_uncert', COMMON_VARIABLES['water_frac_uncert'].copy()],
        ['dark_frac', COMMON_VARIABLES['dark_frac'].copy()],
        ['sig0', COMMON_VARIABLES['sig0'].copy()],
        ['sig0_uncert', COMMON_VARIABLES['sig0_uncert'].copy()],
        ['inc', COMMON_VARIABLES['inc'].copy()],
        ['cross_track', COMMON_VARIABLES['cross_track'].copy()],
        ['num_pixels', COMMON_VARIABLES['num_pixels'].copy()],
        ['qual_flag', COMMON_VARIABLES['qual_flag'].copy()],
        ['ice_clim_flag', COMMON_VARIABLES['ice_clim_flag'].copy()],
        ['ice_dyn_flag', COMMON_VARIABLES['ice_dyn_flag'].copy()],
        ['layover_impact', COMMON_VARIABLES['layover_impact'].copy()],
        ['geoid', COMMON_VARIABLES['geoid'].copy()],
        ['solid_earth_tide', COMMON_VARIABLES['solid_earth_tide'].copy()],
        ['load_tide_sol1', COMMON_VARIABLES['load_tide_sol1'].copy()],
        ['load_tide_sol2', COMMON_VARIABLES['load_tide_sol2'].copy()],
        ['pole_tide', COMMON_VARIABLES['pole_tide'].copy()],
        ['iono_cor_gim_ka', COMMON_VARIABLES['iono_cor_gim_ka'].copy()],
        ['model_dry_tropo_cor', COMMON_VARIABLES['model_dry_tropo_cor'].copy()],
        ['model_wet_tropo_cor', COMMON_VARIABLES['model_wet_tropo_cor'].copy()],
    ])
    
    for key in COMMON_VARIABLES:
        VARIABLES[key]['coordinates'] = 'longitude latitude'
        VARIABLES[key]['dimensions'] = odict([['latitude', 0], ['longitude', 0]])
        
    VARIABLES['longitude']['dimensions'] = odict([['longitude', 0]])
    VARIABLES['latitude']['dimensions'] = odict([['latitude', 0]])
    VARIABLES['crs']['dimensions'] = odict([])

    def get_raster_mapping(self, pixc, mask):
        LOGGER.info('Getting raster mapping')
        pixc_lats = pixc['pixel_cloud']['latitude']
        pixc_lons = np.mod(pixc['pixel_cloud']['longitude'] + 180, 360) - 180
        
        mapping_tmp = []
        for i in range(0, self.dimensions['latitude']):
            mapping_tmp.append([])
            for j in range(0, self.dimensions['longitude']):
                mapping_tmp[i].append([])

        for x in range(0,len(pixc_lats)):
            if mask[x]:
                i = round((pixc_lats[x] - self.latitude_min) / self.resolution).astype(int)
                j = round((pixc_lons[x] - self.longitude_min) / self.resolution).astype(int)
                # check bounds
                if (i >= 0 and i < self.dimensions['latitude'] and
                    j >= 0 and j < self.dimensions['longitude']):
                    mapping_tmp[i][j].append(x)

        return mapping_tmp


class RasterUTMDebug(Product):
    ATTRIBUTES = odict({key:RasterUTM.ATTRIBUTES[key].copy()
                        for key in RasterUTM.ATTRIBUTES})
    DIMENSIONS = odict({key:RasterUTM.DIMENSIONS[key]
                        for key in RasterUTM.DIMENSIONS})
    VARIABLES = odict({key:RasterUTM.VARIABLES[key].copy()
                        for key in RasterUTM.VARIABLES})
    VARIABLES.update(odict([
        ['classification',
         odict([['dtype', 'i1']])],
    ]))
    for key in COMMON_VARIABLES:
        VARIABLES[key]['coordinates'] = 'x y'
        VARIABLES[key]['dimensions'] = odict([['y', 0], ['x', 0]])

    VARIABLES['x']['dimensions'] = odict([['x', 0]])
    VARIABLES['y']['dimensions'] = odict([['y', 0]])
    VARIABLES['crs']['dimensions'] = odict([])


class RasterGeoDebug(Product):
    ATTRIBUTES = odict({key:RasterGeo.ATTRIBUTES[key].copy()
                        for key in RasterGeo.ATTRIBUTES})
    DIMENSIONS = odict({key:RasterGeo.DIMENSIONS[key]
                        for key in RasterGeo.DIMENSIONS})
    VARIABLES = odict({key:RasterGeo.VARIABLES[key].copy()
                        for key in RasterGeo.VARIABLES})
    VARIABLES.update(odict([
        ['classification',
         odict([['dtype', 'i1']])],
    ]))
    for key in COMMON_VARIABLES:
        VARIABLES[key]['coordinates'] = 'longitude latitude'
        VARIABLES[key]['dimensions'] = odict([['latitude', 0], ['longitude', 0]])
        
    VARIABLES['longitude']['dimensions'] = odict([['longitude', 0]])
    VARIABLES['latitude']['dimensions'] = odict([['latitude', 0]])
    VARIABLES['crs']['dimensions'] = odict([])
