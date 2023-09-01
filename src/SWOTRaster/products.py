'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''

import logging
import textwrap
import numpy as np
import operator as op
import SWOTRaster.raster_crs as raster_crs

from osgeo import osr
from datetime import datetime
from shapely.prepared import prep
from shapely.geometry import Point, Polygon
from collections import OrderedDict as odict
from SWOTWater.products.product import Product, ProductTesterMixIn

UNIX_EPOCH = datetime(1970, 1, 1)
SWOT_EPOCH = datetime(2000, 1, 1)
DATETIME_FORMAT_STR = '%Y-%m-%dT%H:%M:%S.%fZ'
LEAPSEC_FORMAT_STR = '%Y-%m-%dT%H:%M:%SZ'
EMPTY_DATETIME = "0000-00-00T00:00:00.000000Z"
EMPTY_LEAPSEC = "0000-00-00T00:00:00Z"

# define constants for each summary quality value
QUAL_IND_GOOD = 0
QUAL_IND_SUSPECT = 1
QUAL_IND_DEGRADED = 2
QUAL_IND_BAD = 3

# define constants for each quality bit
QUAL_IND_SIG0_QUAL_SUSPECT = 1                  # bit 0
QUAL_IND_CLASS_QUAL_SUSPECT = 2                 # bit 1
QUAL_IND_GEOLOCATION_QUAL_SUSPECT = 4           # bit 2
QUAL_IND_WATER_FRACTION_SUSPECT = 8             # bit 3
QUAL_IND_LARGE_UNCERT_SUSPECT = 32              # bit 5
QUAL_IND_BRIGHT_LAND = 128                      # bit 7
QUAL_IND_LOW_COHERENCE_WATER_SUSPECT = 256      # bit 8
QUAL_IND_FEW_PIXELS = 4096                      # bit 12
QUAL_IND_FAR_RANGE_SUSPECT = 8192               # bit 13
QUAL_IND_NEAR_RANGE_SUSPECT = 16384             # bit 14
QUAL_IND_SIG0_QUAL_DEGRADED = 131072            # bit 17
QUAL_IND_CLASS_QUAL_DEGRADED = 262144           # bit 18
QUAL_IND_GEOLOCATION_QUAL_DEGRADED = 524288     # bit 19
QUAL_IND_LOW_COHERENCE_WATER_DEGRADED = 2097152 # bit 21
QUAL_IND_VALUE_BAD = 16777216                   # bit 24
QUAL_IND_NO_PIXELS = 268435456                  # bit 28
QUAL_IND_OUTSIDE_SCENE_BOUNDS = 536870912       # bit 29
QUAL_IND_INNER_SWATH = 1073741824               # bit 30
QUAL_IND_MISSING_KARIN_DATA = 2147483648        # bit 31

POLYGON_EXTENT_DIST = 200000

NONOVERLAP_TILES_PER_SIDE = 2
OVERLAP_TILES_PER_SIDE = 1

DEFAULT_MAX_CHUNK_SIZE = 100000

LOGGER = logging.getLogger(__name__)

def textjoin(text):
    """ Dedent join and strip text """
    text = textwrap.dedent(text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

COMMON_ATTRIBUTES = odict([
    ['Conventions',
     {'dtype': 'str' ,'value': 'CF-1.7',
      'docstr':textjoin("""
          NetCDF-4 conventions adopted in this group. This
          attribute should be set to CF-1.7 to indicate that the group is
          compliant with the Climate and Forecast NetCDF conventions.""") }],
    ['title',
     {'dtype': 'str', 'value':'Level 2 KaRIn High Rate Raster Data Product',
      'docstr': 'Level 2 KaRIn High Rate Raster Data Product'}],
    ['institution',
     {'dtype': 'str', 'value': 'JPL',
      'docstr': 'Name of producing agency.'}],
    ['source',
     {'dtype': 'str', 'value': 'Ka-band radar interferometer',
      'docstr': textjoin("""
          The method of production of the original data.
          If it was model-generated, source should name the model and its
          version, as specifically as could be useful. If it is observational,
          source should characterize it (e.g., 'Ka-band radar interferometer').""")}],
    ['history',
     {'dtype': 'str',
      'docstr': textjoin("""
          UTC time when file generated. Format is:
          'YYYY-MM-DDThh:mm:ssZ : Creation'""")}],
    ['platform',
     {'dtype': 'str' ,'value':'SWOT',
      'docstr': 'SWOT'}],
    ['references',
     {'dtype': 'str', 'value': 'V1.1.1',
      'docstr': textjoin("""
          Published or web-based references that describe
          the data or methods used to product it. Provides version number of
          software generating product.""")}],
    ['reference_document',
     {'dtype': 'str', 'value':'JPL D-56416 - Revision B - August 17, 2023',
      'docstr': textjoin("""
          Name and version of Product Description Document
          to use as reference for product.""")}],
    ['contact',
     {'dtype': 'str',
      'docstr': textjoin("""
          Contact information for producer of product.
          (e.g., 'ops@jpl.nasa.gov').""")}],
    ['cycle_number',
     {'dtype': 'i2',
      'docstr': 'Cycle number of the product granule.'}],
    ['pass_number',
     {'dtype': 'i2',
      'docstr': 'Pass number of the product granule.'}],
    ['scene_number',
     {'dtype': 'i2',
      'docstr': 'Scene number of the product granule.'}],
    ['tile_numbers',
     {'dtype': 'i2',
      'docstr': textjoin("""
          List of pixel cloud tile numbers in the product granule.
          The numbers are listed in order of increasing measurement time for the
          left side, followed by the right side.""")}],
    ['tile_names',
     {'dtype': 'str',
      'docstr': textjoin("""
          Pixelcloud tile names in the product granule using format PPP_TTTS,
          where PPP is a 3 digit pass number with leading zeros, TTT is a
          3 digit tile number within the pass, and S is a character 'L' or 'R'
          for the left and right swath, respectively. The tile order matches
          that of the tile_numbers attribute.""")}],
    ['tile_polarizations',
     {'dtype': 'str',
      'docstr': textjoin("""
          List of pixel cloud tile polarization flags, inficating whether the
          tile was observed with a horizontal (H) or vertical (V) radar signal
          polarization. The tile order matches that of the tile_numbers
          attribute.""")}],
    ['coordinate_reference_system',
     {'dtype': 'str',
      'docstr': 'Name of the coordinate reference system.'}],
    ['resolution',
     {'dtype': 'f4',
      'docstr': textjoin("""
          Raster sampling grid resolution. Units depend on the coordinate
          reference system.""")}],
    ['short_name',
     {'dtype': 'str', 'value': 'L2_HR_Raster',
      'docstr': 'L2_HR_Raster'}],
    ['descriptor_string',
     {'dtype': 'str',
      'docstr': '<GridResolution><GridUnits>_<CoordinateSystem>_'
          + '<GranuleOverlapFlag>_x_x_x'}],
    ['crid',
     {'dtype': 'str',
      'docstr': textjoin("""
          Composite release identifier (CRID) of the data system used to
          generate this file""")}],
    ['product_version',
     {'dtype': 'str', 'value': 'V1.1.1',
      'docstr': 'Version identifier of this data file'}],
    ['pge_name',
     {'dtype': 'str',
      'docstr': textjoin("""
          Name of the product generation executable (PGE) that created this
          file""")}],
    ['pge_version',
     {'dtype': 'str',
      'docstr': textjoin("""
          Version identifier of the product generation executable (PGE) that
          created this file""")}],
    ['time_granule_start',
     {'dtype': 'str',
      'docstr': textjoin("""
          Nominal starting UTC time of product granule.
          Format is: YYYY-MM-DDThh:mm:ss.ssssssZ""")}],
    ['time_granule_end',
     {'dtype': 'str',
      'docstr': textjoin("""
          Nominal ending UTC time of product granule.
          Format is: YYYY-MM-DDThh:mm:ss.ssssssZ""")}],
    ['time_coverage_start',
     {'dtype': 'str',
      'docstr': textjoin("""
          UTC time of first measurement.
          Format is: YYYY-MM-DDThh:mm:ss.ssssssZ""")}],
    ['time_coverage_end',
     {'dtype': 'str',
      'docstr': textjoin("""
          UTC time of last measurement.
          Format is: YYYY-MM-DDThh:mm:ss.ssssssZ""")}],
    ['geospatial_lon_min',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Westernmost longitude (deg) of raster sampling grid.""")}],
    ['geospatial_lon_max',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Easternmost longitude (deg) of raster sampling grid.""")}],
    ['geospatial_lat_min',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Southernmost longitude (deg) of raster sampling grid.""")}],
    ['geospatial_lat_max',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Northernmost longitude (deg) of raster sampling grid.""")}],
    ['left_first_longitude',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Nominal swath corner longitude for the first range line and left edge
          of the swath (degrees_east).""")}],
    ['left_first_latitude',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Nominal swath corner longitude for the first range line and left edge
          of the swath (degrees_north).""")}],
    ['left_last_longitude',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Nominal swath corner longitude for the last range line and left edge
          of the swath (degrees_east).""")}],
    ['left_last_latitude',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Nominal swath corner longitude for the last range line and left edge
          of the swath (degrees_north).""")}],
    ['right_first_longitude',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Nominal swath corner longitude for the first range line and right edge
          of the swath (degrees_east).""")}],
    ['right_first_latitude',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Nominal swath corner longitude for the first range line and right edge
          of the swath (degrees_north).""")}],
    ['right_last_longitude',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Nominal swath corner longitude for the last range line and right edge
          of the swath (degrees_east).""")}],
    ['right_last_latitude',
     {'dtype': 'f8',
      'docstr': textjoin("""
          Nominal swath corner longitude for the last range line and right edge
          of the swath (degrees_north).""")}],
    ['xref_l2_hr_pixc_files',
     {'dtype': 'str',
      'docstr': textjoin("""
          Names of input Level 2 high rate water mask pixel cloud files.""")}],
    ['xref_l2_hr_pixcvec_files',
     {'dtype': 'str',
      'docstr': textjoin("""
          Names of input Level 2 high rate pixel cloud vector attribute
          files.""")}],
    ['xref_param_l2_hr_raster_file',
     {'dtype': 'str',
      'docstr': textjoin("""
          Name of input Level 2 high rate raster processor configuration
          parameters file.""")}],
    ['xref_reforbittrack_files',
     {'dtype': 'str',
      'docstr': textjoin("""
          Names of input reference orbit track files.""")}],
])

COMMON_VARIABLES = odict([
    ['wse',
     odict([['dtype', 'f4'],
            ['long_name', 'water surface elevation above geoid'],
            ['grid_mapping', 'crs'],
            ['units', 'm'],
            ['quality_flag', 'wse_qual'],
            ['valid_min', -1500],
            ['valid_max', 15000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Water surface elevation of the pixel above the geoid and after
                using models to subtract the effects of tides
                (solid_earth_tide, load_tide_fes, pole_tide).""")],
        ])],
    ['wse_qual',
     odict([['dtype', 'u1'],
            ['long_name', 'summary quality indicator for the water surface elevation'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', 'good suspect degraded bad'],
            ['flag_values', np.array([0, 1, 2, 3]).astype('u1')],
            ['valid_min', 0],
            ['valid_max', 3],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Summary quality indicator for the water surface elevation quantities.
                A value of 0 indicates a nominal measurement, 1 indicates a
                suspect measurement, 2 indicates a degraded measurement,
                and 3 indicates a bad measurement.""")],
        ])],
    ['wse_qual_bitwise',
     odict([['dtype', 'u4'],
            ['long_name', 'bitwise quality indicator for the water surface elevation'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', textjoin("""
                classification_qual_suspect
                geolocation_qual_suspect
                large_uncert_suspect
                bright_land
                few_pixels
                far_range_suspect
                near_range_suspect
                classification_qual_degraded
                geolocation_qual_degraded
                low_coherence_water_degraded
                value_bad
                no_pixels
                outside_scene_bounds
                inner_swath
                missing_karin_data""")],
            ['flag_masks', np.array([
                QUAL_IND_CLASS_QUAL_SUSPECT,
                QUAL_IND_GEOLOCATION_QUAL_SUSPECT,
                QUAL_IND_LARGE_UNCERT_SUSPECT,
                QUAL_IND_BRIGHT_LAND,
                QUAL_IND_FEW_PIXELS,
                QUAL_IND_FAR_RANGE_SUSPECT,
                QUAL_IND_NEAR_RANGE_SUSPECT,
                QUAL_IND_CLASS_QUAL_DEGRADED,
                QUAL_IND_GEOLOCATION_QUAL_DEGRADED,
                QUAL_IND_LOW_COHERENCE_WATER_DEGRADED,
                QUAL_IND_VALUE_BAD,
                QUAL_IND_NO_PIXELS,
                QUAL_IND_OUTSIDE_SCENE_BOUNDS,
                QUAL_IND_INNER_SWATH,
                QUAL_IND_MISSING_KARIN_DATA
            ]).astype('u4')],
            ['valid_min', 0],
            ['valid_max', 4046221478],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Bitwise quality indicator for the water surface elevation quantities.
                If this word is interpreted as an unsigned integer, a value of 0
                indicates good data, positive values less than 32768 represent
                suspect data, values greater than or equal to 32768 but
                less than 8388608 represent degraded data, and values
                greater than or equal to 8388608 represent bad data.""")],
        ])],
    ['wse_uncert',
     odict([['dtype', 'f4'],
            ['long_name', 'uncertainty in the water surface elevation'],
            ['grid_mapping', 'crs'],
            ['units', 'm'],
            ['valid_min', 0],
            ['valid_max', 999999],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                1-sigma uncertainty in the water surface elevation.""")],
        ])],
    ['water_area',
     odict([['dtype', 'f4'],
            ['long_name', 'water surface area'],
            ['grid_mapping', 'crs'],
            ['units', 'm^2'],
            ['quality_flag', 'water_area_qual'],
            ['valid_min', -2000000],
            ['valid_max', 2000000000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Surface area of the water pixels.""")],
        ])],
    ['water_area_qual',
     odict([['dtype', 'u1'],
            ['long_name', 'summary quality indicator for the water surface area'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', 'good suspect degraded bad'],
            ['flag_values', np.array([0, 1, 2, 3]).astype('u1')],
            ['valid_min', 0],
            ['valid_max', 3],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Summary quality indicator for the water surface area and
                water fraction quantities.
                A value of 0 indicates a nominal measurement, 1 indicates a
                suspect measurement, 2 indicates a degraded measurement,
                and 3 indicates a bad measurement.""")],
        ])],
    ['water_area_qual_bitwise',
     odict([['dtype', 'u4'],
            ['long_name', 'bitwise quality indicator for the water surface area'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', textjoin("""
                classification_qual_suspect
                geolocation_qual_suspect
                water_fraction_suspect
                large_uncert_suspect
                bright_land
                low_coherence_water_suspect
                few_pixels
                far_range_suspect
                near_range_suspect
                classification_qual_degraded
                geolocation_qual_degraded
                value_bad
                no_pixels
                outside_scene_bounds
                inner_swath
                missing_karin_data""")],
            ['flag_masks', np.array([
                QUAL_IND_CLASS_QUAL_SUSPECT,
                QUAL_IND_GEOLOCATION_QUAL_SUSPECT,
                QUAL_IND_WATER_FRACTION_SUSPECT,
                QUAL_IND_LARGE_UNCERT_SUSPECT,
                QUAL_IND_BRIGHT_LAND,
                QUAL_IND_LOW_COHERENCE_WATER_SUSPECT,
                QUAL_IND_FEW_PIXELS,
                QUAL_IND_FAR_RANGE_SUSPECT,
                QUAL_IND_NEAR_RANGE_SUSPECT,
                QUAL_IND_CLASS_QUAL_DEGRADED,
                QUAL_IND_GEOLOCATION_QUAL_DEGRADED,
                QUAL_IND_VALUE_BAD,
                QUAL_IND_NO_PIXELS,
                QUAL_IND_OUTSIDE_SCENE_BOUNDS,
                QUAL_IND_INNER_SWATH,
                QUAL_IND_MISSING_KARIN_DATA
            ]).astype('u4')],
            ['valid_min', 0],
            ['valid_max', 4044124590],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Bitwise quality indicator for the water surface area and water
                fraction quantities.
                If this word is interpreted as an unsigned integer, a value of 0
                indicates good data, positive values less than 32768 represent
                suspect data, values greater than or equal to 32768 but
                less than 8388608 represent degraded data, and values
                greater than or equal to 8388608 represent bad data.""")],
        ])],
    ['water_area_uncert',
     odict([['dtype', 'f4'],
            ['long_name', 'uncertainty in the water surface area'],
            ['grid_mapping', 'crs'],
            ['units', 'm^2'],
            ['valid_min', 0],
            ['valid_max', 2000000000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                1-sigma uncertainty in the water surface area.""")],
        ])],
    ['water_frac',
     odict([['dtype', 'f4'],
            ['long_name', 'water fraction'],
            ['grid_mapping', 'crs'],
            ['units', '1'],
            ['quality_flag', 'water_area_qual'],
            ['valid_min', -1000],
            ['valid_max', 10000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Fraction of the pixel that is water.""")],
        ])],
    ['water_frac_uncert',
     odict([['dtype', 'f4'],
            ['long_name', 'uncertainty in the water fraction'],
            ['grid_mapping', 'crs'],
            ['units', '1'],
            ['valid_min', 0],
            ['valid_max', 999999],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                1-sigma uncertainty in the water fraction.""")],
        ])],
    ['sig0',
     odict([['dtype', 'f4'],
            ['long_name', 'sigma0'],
            ['grid_mapping', 'crs'],
            ['units', '1'],
            ['quality_flag', 'sig0_qual'],
            ['valid_min', -1000],
            ['valid_max', 10000000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Normalized radar cross section (sigma0) in real, linear units
                (not decibels). The value may be negative due to noise
                subtraction.""")],
        ])],
    ['sig0_qual',
     odict([['dtype', 'u1'],
            ['long_name', 'summary quality indicator for the sigma0'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', 'good suspect degraded bad'],
            ['flag_values', np.array([0, 1, 2, 3]).astype('u1')],
            ['valid_min', 0],
            ['valid_max', 3],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Summary quality indicator for the sigma0 quantities.
                A value of 0 indicates a nominal measurement, 1 indicates a
                suspect measurement, 2 indicates a degraded measurement,
                and 3 indicates a bad measurement.""")],
        ])],
    ['sig0_qual_bitwise',
     odict([['dtype', 'u4'],
            ['long_name', 'bitwise quality indicator for the sigma0'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', textjoin("""
                sig0_qual_suspect
                classification_qual_suspect
                geolocation_qual_suspect
                large_uncert_suspect
                bright_land
                low_coherence_water_suspect
                few_pixels
                far_range_suspect
                near_range_suspect
                sig0_qual_degraded
                classification_qual_degraded
                geolocation_qual_degraded
                value_bad
                no_pixels
                outside_scene_bounds
                inner_swath
                missing_karin_data""")],
            ['flag_masks', np.array([
                QUAL_IND_SIG0_QUAL_SUSPECT,
                QUAL_IND_CLASS_QUAL_SUSPECT,
                QUAL_IND_GEOLOCATION_QUAL_SUSPECT,
                QUAL_IND_LARGE_UNCERT_SUSPECT,
                QUAL_IND_BRIGHT_LAND,
                QUAL_IND_LOW_COHERENCE_WATER_SUSPECT,
                QUAL_IND_FEW_PIXELS,
                QUAL_IND_FAR_RANGE_SUSPECT,
                QUAL_IND_NEAR_RANGE_SUSPECT,
                QUAL_IND_SIG0_QUAL_DEGRADED,
                QUAL_IND_CLASS_QUAL_DEGRADED,
                QUAL_IND_GEOLOCATION_QUAL_DEGRADED,
                QUAL_IND_VALUE_BAD,
                QUAL_IND_NO_PIXELS,
                QUAL_IND_OUTSIDE_SCENE_BOUNDS,
                QUAL_IND_INNER_SWATH,
                QUAL_IND_MISSING_KARIN_DATA
            ]).astype('u4')],
            ['valid_min', 0],
            ['valid_max', 4044255655],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Bitwise quality indicator for the sigma0 quantities.
                If this word is interpreted as an unsigned integer, a value of 0
                indicates good data, positive values less than 32768 represent
                suspect data, values greater than or equal to 32768 but
                less than 8388608 represent degraded data, and values
                greater than or equal to 8388608 represent bad data.""")],
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
                1-sigma uncertainty in sigma0. The value is provided in
                linear units. This value is a one-sigma additive
                (not multiplicative) uncertainty term, which can be added to or
                subtracted from sigma0.""")],
        ])],
    ['inc',
     odict([['dtype', 'f4'],
            ['long_name', 'incidence angle'],
            ['grid_mapping', 'crs'],
            ['units', 'degrees'],
            ['valid_min', 0],
            ['valid_max', 90],
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
    ['illumination_time',
     odict([['dtype', 'f8'],
            ['long_name', 'time of illumination of each pixel (UTC)'],
            ['standard_name','time'],
            ['calendar','gregorian'],
            ['tai_utc_difference', '[Value of TAI-UTC at time of first record]'],
            ['leap_second','YYYY-MM-DDThh:mm:ssZ'],
            ['units', 'seconds since 2000-01-01 00:00:00.000'],
            ['comment', textjoin("""
                Time of measurement in seconds
                in the UTC time scale since 1 Jan 2000 00:00:00 UTC.
                [tai_utc_difference] is the difference between TAI
                and UTC reference time (seconds) for the first
                measurement of the data set. If a leap second occurs
                within the data set, the attribute leap_second is set
                to the UTC time at which the leap second occurs.""")],
        ])],
    ['illumination_time_tai',
     odict([['dtype', 'f8'],
            ['long_name', 'time of illumination of each pixel (TAI)'],
            ['standard_name','time'],
            ['calendar','gregorian'],
            ['units', 'seconds since 2000-01-01 00:00:00.000'],
            ['comment', textjoin("""
                Time of measurement in seconds
                in the TAI time scale since 1 Jan 2000 00:00:00 TAI.
                This time scale contains no leap seconds. The
                difference (in seconds) with time in UTC is given
                by the attribute [illumination_time:tai_utc_difference].""")],
        ])],
    ['n_wse_pix',
     odict([['dtype', 'u4'],
            ['long_name', 'number of water surface elevation pixels'],
            ['grid_mapping', 'crs'],
            ['units', 'l'],
            ['valid_min', 0],
            ['valid_max', 999999],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Number of pixel cloud samples used in water surface elevation
                aggregation.""")],
        ])],
    ['n_water_area_pix',
     odict([['dtype', 'u4'],
            ['long_name', 'number of water surface area pixels'],
            ['grid_mapping', 'crs'],
            ['units', 'l'],
            ['valid_min', 0],
            ['valid_max', 999999],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Number of pixel cloud samples used in water surface area and
                water fraction aggregation.""")],
        ])],
    ['n_sig0_pix',
     odict([['dtype', 'u4'],
            ['long_name', 'number of sigma0 pixels'],
            ['grid_mapping', 'crs'],
            ['units', 'l'],
            ['valid_min', 0],
            ['valid_max', 999999],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Number of pixel cloud samples used in sigma0 aggregation.""")],
        ])],
    ['n_other_pix',
     odict([['dtype', 'u4'],
            ['long_name', 'number of other pixels'],
            ['grid_mapping', 'crs'],
            ['units', 'l'],
            ['valid_min', 0],
            ['valid_max', 999999],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Number of pixel cloud samples used in aggregation of
                quantities not related to water surface elevation,
                water surface area, water fraction or sigma0.""")],
        ])],
    ['dark_frac',
     odict([['dtype', 'f4'],
            ['long_name', 'fractional area of dark water'],
            ['grid_mapping', 'crs'],
            ['units', 'l'],
            ['valid_min', -1000],
            ['valid_max', 10000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Fraction of pixel water surface area covered by dark water.""")],
        ])],
    ['ice_clim_flag',
     odict([['dtype', 'u1'],
            ['long_name', 'climatological ice cover flag'],
            ['standard_name', 'status_flag'],
            ['source', 'UNC'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', textjoin("""
                no_ice_cover uncertain_ice_cover full_ice_cover""")],
            ['flag_values', np.array([0, 1, 2]).astype('u1')],
            ['valid_min', 0],
            ['valid_max', 2],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Climatological ice cover flag indicating whether the pixel is
                ice-covered on the day of the observation based on external
                climatological information (not the SWOT measurement). Values of
                0, 1, and 2 indicate that the pixel is likely not ice covered,
                may or may not be partially or fully ice covered, and likely
                fully ice covered, respectively.""")],
        ])],
    ['ice_dyn_flag',
     odict([['dtype', 'u1'],
            ['long_name', 'dynamic ice cover flag'],
            ['standard_name', 'status_flag'],
            ['source', 'UNC'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', textjoin("""
                no_ice_cover partial_ice_cover full_ice_cover""")],
            ['flag_values', np.array([0, 1, 2]).astype('u1')],
            ['valid_min', 0],
            ['valid_max', 2],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Dynamic ice cover flag indicating whether the surface is
                ice-covered on the day of the observation based on
                analysis of external satellite optical data.  Values of
                0, 1, and 2 indicate that the pixel is not ice covered,
                partially ice covered, and fully ice covered, respectively.""")],
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
                Estimate of the water surface elevation error caused by layover.""")],
        ])],
    ['sig0_cor_atmos_model',
     odict([['dtype', 'f4'],
            ['long_name', textjoin("""
                two-way atmospheric correction to sigma0 from model""")],
            ['source', 'European Centre for Medium-Range Weather Forecasts'],
            ['institution', 'ECMWF'],
            ['grid_mapping', 'crs'],
            ['units', '1'],
            ['valid_min', 1],
            ['valid_max', 10],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Atmospheric correction to sigma0 from weather model data as
                a linear power multiplier (not decibels).
                sig0_cor_atmos_model is already applied in computing sig0.""")],
        ])],
    ['height_cor_xover',
     odict([['dtype', 'f4'],
            ['long_name', 'height correction from KaRIn crossovers'],
            ['grid_mapping', 'crs'],
            ['units', 'm'],
            ['valid_min', -10],
            ['valid_max', 10],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Height correction from KaRIn crossover calibration. The
                correction is applied before geolocation but reported as
                an equivalent height correction.""")],
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
    ['load_tide_fes',
     odict([['dtype', 'f4'],
            ['long_name', 'geocentric load tide height (FES)'],
            ['source', 'FES2014b (Carrere et al., 2016)'],
            ['institution', 'LEGOS/CNES'],
            ['grid_mapping', 'crs'],
            ['units', 'm'],
            ['valid_min', -0.2],
            ['valid_max', 0.2],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Geocentric load tide height. The effect of the ocean tide
                loading of the Earth’s crust.""")],
        ])],
    ['load_tide_got',
     odict([['dtype', 'f4'],
            ['long_name', 'geocentric load tide height (GOT)'],
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
                The reported water surface elevation, latitude and longitude are
                computed after adding negative media corrections to
                uncorrected range along slant-range paths, accounting for
                the differential delay between the two KaRIn antennas. The
                equivalent vertical correction is computed by applying
                obliquity factors to the slant-path correction. Adding the
                reported correction to the reported water surface elevation
                results in the uncorrected pixel height.""")],
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
                The reported water surface elevation, latitude and longitude are
                computed after adding negative media corrections to
                uncorrected range along slant-range paths, accounting for
                the differential delay between the two KaRIn antennas. The
                equivalent vertical correction is computed by applying
                obliquity factors to the slant-path correction. Adding the
                reported correction to the reported water surface elevation
                results in the uncorrected pixel height.""")],
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
                The reported water surface elevation, latitude and longitude are
                computed after adding negative media corrections to
                uncorrected range along slant-range paths, accounting for
                the differential delay between the two KaRIn antennas. The
                equivalent vertical correction is computed by applying
                obliquity factors to the slant-path correction. Adding the
                reported correction to the reported water surface elevation
                results in the uncorrected pixel height.""")],
        ])],
])


class RasterUTM(ProductTesterMixIn, Product):
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
        ['platform', COMMON_ATTRIBUTES['platform']],
        ['references', COMMON_ATTRIBUTES['references']],
        ['reference_document', COMMON_ATTRIBUTES['reference_document']],
        ['contact', COMMON_ATTRIBUTES['contact']],
        ['cycle_number', COMMON_ATTRIBUTES['cycle_number']],
        ['pass_number', COMMON_ATTRIBUTES['pass_number']],
        ['scene_number', COMMON_ATTRIBUTES['scene_number']],
        ['tile_numbers', COMMON_ATTRIBUTES['tile_numbers']],
        ['tile_names', COMMON_ATTRIBUTES['tile_names']],
        ['tile_polarizations', COMMON_ATTRIBUTES['tile_polarizations']],
        ['coordinate_reference_system',
         {'dtype': COMMON_ATTRIBUTES['coordinate_reference_system']['dtype'],
          'value':'Universal Transverse Mercator',
          'docstr': COMMON_ATTRIBUTES['coordinate_reference_system']['docstr']}],
        ['resolution', COMMON_ATTRIBUTES['resolution']],
        ['short_name', COMMON_ATTRIBUTES['short_name']],
        ['descriptor_string', COMMON_ATTRIBUTES['descriptor_string']],
        ['crid', COMMON_ATTRIBUTES['crid']],
        ['product_version', COMMON_ATTRIBUTES['product_version']],
        ['pge_name', COMMON_ATTRIBUTES['pge_name']],
        ['pge_version', COMMON_ATTRIBUTES['pge_version']],
        ['time_granule_start', COMMON_ATTRIBUTES['time_granule_start']],
        ['time_granule_end', COMMON_ATTRIBUTES['time_granule_end']],
        ['time_coverage_start', COMMON_ATTRIBUTES['time_coverage_start']],
        ['time_coverage_end', COMMON_ATTRIBUTES['time_coverage_end']],
        ['geospatial_lon_min', COMMON_ATTRIBUTES['geospatial_lon_min']],
        ['geospatial_lon_max', COMMON_ATTRIBUTES['geospatial_lon_max']],
        ['geospatial_lat_min', COMMON_ATTRIBUTES['geospatial_lat_min']],
        ['geospatial_lat_max', COMMON_ATTRIBUTES['geospatial_lat_max']],
        ['left_first_longitude', COMMON_ATTRIBUTES['left_first_longitude']],
        ['left_first_latitude', COMMON_ATTRIBUTES['left_first_latitude']],
        ['left_last_longitude', COMMON_ATTRIBUTES['left_last_longitude']],
        ['left_last_latitude', COMMON_ATTRIBUTES['left_first_latitude']],
        ['right_first_longitude', COMMON_ATTRIBUTES['right_first_longitude']],
        ['right_first_latitude', COMMON_ATTRIBUTES['right_first_latitude']],
        ['right_last_longitude', COMMON_ATTRIBUTES['right_last_longitude']],
        ['right_last_latitude', COMMON_ATTRIBUTES['right_last_latitude']],
        ['xref_l2_hr_pixc_files', COMMON_ATTRIBUTES['xref_l2_hr_pixc_files']],
        ['xref_l2_hr_pixcvec_files', COMMON_ATTRIBUTES['xref_l2_hr_pixcvec_files']],
        ['xref_param_l2_hr_raster_file', COMMON_ATTRIBUTES['xref_param_l2_hr_raster_file']],
        ['xref_reforbittrack_files', COMMON_ATTRIBUTES['xref_reforbittrack_files']],
        ['utm_zone_num', {'dtype': 'i2',
                          'docstr': 'UTM zone number.'}],
        ['mgrs_latitude_band', {'dtype': 'str',
                                'docstr': 'MGRS latitude band.'}],
        ['x_min', {'dtype': 'f8',
                   'docstr': 'Projected minimum x (easting) coordinate.'}],
        ['x_max', {'dtype': 'f8',
                   'docstr': 'Projected maximum x (easting) coordinate.'}],
        ['y_min', {'dtype': 'f8',
                   'docstr': 'Projected minimum y (northing) coordinate.'}],
        ['y_max', {'dtype': 'f8',
                   'docstr': 'Projected maximum y (northing) coordinate.'}],
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
                ['semi_major_axis', raster_crs.ELLIPSOID_SEMI_MAJOR_AXIS],
                ['inverse_flattening', raster_crs.ELLIPSOID_INVERSE_FLATTENING],
                ['crs_wkt', '[OGS Well-Known Text string]'],
                ['spatial_ref', '[OGS Well-Known Text string]'],
                ['comment', 'UTM zone coordinate reference system.'],
         ])],
        ['x',
         odict([['dtype', 'f8'],
                ['long_name', 'x coordinate of projection'],
                ['standard_name', 'projection_x_coordinate'],
                ['units', 'm'],
                ['valid_min', -10000000],
                ['valid_max', 10000000],
                ['comment', textjoin("""
                    UTM easting coordinate of the pixel.""")],
         ])],
        ['y',
         odict([['dtype', 'f8'],
                ['long_name', 'y coordinate of projection'],
                ['standard_name', 'projection_y_coordinate'],
                ['units', 'm'],
                ['valid_min', -20000000],
                ['valid_max', 20000000],
                ['comment', textjoin("""
                    UTM northing coordinate of the pixel.""")],
         ])],
        ['longitude',
         odict([['dtype', 'f8'],
                ['long_name', 'longitude (degrees East)'],
                ['standard_name', 'longitude'],
                ['grid_mapping', 'crs'],
                ['units', 'degrees_east'],
                ['valid_min', -180],
                ['valid_max', 180],
                ['comment', textjoin("""
                    Geodetic longitude [-180,180) (east of the Greenwich meridian)
                    of the pixel.""")],
            ])],
        ['latitude',
         odict([['dtype', 'f8'],
                ['long_name', 'latitude (positive N, negative S)'],
                ['standard_name', 'latitude'],
                ['grid_mapping', 'crs'],
                ['units', 'degrees_north'],
                ['valid_min', -80],
                ['valid_max', 80],
                ['comment', textjoin("""
                    Geodetic latitude [-80,80] (degrees north of equator) of
                    the pixel.""")]
            ])],
        ['wse', COMMON_VARIABLES['wse'].copy()],
        ['wse_qual', COMMON_VARIABLES['wse_qual'].copy()],
        ['wse_qual_bitwise', COMMON_VARIABLES['wse_qual_bitwise'].copy()],
        ['wse_uncert', COMMON_VARIABLES['wse_uncert'].copy()],
        ['water_area', COMMON_VARIABLES['water_area'].copy()],
        ['water_area_qual', COMMON_VARIABLES['water_area_qual'].copy()],
        ['water_area_qual_bitwise', COMMON_VARIABLES['water_area_qual_bitwise'].copy()],
        ['water_area_uncert', COMMON_VARIABLES['water_area_uncert'].copy()],
        ['water_frac', COMMON_VARIABLES['water_frac'].copy()],
        ['water_frac_uncert', COMMON_VARIABLES['water_frac_uncert'].copy()],
        ['sig0', COMMON_VARIABLES['sig0'].copy()],
        ['sig0_qual', COMMON_VARIABLES['sig0_qual'].copy()],
        ['sig0_qual_bitwise', COMMON_VARIABLES['sig0_qual_bitwise'].copy()],
        ['sig0_uncert', COMMON_VARIABLES['sig0_uncert'].copy()],
        ['inc', COMMON_VARIABLES['inc'].copy()],
        ['cross_track', COMMON_VARIABLES['cross_track'].copy()],
        ['illumination_time', COMMON_VARIABLES['illumination_time'].copy()],
        ['illumination_time_tai', COMMON_VARIABLES['illumination_time_tai'].copy()],
        ['n_wse_pix', COMMON_VARIABLES['n_wse_pix'].copy()],
        ['n_water_area_pix', COMMON_VARIABLES['n_water_area_pix'].copy()],
        ['n_sig0_pix', COMMON_VARIABLES['n_sig0_pix'].copy()],
        ['n_other_pix', COMMON_VARIABLES['n_other_pix'].copy()],
        ['dark_frac', COMMON_VARIABLES['dark_frac'].copy()],
        ['ice_clim_flag', COMMON_VARIABLES['ice_clim_flag'].copy()],
        ['ice_dyn_flag', COMMON_VARIABLES['ice_dyn_flag'].copy()],
        ['layover_impact', COMMON_VARIABLES['layover_impact'].copy()],
        ['sig0_cor_atmos_model', COMMON_VARIABLES['sig0_cor_atmos_model'].copy()],
        ['height_cor_xover', COMMON_VARIABLES['height_cor_xover'].copy()],
        ['geoid', COMMON_VARIABLES['geoid'].copy()],
        ['solid_earth_tide', COMMON_VARIABLES['solid_earth_tide'].copy()],
        ['load_tide_fes', COMMON_VARIABLES['load_tide_fes'].copy()],
        ['load_tide_got', COMMON_VARIABLES['load_tide_got'].copy()],
        ['pole_tide', COMMON_VARIABLES['pole_tide'].copy()],
        ['model_dry_tropo_cor', COMMON_VARIABLES['model_dry_tropo_cor'].copy()],
        ['model_wet_tropo_cor', COMMON_VARIABLES['model_wet_tropo_cor'].copy()],
        ['iono_cor_gim_ka', COMMON_VARIABLES['iono_cor_gim_ka'].copy()],
    ])

    for key in VARIABLES:
        VARIABLES[key]['coordinates'] = 'x y'
        VARIABLES[key]['dimensions'] = odict([['y', 0], ['x', 0]])

    VARIABLES['latitude']['coordinates'] = 'x y'
    VARIABLES['latitude']['dimensions'] = odict([['y', 0], ['x', 0]])
    VARIABLES['longitude']['coordinates'] = 'x y'
    VARIABLES['longitude']['dimensions'] = odict([['y', 0], ['x', 0]])

    VARIABLES['x']['dimensions'] = odict([['x', 0]])
    VARIABLES['y']['dimensions'] = odict([['y', 0]])
    VARIABLES['crs']['dimensions'] = odict([])

    def get_raster_mapping(self, pixc, mask, use_improved_geoloc=True,
                           max_chunk_size=DEFAULT_MAX_CHUNK_SIZE):
        """ Get the mapping of pixc points to raster bins """
        LOGGER.info('getting raster mapping')

        if use_improved_geoloc:
            lat_keyword = 'improved_latitude'
            lon_keyword = 'improved_longitude'
        else:
            lat_keyword = 'latitude'
            lon_keyword = 'longitude'

        mask = np.logical_and.reduce((mask,
            np.logical_not(np.ma.getmaskarray(pixc['pixel_cloud'][lat_keyword])),
            np.logical_not(np.ma.getmaskarray(pixc['pixel_cloud'][lon_keyword]))))

        pixc_lats = pixc['pixel_cloud'][lat_keyword][mask]
        pixc_lons = raster_crs.lon_360to180(
            pixc['pixel_cloud'][lon_keyword][mask])
        pixc_idx = np.where(mask)[0]
        nb_pix = pixc_lats.size

        input_crs = raster_crs.wgs84_crs()
        output_crs = raster_crs.utm_crs(
            self.utm_zone_num, self.mgrs_latitude_band)
        transf = osr.CoordinateTransformation(input_crs, output_crs)

        # Split the data into more manageable chunks and convert to UTM
        transf_points = []
        for start_idx in np.arange(0, nb_pix, max_chunk_size):
            end_idx = min(start_idx + max_chunk_size, nb_pix)
            points = [(lat, lon) for lat, lon in zip(
                pixc_lats[start_idx:end_idx], pixc_lons[start_idx:end_idx])]
            transf_points.extend(transf.TransformPoints(points))

        mapping_tmp = []
        for i in range(0, self.dimensions['y']):
            mapping_tmp.append([])
            for j in range(0, self.dimensions['x']):
                mapping_tmp[i].append([])

        ys = np.array([point[1] for point in transf_points])
        xs = np.array([point[0] for point in transf_points])
        i_tmp = np.round((ys - self.y_min) / self.resolution).astype(int)
        j_tmp = np.round((xs - self.x_min) / self.resolution).astype(int)

        idx_mask = np.logical_and.reduce((
            i_tmp >= 0, i_tmp < self.dimensions['y'],
            j_tmp >= 0, j_tmp < self.dimensions['x']))

        for i,j,m,x in zip(i_tmp, j_tmp, idx_mask, pixc_idx):
            if m: mapping_tmp[i][j].append(x)

        return mapping_tmp

    def crop_to_bounds(self, swath_polygon_points):
        """ Crop raster to the given swath polygon """
        LOGGER.info('cropping to bounds')

        # Convert polygon points to UTM
        input_crs = raster_crs.wgs84_crs()
        output_crs = raster_crs.utm_crs(
            self.utm_zone_num, self.mgrs_latitude_band)

        transf = osr.CoordinateTransformation(input_crs, output_crs)
        transf_points = transf.TransformPoints(swath_polygon_points)
        swath_polygon_points_utm = [point[:2] for point in transf_points]

        poly_prep = prep(Polygon(swath_polygon_points_utm))

        # Check whether each pixel center intersects with the polygon
        mask = np.zeros((self.dimensions['y'], self.dimensions['x']))
        for i in range(0, self.dimensions['y']):
            for j in range(0, self.dimensions['x']):
                pt = Point(self.x[j], self.y[i])
                mask[i][j] = poly_prep.intersects(pt)

        # Mask the datasets
        for var in self.variables:
            if var in ['wse_qual_bitwise', 'water_area_qual_bitwise',
                       'sig0_qual_bitwise']:
                self.variables[var][np.logical_not(mask)] = \
                    QUAL_IND_OUTSIDE_SCENE_BOUNDS + QUAL_IND_NO_PIXELS \
                    + QUAL_IND_FEW_PIXELS
            elif var in ['wse_qual', 'water_area_qual', 'sig0_qual']:
                self.variables[var][np.logical_not(mask)] = \
                    QUAL_IND_BAD
            elif var in ['n_wse_pix', 'n_water_area_pix', 'n_sig0_pix',
                         'n_other_pix']:
                self.variables[var][np.logical_not(mask)] = 0
            elif var not in ['crs', 'x', 'y']:
                self.variables[var].mask = np.logical_or(
                    self.variables[var].mask, np.logical_not(mask))

        # Set the time coverage start and end
        if np.all(self.illumination_time.mask):
            start_illumination_time = EMPTY_DATETIME
            end_illumination_time = EMPTY_DATETIME
        else:
            start_illumination_time = np.min(self.illumination_time)
            end_illumination_time = np.max(self.illumination_time)
            start_time = datetime.utcfromtimestamp(
                (SWOT_EPOCH-UNIX_EPOCH).total_seconds() \
                + start_illumination_time)
            end_time = datetime.utcfromtimestamp(
                (SWOT_EPOCH-UNIX_EPOCH).total_seconds() \
                + end_illumination_time)
            self.time_coverage_start = start_time.strftime(DATETIME_FORMAT_STR)
            self.time_coverage_end = end_time.strftime(DATETIME_FORMAT_STR)

    def get_uncorrected_height(self):
        """ Get the height with wse geophysical corrections removed """
        LOGGER.info('getting uncorrected height')

        height = self.wse + (
            self.geoid +
            self.solid_earth_tide +
            self.load_tide_fes +
            self.pole_tide)
        return height

    def is_empty(self):
        """ Check if the raster is empty """

        for variable in COMMON_VARIABLES:
            var_data = getattr(self, variable)
            if np.logical_not(np.all(var_data.mask)):
                return 0
        return 1


class RasterGeo(ProductTesterMixIn, Product):
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
        ['platform', COMMON_ATTRIBUTES['platform']],
        ['references', COMMON_ATTRIBUTES['references']],
        ['reference_document', COMMON_ATTRIBUTES['reference_document']],
        ['contact', COMMON_ATTRIBUTES['contact']],
        ['cycle_number', COMMON_ATTRIBUTES['cycle_number']],
        ['pass_number', COMMON_ATTRIBUTES['pass_number']],
        ['scene_number', COMMON_ATTRIBUTES['scene_number']],
        ['tile_numbers', COMMON_ATTRIBUTES['tile_numbers']],
        ['tile_names', COMMON_ATTRIBUTES['tile_names']],
        ['tile_polarizations', COMMON_ATTRIBUTES['tile_polarizations']],
        ['coordinate_reference_system',
         {'dtype': COMMON_ATTRIBUTES['coordinate_reference_system']['dtype'],
          'value':'Geodetic Latitude/Longitude',
          'docstr': COMMON_ATTRIBUTES['coordinate_reference_system']['docstr']}],
        ['resolution', COMMON_ATTRIBUTES['resolution']],
        ['short_name', COMMON_ATTRIBUTES['short_name']],
        ['descriptor_string', COMMON_ATTRIBUTES['descriptor_string']],
        ['crid', COMMON_ATTRIBUTES['crid']],
        ['product_version', COMMON_ATTRIBUTES['product_version']],
        ['pge_name', COMMON_ATTRIBUTES['pge_name']],
        ['pge_version', COMMON_ATTRIBUTES['pge_version']],
        ['time_granule_start', COMMON_ATTRIBUTES['time_granule_start']],
        ['time_granule_end', COMMON_ATTRIBUTES['time_granule_end']],
        ['time_coverage_start', COMMON_ATTRIBUTES['time_coverage_start']],
        ['time_coverage_end', COMMON_ATTRIBUTES['time_coverage_end']],
        ['geospatial_lon_min', COMMON_ATTRIBUTES['geospatial_lon_min']],
        ['geospatial_lon_max', COMMON_ATTRIBUTES['geospatial_lon_max']],
        ['geospatial_lat_min', COMMON_ATTRIBUTES['geospatial_lat_min']],
        ['geospatial_lat_max', COMMON_ATTRIBUTES['geospatial_lat_max']],
        ['left_first_longitude', COMMON_ATTRIBUTES['left_first_longitude']],
        ['left_first_latitude', COMMON_ATTRIBUTES['left_first_latitude']],
        ['left_last_longitude', COMMON_ATTRIBUTES['left_last_longitude']],
        ['left_last_latitude', COMMON_ATTRIBUTES['left_first_latitude']],
        ['right_first_longitude', COMMON_ATTRIBUTES['right_first_longitude']],
        ['right_first_latitude', COMMON_ATTRIBUTES['right_first_latitude']],
        ['right_last_longitude', COMMON_ATTRIBUTES['right_last_longitude']],
        ['right_last_latitude', COMMON_ATTRIBUTES['right_last_latitude']],
        ['xref_l2_hr_pixc_files', COMMON_ATTRIBUTES['xref_l2_hr_pixc_files']],
        ['xref_l2_hr_pixcvec_files', COMMON_ATTRIBUTES['xref_l2_hr_pixcvec_files']],
        ['xref_param_l2_hr_raster_file', COMMON_ATTRIBUTES['xref_param_l2_hr_raster_file']],
        ['xref_reforbittrack_files', COMMON_ATTRIBUTES['xref_reforbittrack_files']],
        ['longitude_min', {'dtype': 'f8',
                           'docstr': 'Minimum longitude coordinate.'}],
        ['longitude_max', {'dtype': 'f8',
                           'docstr': 'Maximum longitude coordinate.'}],
        ['latitude_min', {'dtype': 'f8',
                          'docstr': 'Minimum latitude coordinate.'}],
        ['latitude_max', {'dtype': 'f8',
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
                ['semi_major_axis', raster_crs.ELLIPSOID_SEMI_MAJOR_AXIS],
                ['inverse_flattening', raster_crs.ELLIPSOID_INVERSE_FLATTENING],
                ['crs_wkt', '[OGS Well-Known Text string]'],
                ['spatial_ref', '[OGS Well-Known Text string]'],
                ['comment', 'Geodetic lat/lon coordinate reference system.'],
        ])],
        ['longitude',
         odict([['dtype', 'f8'],
                ['long_name', 'longitude (degrees East)'],
                ['standard_name', 'longitude'],
                ['units', 'degrees_east'],
                ['valid_min', -180],
                ['valid_max', 180],
                ['comment', textjoin("""
                    Geodetic longitude [-180,180) (east of the Greenwich meridian)
                    of the pixel.""")],
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
                    the pixel.""")]
        ])],
        ['wse', COMMON_VARIABLES['wse'].copy()],
        ['wse_qual', COMMON_VARIABLES['wse_qual'].copy()],
        ['wse_qual_bitwise', COMMON_VARIABLES['wse_qual_bitwise'].copy()],
        ['wse_uncert', COMMON_VARIABLES['wse_uncert'].copy()],
        ['water_area', COMMON_VARIABLES['water_area'].copy()],
        ['water_area_qual', COMMON_VARIABLES['water_area_qual'].copy()],
        ['water_area_qual_bitwise', COMMON_VARIABLES['water_area_qual_bitwise'].copy()],
        ['water_area_uncert', COMMON_VARIABLES['water_area_uncert'].copy()],
        ['water_frac', COMMON_VARIABLES['water_frac'].copy()],
        ['water_frac_uncert', COMMON_VARIABLES['water_frac_uncert'].copy()],
        ['sig0', COMMON_VARIABLES['sig0'].copy()],
        ['sig0_qual', COMMON_VARIABLES['sig0_qual'].copy()],
        ['sig0_qual_bitwise', COMMON_VARIABLES['sig0_qual_bitwise'].copy()],
        ['sig0_uncert', COMMON_VARIABLES['sig0_uncert'].copy()],
        ['inc', COMMON_VARIABLES['inc'].copy()],
        ['cross_track', COMMON_VARIABLES['cross_track'].copy()],
        ['illumination_time', COMMON_VARIABLES['illumination_time'].copy()],
        ['illumination_time_tai', COMMON_VARIABLES['illumination_time_tai'].copy()],
        ['n_wse_pix', COMMON_VARIABLES['n_wse_pix'].copy()],
        ['n_water_area_pix', COMMON_VARIABLES['n_water_area_pix'].copy()],
        ['n_sig0_pix', COMMON_VARIABLES['n_sig0_pix'].copy()],
        ['n_other_pix', COMMON_VARIABLES['n_other_pix'].copy()],
        ['dark_frac', COMMON_VARIABLES['dark_frac'].copy()],
        ['ice_clim_flag', COMMON_VARIABLES['ice_clim_flag'].copy()],
        ['ice_dyn_flag', COMMON_VARIABLES['ice_dyn_flag'].copy()],
        ['layover_impact', COMMON_VARIABLES['layover_impact'].copy()],
        ['sig0_cor_atmos_model', COMMON_VARIABLES['sig0_cor_atmos_model'].copy()],
        ['height_cor_xover', COMMON_VARIABLES['height_cor_xover'].copy()],
        ['geoid', COMMON_VARIABLES['geoid'].copy()],
        ['solid_earth_tide', COMMON_VARIABLES['solid_earth_tide'].copy()],
        ['load_tide_fes', COMMON_VARIABLES['load_tide_fes'].copy()],
        ['load_tide_got', COMMON_VARIABLES['load_tide_got'].copy()],
        ['pole_tide', COMMON_VARIABLES['pole_tide'].copy()],
        ['model_dry_tropo_cor', COMMON_VARIABLES['model_dry_tropo_cor'].copy()],
        ['model_wet_tropo_cor', COMMON_VARIABLES['model_wet_tropo_cor'].copy()],
        ['iono_cor_gim_ka', COMMON_VARIABLES['iono_cor_gim_ka'].copy()],
    ])

    for key in VARIABLES:
        VARIABLES[key]['coordinates'] = 'longitude latitude'
        VARIABLES[key]['dimensions'] = odict([['latitude', 0], ['longitude', 0]])

    VARIABLES['longitude']['dimensions'] = odict([['longitude', 0]])
    VARIABLES['latitude']['dimensions'] = odict([['latitude', 0]])
    VARIABLES['crs']['dimensions'] = odict([])

    def get_raster_mapping(self, pixc, mask, use_improved_geoloc=True):
        """ Get the mapping of pixc points to raster bins """
        LOGGER.info('getting raster mapping')

        if use_improved_geoloc:
            lat_keyword = 'improved_latitude'
            lon_keyword = 'improved_longitude'
        else:
            lat_keyword = 'latitude'
            lon_keyword = 'longitude'

        mask = np.logical_and.reduce((mask,
            np.logical_not(np.ma.getmaskarray(pixc['pixel_cloud'][lat_keyword])),
            np.logical_not(np.ma.getmaskarray(pixc['pixel_cloud'][lon_keyword]))))

        pixc_lats = pixc['pixel_cloud'][lat_keyword][mask]
        pixc_lons = raster_crs.lon_360to180(
            pixc['pixel_cloud'][lon_keyword][mask])
        pixc_idx = np.where(mask)[0]

        mapping_tmp = []
        for i in range(0, self.dimensions['latitude']):
            mapping_tmp.append([])
            for j in range(0, self.dimensions['longitude']):
                mapping_tmp[i].append([])

        i_tmp = np.round(
            (pixc_lats - self.latitude_min)/self.resolution).astype(int)

        # Handle longitude wrap
        lons_diff = pixc_lons - self.longitude_min
        wrapped_mask = lons_diff < -180
        j_tmp = np.round(lons_diff / self.resolution).astype(int)
        j_tmp[wrapped_mask] = np.round(
            (lons_diff[wrapped_mask] + 360)/self.resolution).astype(int)

        idx_mask = np.logical_and.reduce((
            i_tmp >= 0, i_tmp < self.dimensions['latitude'],
            j_tmp >= 0, j_tmp < self.dimensions['longitude']))

        for i,j,m,x in zip(i_tmp, j_tmp, idx_mask, pixc_idx):
            if m: mapping_tmp[i][j].append(x)

        return mapping_tmp

    def crop_to_bounds(self, swath_polygon_points):
        """ Crop raster to the given swath polygon """
        LOGGER.info('cropping to bounds')

        poly = Polygon([[point[1], point[0]] for point in swath_polygon_points])

        # Handle longitude wrap
        poly_prep = prep(raster_crs.split_wrapped_longitude_polygon(poly))

        # Check whether each pixel center intersects with split_polys
        mask = np.zeros((self.dimensions['latitude'],
                         self.dimensions['longitude']), dtype=bool)
        for i in range(0, self.dimensions['latitude']):
            for j in range(0, self.dimensions['longitude']):
                pt = Point(self.longitude[j], self.latitude[i])
                mask[i][j] = poly_prep.intersects(pt)

        # Mask the datasets
        for var in self.variables:
            if var in ['wse_qual_bitwise', 'water_area_qual_bitwise',
                       'sig0_qual_bitwise']:
                self.variables[var][np.logical_not(mask)] = \
                    QUAL_IND_OUTSIDE_SCENE_BOUNDS + QUAL_IND_NO_PIXELS \
                    + QUAL_IND_FEW_PIXELS
            elif var in ['wse_qual', 'water_area_qual', 'sig0_qual']:
                self.variables[var][np.logical_not(mask)] = \
                    QUAL_IND_BAD
            elif var in ['n_wse_pix', 'n_water_area_pix', 'n_sig0_pix',
                         'n_other_pix']:
                self.variables[var][np.logical_not(mask)] = 0
            elif var not in ['crs', 'longitude', 'latitude']:
                self.variables[var].mask = np.logical_or(
                    self.variables[var].mask, np.logical_not(mask))

        # Set the time coverage start and end
        if np.all(self.illumination_time.mask):
            start_illumination_time = EMPTY_DATETIME
            end_illumination_time = EMPTY_DATETIME
        else:
            start_illumination_time = np.min(self.illumination_time)
            end_illumination_time = np.max(self.illumination_time)
            start_time = datetime.utcfromtimestamp(
                (SWOT_EPOCH-UNIX_EPOCH).total_seconds() \
                + start_illumination_time)
            end_time = datetime.utcfromtimestamp(
                (SWOT_EPOCH-UNIX_EPOCH).total_seconds() \
                + end_illumination_time)
            self.time_coverage_start = start_time.strftime(DATETIME_FORMAT_STR)
            self.time_coverage_end = end_time.strftime(DATETIME_FORMAT_STR)

    def get_uncorrected_height(self):
        """ Get the height with wse geophysical corrections removed """
        LOGGER.info('getting uncorrected height')

        height = self.wse + (
            self.geoid +
            self.solid_earth_tide +
            self.load_tide_fes +
            self.pole_tide)
        return height

    def is_empty(self):
        """ Check if the raster is empty """

        for variable in COMMON_VARIABLES:
            var_data = getattr(self, variable)
            if np.logical_not(np.all(var_data.mask)):
                return 0
        return 1


class RasterUTMDebug(RasterUTM):
    ATTRIBUTES = odict({key:RasterUTM.ATTRIBUTES[key].copy()
                        for key in RasterUTM.ATTRIBUTES})
    DIMENSIONS = odict({key:RasterUTM.DIMENSIONS[key]
                        for key in RasterUTM.DIMENSIONS})
    VARIABLES = odict({key:RasterUTM.VARIABLES[key].copy()
                        for key in RasterUTM.VARIABLES})
    VARIABLES.update(odict([
        ['classification',
         odict([['dtype', 'u1']])],
    ]))
    for key in VARIABLES:
        VARIABLES[key]['coordinates'] = 'x y'
        VARIABLES[key]['dimensions'] = odict([['y', 0], ['x', 0]])

    VARIABLES['x']['dimensions'] = odict([['x', 0]])
    VARIABLES['y']['dimensions'] = odict([['y', 0]])
    VARIABLES['crs']['dimensions'] = odict([])
    VARIABLES['classification']['dimensions'] = \
        odict([['y', 0], ['x', 0]])


class RasterGeoDebug(RasterGeo):
    ATTRIBUTES = odict({key:RasterGeo.ATTRIBUTES[key].copy()
                        for key in RasterGeo.ATTRIBUTES})
    DIMENSIONS = odict({key:RasterGeo.DIMENSIONS[key]
                        for key in RasterGeo.DIMENSIONS})
    VARIABLES = odict({key:RasterGeo.VARIABLES[key].copy()
                        for key in RasterGeo.VARIABLES})
    VARIABLES.update(odict([
        ['classification',
         odict([['dtype', 'u1']])],
    ]))
    for key in VARIABLES:
        VARIABLES[key]['coordinates'] = 'longitude latitude'
        VARIABLES[key]['dimensions'] = odict([['latitude', 0], ['longitude', 0]])

    VARIABLES['longitude']['dimensions'] = odict([['longitude', 0]])
    VARIABLES['latitude']['dimensions'] = odict([['latitude', 0]])
    VARIABLES['crs']['dimensions'] = odict([])
    VARIABLES['classification']['dimensions'] = \
        odict([['latitude', 0], ['longitude', 0]])


class ScenePixc(Product):
    ATTRIBUTES = odict([
        ['scene_cycle_number', odict([])],
        ['scene_pass_number', odict([])],
        ['scene_number', odict([])],
        ['cycle_numbers', odict([])],
        ['pass_numbers', odict([])],
        ['tile_numbers', odict([])],
        ['tile_names', odict([])],
        ['tile_polarizations', odict([])],
        ['time_granule_start', odict([])],
        ['time_granule_end', odict([])],
        ['time_coverage_start', odict([])],
        ['time_coverage_end', odict([])],
        ['left_time_granule_start', odict([])],
        ['left_time_granule_end', odict([])],
        ['right_time_granule_start', odict([])],
        ['right_time_granule_end', odict([])],
        ['left_time_coverage_start', odict([])],
        ['left_time_coverage_end', odict([])],
        ['right_time_coverage_start', odict([])],
        ['right_time_coverage_end', odict([])],
        ['wavelength', odict([])],
        ['near_range', odict([])],
        ['nominal_slant_range_spacing', odict([])],
        ['left_first_longitude', odict([])],
        ['left_last_longitude', odict([])],
        ['left_first_latitude', odict([])],
        ['left_last_latitude', odict([])],
        ['right_first_longitude', odict([])],
        ['right_last_longitude', odict([])],
        ['right_first_latitude', odict([])],
        ['right_last_latitude', odict([])],
        ['geospatial_lon_min', odict([])],
        ['geospatial_lon_max', odict([])],
        ['geospatial_lat_min', odict([])],
        ['geospatial_lat_max', odict([])],
        ['leap_second', odict([])],
    ])
    GROUPS = odict([
        ['pixel_cloud', 'ScenePixelCloud'],
        ['tvp', 'SceneTVP'],
    ])

    @classmethod
    def from_tile(cls, pixc_tile, pixcvec_tile=None, mask=None):
        """ Construct self from a single pixc tile (and associated pixcvec tile) """
        LOGGER.info('constructing scene pixc from tile')

        scene_pixc = cls()

        # Copy over attributes
        scene_pixc.scene_cycle_number = pixc_tile.cycle_number
        scene_pixc.scene_pass_number = pixc_tile.pass_number
        scene_pixc.scene_number = np.ceil(
            pixc_tile.tile_number/NONOVERLAP_TILES_PER_SIDE).astype('i2')
        scene_pixc.cycle_numbers = [pixc_tile.cycle_number]
        scene_pixc.pass_numbers = [pixc_tile.pass_number]
        scene_pixc.tile_numbers = [pixc_tile.tile_number]
        scene_pixc.tile_names = pixc_tile.tile_name
        scene_pixc.tile_polarizations = pixc_tile.polarization
        scene_pixc.time_granule_start = pixc_tile.time_granule_start
        scene_pixc.time_granule_end = pixc_tile.time_granule_end
        scene_pixc.time_coverage_start = pixc_tile.time_coverage_start
        scene_pixc.time_coverage_end = pixc_tile.time_coverage_end
        scene_pixc.wavelength = pixc_tile.wavelength
        scene_pixc.near_range = pixc_tile.near_range
        scene_pixc.nominal_slant_range_spacing = \
            pixc_tile.nominal_slant_range_spacing

        swath_side = pixc_tile.swath_side

        if swath_side.lower() == 'l':
            scene_pixc.left_time_granule_start = pixc_tile.time_granule_start
            scene_pixc.left_time_granule_end = pixc_tile.time_granule_end
            scene_pixc.left_time_coverage_start = pixc_tile.time_coverage_start
            scene_pixc.left_time_coverage_end = pixc_tile.time_coverage_end
            scene_pixc.right_time_granule_start = None
            scene_pixc.right_time_granule_end = None
            scene_pixc.right_time_coverage_start = None
            scene_pixc.right_time_coverage_end = None

            scene_pixc.left_first_longitude = pixc_tile.outer_first_longitude
            scene_pixc.left_last_longitude = pixc_tile.outer_last_longitude
            scene_pixc.left_first_latitude = pixc_tile.outer_first_latitude
            scene_pixc.left_last_latitude = pixc_tile.outer_last_latitude
            scene_pixc.right_first_longitude = pixc_tile.inner_first_longitude
            scene_pixc.right_last_longitude = pixc_tile.inner_last_longitude
            scene_pixc.right_first_latitude = pixc_tile.inner_first_latitude
            scene_pixc.right_last_latitude = pixc_tile.inner_last_latitude

        elif swath_side.lower() == 'r':
            scene_pixc.right_time_granule_start = pixc_tile.time_granule_start
            scene_pixc.right_time_granule_end = pixc_tile.time_granule_end
            scene_pixc.right_time_coverage_start = pixc_tile.time_coverage_start
            scene_pixc.right_time_coverage_end = pixc_tile.time_coverage_end
            scene_pixc.left_time_granule_start = None
            scene_pixc.left_time_granule_end = None
            scene_pixc.left_time_coverage_start = None
            scene_pixc.left_time_coverage_end = None

            scene_pixc.left_first_longitude = pixc_tile.inner_first_longitude
            scene_pixc.left_last_longitude = pixc_tile.inner_last_longitude
            scene_pixc.left_first_latitude = pixc_tile.inner_first_latitude
            scene_pixc.left_last_latitude = pixc_tile.inner_last_latitude
            scene_pixc.right_first_longitude = pixc_tile.outer_first_longitude
            scene_pixc.right_last_longitude = pixc_tile.outer_last_longitude
            scene_pixc.right_first_latitude = pixc_tile.outer_first_latitude
            scene_pixc.right_last_latitude = pixc_tile.outer_last_latitude

        lats = [scene_pixc.left_first_latitude,
                scene_pixc.right_first_latitude,
                scene_pixc.left_last_latitude,
                scene_pixc.right_last_latitude]
        lons = [scene_pixc.left_first_latitude,
                scene_pixc.right_first_latitude,
                scene_pixc.left_last_latitude,
                scene_pixc.right_last_latitude]
        scene_pixc.geospatial_lat_min = min(lats)
        scene_pixc.geospatial_lat_max = max(lats)

        # Handle longitude wrap
        shifted_lons = raster_crs.shift_wrapped_longitude(lons)
        lon_min = min(shifted_lons)
        lon_max = max(shifted_lons)
        # Wrap to between -180 and 180 degrees longitude
        scene_pixc.geospatial_lon_min = raster_crs.lon_360to180(lon_min)
        scene_pixc.geospatial_lon_max = raster_crs.lon_360to180(lon_max)

        leap_second = pixc_tile.pixel_cloud.VARIABLES['illumination_time']['leap_second']
        if leap_second is not None and leap_second != 'YYYY-MM-DDThh:mm:ssZ':
            scene_pixc.leap_second = leap_second
        else:
            scene_pixc.leap_second = EMPTY_LEAPSEC

        # Copy over groups
        scene_pixc['pixel_cloud'] = ScenePixelCloud.from_tile(
            pixc_tile, pixcvec_tile, mask)
        scene_pixc['tvp'] = SceneTVP.from_tile(pixc_tile)

        return scene_pixc

    @classmethod
    def from_tiles(cls, pixc_tiles, swath_edges, swath_polygon_points,
                   granule_start_time, granule_end_time,
                   cycle_number, pass_number, scene_number, pixcvec_tiles=None,
                   mask=None):
        """ Constructs self from a list of pixc tiles (and associated pixcvec
           tiles). Pixcvec_tiles must either have a one-to-one correspondence
           with pixc_tiles or be None. """
        LOGGER.info('constructing scene pixc from tiles')

        num_tiles = len(pixc_tiles)
        if pixcvec_tiles is None:
            pixcvec_tiles = [None]*num_tiles

        tile_objs = []
        for tile_idx in range(num_tiles):
            tile_objs.append(cls.from_tile(pixc_tiles[tile_idx],
                                           pixcvec_tiles[tile_idx],
                                           mask))

        # Add all of the pixel_cloud/tvp data
        scene_pixc = np.array(tile_objs).sum()
        granule_start_times = [datetime.strptime(
            tile.time_granule_start, DATETIME_FORMAT_STR) for tile in tile_objs]
        granule_end_times = [datetime.strptime(
            tile.time_granule_end, DATETIME_FORMAT_STR) for tile in tile_objs]
        coverage_start_times = [datetime.strptime(
            tile.time_coverage_start, DATETIME_FORMAT_STR) for tile in tile_objs
                                if tile.time_coverage_start != EMPTY_DATETIME]
        coverage_end_times = [datetime.strptime(
            tile.time_coverage_end, DATETIME_FORMAT_STR) for tile in tile_objs
                              if tile.time_coverage_end != EMPTY_DATETIME]
        scene_pixc.scene_cycle_number = np.short(cycle_number)
        scene_pixc.scene_pass_number = np.short(pass_number)
        scene_pixc.scene_number = np.short(scene_number)

        # Sort the tile level attributes based on swath side first,
        # then the rest of the name (i.e. side_cycle_pass_tile)
        sort_indices = np.argsort([tile_name[-1].lower() + tile_name[:-1]
                                   for tile_name in tile.tile_names])
        scene_pixc.tile_numbers = [tile_num for i in sort_indices
                                    for tile_num in tile_objs[i].tile_numbers]
        scene_pixc.tile_names = ', '.join(
            [tile_name for i in sort_indices
             for tile_name in tile_objs[i].tile_names])
        scene_pixc.tile_polarizations =', '.join(
            [tile_pol for i in sort_indices
             for tile_pol in tile_objs[i].tile_polarizations])

        if len(coverage_start_times) > 0:
            coverage_start_time = min(coverage_start_times)
        else:
            coverage_start_time = EMPTY_DATETIME

        if len(coverage_end_times) > 0:
            coverage_end_time = max(coverage_end_times)
        else:
            coverage_end_time = EMPTY_DATETIME

        scene_pixc.time_coverage_start = \
            coverage_start_time.strftime(DATETIME_FORMAT_STR)
        scene_pixc.time_coverage_end = \
            coverage_end_time.strftime(DATETIME_FORMAT_STR)

        scene_pixc.set_extent(swath_edges, swath_polygon_points,
                              granule_start_time, granule_end_time)

        # Copy most attributes from one of the central tiles
        # Central tile is one with the median time
        central_tile_index = granule_start_times.index(
            np.percentile(granule_start_times, 50, interpolation='nearest'))
        scene_pixc.wavelength = tile_objs[central_tile_index].wavelength
        scene_pixc.near_range = tile_objs[central_tile_index].near_range
        scene_pixc.nominal_slant_range_spacing = \
            tile_objs[central_tile_index].nominal_slant_range_spacing

        return scene_pixc

    def set_extent(self, swath_edges, swath_polygon_points,
                   granule_start_time, granule_end_time):
        """ Sets the geospatial and temporal extent attributes """
        # Set the first/last lats/lons from the swath edges
        # swath_edges = ((left_first_lat, left_first_lon),
        #                (right_first_lat, right_first_lon),
        #                (left_last_lat, left_last_lon),
        #                (right_last_lat, right_last_lon))
        self.left_first_latitude = swath_edges[0][0]
        self.left_first_longitude = swath_edges[0][1]
        self.right_first_latitude = swath_edges[1][0]
        self.right_first_longitude = swath_edges[1][1]
        self.left_last_latitude = swath_edges[2][0]
        self.left_last_longitude = swath_edges[2][1]
        self.right_last_latitude = swath_edges[3][0]
        self.right_last_longitude = swath_edges[3][1]

        lats = [latlon[0] for latlon in swath_polygon_points]
        lons = [latlon[1] for latlon in swath_polygon_points]
        self.geospatial_lat_min = min(lats)
        self.geospatial_lat_max = max(lats)

        # Handle longitude wrap
        shifted_lons = raster_crs.shift_wrapped_longitude(lons)
        lon_min = min(shifted_lons)
        lon_max = max(shifted_lons)
        # Wrap to between -180 and 180 degrees longitude
        self.geospatial_lon_min = raster_crs.lon_360to180(lon_min)
        self.geospatial_lon_max = raster_crs.lon_360to180(lon_max)

        self.time_granule_start = \
            granule_start_time.strftime(DATETIME_FORMAT_STR)
        self.time_granule_end = \
            granule_end_time.strftime(DATETIME_FORMAT_STR)

    def get_summary_qual_flag(self, qual_flag, suspect_qual_flag_mask,
                              degraded_qual_flag_mask, bad_qual_flag_mask):
        """ Get summary quality flag from quality bitflag """
        LOGGER.info('getting summary quality flag: {}'.format(qual_flag))
        flag = QUAL_IND_GOOD*np.ones(np.shape(self.pixel_cloud['latitude']))
        flag[self.get_qual_mask(qual_flag, suspect_qual_flag_mask, False)] = \
            QUAL_IND_SUSPECT
        flag[self.get_qual_mask(qual_flag, degraded_qual_flag_mask, False)] = \
            QUAL_IND_DEGRADED
        flag[self.get_qual_mask(qual_flag, bad_qual_flag_mask, False)] = \
            QUAL_IND_BAD
        return flag

    def get_qual_mask(self, qual_flag, qual_flag_mask,
                      include_zero_qual_value=True):
        """ Get mask of valid pixc points from quality flag """
        LOGGER.info('getting qual mask: {} - {}'.format(
            qual_flag, qual_flag_mask))

        if qual_flag == 'pixc_line_qual':
            flag = self.pixel_cloud[qual_flag][self.pixel_cloud['azimuth_index']]
        else:
            flag = self.pixel_cloud[qual_flag]

        mask = np.bitwise_and(flag, qual_flag_mask) > 0

        if include_zero_qual_value:
            mask = np.logical_or(mask, flag==0)

        return mask==1

    def get_mask(self, valid_classes, use_improved_geoloc=True):
        """ Get mask of valid pixc points for aggregation """
        LOGGER.info('getting mask')

        if use_improved_geoloc:
            lat_keyword = 'improved_latitude'
            lon_keyword = 'improved_longitude'
        else:
            lat_keyword = 'latitude'
            lon_keyword = 'longitude'

        lats = self.pixel_cloud[lat_keyword]
        lons = self.pixel_cloud[lon_keyword]
        pixc_classif = self.pixel_cloud['classification']

        mask = np.ones(np.shape(lats))

        if np.ma.is_masked(lats):
            mask[lats.mask] = 0
        if np.ma.is_masked(lons):
            mask[lons.mask] = 0

        mask[np.isnan(lats)] = 0
        mask[np.isnan(lons)] = 0
        mask[np.isnan(pixc_classif)] = 0

        classif_mask = np.isin(pixc_classif, valid_classes)
        mask[np.logical_not(classif_mask)] = 0

        return mask==1

    def __add__(self, other):
        """ Add other to self """
        klass = ScenePixc()
        klass.tvp = self.tvp + other.tvp
        klass.pixel_cloud = self.pixel_cloud + other.pixel_cloud

        # Handle merged TVP with overlap discarded
        tvp_time = np.ma.concatenate((self.tvp.time, other.tvp.time))
        tvp_swath_side = np.ma.concatenate(
            (self.tvp.swath_side, other.tvp.swath_side))
        [junk, rev_indx] = np.unique(
            np.column_stack((tvp_time, tvp_swath_side=='R')),
            axis=0, return_inverse=True)
        unsorted_pixc_line_to_tvp = np.ma.concatenate((
            self.pixel_cloud.pixc_line_to_tvp,
            len(self.tvp.time) + other.pixel_cloud.pixc_line_to_tvp)).astype(int)
        klass.pixel_cloud.pixc_line_to_tvp = rev_indx[unsorted_pixc_line_to_tvp]

        # Set attributes from self
        for field in self.ATTRIBUTES.keys():
            attr_val = getattr(self, field)
            setattr(klass, field, attr_val)

        # Merge special attributes
        cycle_numbers = np.concatenate((self.cycle_numbers, other.cycle_numbers))
        pass_numbers = np.concatenate((self.pass_numbers, other.pass_numbers))
        tile_numbers = np.concatenate((self.tile_numbers, other.tile_numbers))
        tile_names = np.concatenate((
            self.tile_names.split(', '), other.tile_names.split(', ')))
        tile_polarizations = np.concatenate((
            self.tile_polarizations.split(', '), other.tile_polarizations.split(', ')))

        # Set the scene level attributes for the scene of the middle tile
        # These should almost always be overwritten later based on the actual
        # scene that we want to rasterize
        tile_name_sort_indices = np.argsort(
            [tile_name for tile_name in tile_names])
        mid_tile_index = tile_name_sort_indices[int(len(tile_name_sort_indices)/2)]
        klass.scene_cycle_number = cycle_numbers[mid_tile_index]
        klass.scene_pass_number = pass_numbers[mid_tile_index]
        klass.scene_number = np.ceil(
            tile_numbers[mid_tile_index]/NONOVERLAP_TILES_PER_SIDE).astype('i2')

        # Sort the tile level attributes based on swath side first,
        # then the rest of the name (i.e. side_cycle_pass_tile)
        sort_indices = np.argsort([tile_name[-1].lower() + tile_name[:-1]
                                   for tile_name in tile_names])
        klass.cycle_numbers = cycle_numbers[sort_indices]
        klass.pass_numbers = pass_numbers[sort_indices]
        klass.tile_numbers = tile_numbers[sort_indices]
        klass.tile_names = ', '.join(tile_names[sort_indices])
        klass.tile_polarizations = ', '.join(tile_polarizations[sort_indices])

        # Overwrite the temporal extent attributes if others are better than self
        # (note that left/right time attributes can be None)
        def _datetime_str_comp(d0, d1, comp=op.le,
                               format_str=DATETIME_FORMAT_STR,
                               empty_value=EMPTY_DATETIME):
            # Compares d0 and d1 with comparison in comp argument
            # Returns False if d0 is None or "None"
            # Returns True if d0 is not None or "None" and d1 is None or "None"
            if d0 is None or d0.lower()=='none' or d0==empty_value:
                return False
            if d1 is None or d1.lower()=='none' or d1==empty_value:
                return True
            _d0 = datetime.strptime(d0, format_str)
            _d1 = datetime.strptime(d1, format_str)
            return comp(_d0, _d1)

        if _datetime_str_comp(other.time_granule_start,
                              self.time_granule_start, comp=op.lt):
            klass_time_granule_start = other.time_granule_start

        if _datetime_str_comp(other.time_granule_end,
                              self.time_granule_end, comp=op.gt):
            klass_time_granule_end = other.time_granule_end

        if _datetime_str_comp(other.time_coverage_start,
                              self.time_coverage_start, comp=op.lt):
            klass_time_coverage_start = other.time_coverage_start

        if _datetime_str_comp(other.time_coverage_end,
                              self.time_coverage_end, comp=op.gt):
            klass_time_coverage_end = other.time_coverage_end

        if _datetime_str_comp(other.left_time_granule_start,
                              self.left_time_granule_start, comp=op.lt):
            klass.left_time_granule_start = other.left_time_granule_start
            klass.left_first_latitude = other.left_first_latitude
            klass.left_first_longitude = other.left_first_longitude

        if _datetime_str_comp(other.left_time_granule_end,
                              self.left_time_granule_end, comp=op.gt):
            klass.left_time_granule_end = other.left_time_granule_end
            klass.left_last_latitude = other.left_last_latitude
            klass.left_last_longitude = other.left_last_longitude

        if _datetime_str_comp(other.right_time_granule_start,
                              self.right_time_granule_start, comp=op.lt):
            klass.right_time_granule_start = other.right_time_granule_start
            klass.right_first_latitude = other.right_first_latitude
            klass.right_first_longitude = other.right_first_longitude

        if _datetime_str_comp(other.right_time_granule_end,
                              self.right_time_granule_end, comp=op.gt):
            klass.right_time_granule_end = other.right_time_granule_end
            klass.right_last_latitude = other.right_last_latitude
            klass.right_last_longitude = other.right_last_longitude

        if _datetime_str_comp(other.left_time_coverage_start,
                              self.left_time_coverage_start, comp=op.lt):
            klass.left_time_coverage_start = other.left_time_coverage_start

        if _datetime_str_comp(other.left_time_coverage_end,
                              self.left_time_coverage_end, comp=op.gt):
            klass.left_time_coverage_end = other.left_time_coverage_end

        if _datetime_str_comp(other.right_time_coverage_start,
                              self.right_time_coverage_start, comp=op.lt):
            klass.right_time_coverage_start = other.right_time_coverage_start

        if _datetime_str_comp(other.right_time_coverage_end,
                              self.right_time_coverage_end, comp=op.gt):
            klass.right_time_coverage_end = other.right_time_coverage_end

        # Get geospatial bounds from self and other's geospatial bounds
        klass.geospatial_lat_min = min(self.geospatial_lat_min,
                                       other.geospatial_lat_min)
        klass.geospatial_lat_max = max(self.geospatial_lat_max,
                                       other.geospatial_lat_max)
        klass.geospatial_lon_min = min(self.geospatial_lon_min,
                                       other.geospatial_lon_min)
        klass.geospatial_lon_max = max(self.geospatial_lon_max,
                                       other.geospatial_lon_max)

        # Get the earlier leap second
        if _datetime_str_comp(other.leap_second, self.leap_second,
                              comp=op.lt, format_str=LEAPSEC_FORMAT_STR,
                              empty_value=EMPTY_LEAPSEC):
            klass.leap_second = other.leap_second
        if klass.leap_second is None or klass.leap_second.lower()=='none':
            klass.leap_second = EMPTY_LEAPSEC

        return klass


class ScenePixelCloud(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype': 'str',
            'value':'cloud of geolocated interferogram pixels'}],
        ['looks_to_efflooks',{'dtype': 'f8',
            'docstr':'ratio of the number of real looks to the effective number of independent looks'}],
    ])
    ATTRIBUTES['description']['docstr'] = ATTRIBUTES['description']['value']
    DIMENSIONS = odict([['points', 0], ['num_pixc_lines', 0]])
    VARIABLES = odict([
        ['latitude', odict([])],
        ['longitude', odict([])],
        ['height', odict([])],
        ['improved_latitude', odict([])],
        ['improved_longitude', odict([])],
        ['improved_height', odict([])],
        ['azimuth_index', odict([])],
        ['range_index', odict([])],
        ['interferogram', odict([])],
        ['classification', odict([])],
        ['eff_num_rare_looks', odict([])],
        ['eff_num_medium_looks', odict([])],
        ['power_plus_y', odict([])],
        ['power_minus_y', odict([])],
        ['dheight_dphase', odict([])],
        ['dlatitude_dphase', odict([])],
        ['dlongitude_dphase', odict([])],
        ['phase_noise_std', odict([])],
        ['pixel_area', odict([])],
        ['water_frac', odict([])],
        ['water_frac_uncert', odict([])],
        ['darea_dheight', odict([])],
        ['false_detection_rate', odict([])],
        ['missed_detection_rate', odict([])],
        ['cross_track', odict([])],
        ['sig0', odict([])],
        ['sig0_uncert', odict([])],
        ['inc', odict([])],
        ['illumination_time', odict([])],
        ['illumination_time_tai', odict([])],
        ['ice_clim_flag', odict([])],
        ['ice_dyn_flag', odict([])],
        ['bright_land_flag', odict([])],
        ['layover_impact', odict([])],
        ['sig0_cor_atmos_model', odict([])],
        ['height_cor_xover', odict([])],
        ['geoid', odict([])],
        ['solid_earth_tide', odict([])],
        ['load_tide_fes', odict([])],
        ['load_tide_got', odict([])],
        ['pole_tide', odict([])],
        ['model_dry_tropo_cor', odict([])],
        ['model_wet_tropo_cor', odict([])],
        ['iono_cor_gim_ka', odict([])],
        ['interferogram_qual', odict([])],
        ['classification_qual', odict([])],
        ['geolocation_qual', odict([])],
        ['sig0_qual', odict([])],
        ['pixc_line_qual', odict([])],
        ['pixc_line_to_tvp', odict([])]
    ])

    for name, reference in VARIABLES.items():
        reference['dimensions'] = odict([['points', 0]])
    VARIABLES['pixc_line_qual']['dimensions'] = odict([['num_pixc_lines',0],])
    VARIABLES['pixc_line_to_tvp']['dimensions'] = odict([['num_pixc_lines',0],])

    @classmethod
    def from_tile(cls, pixc_tile, pixcvec_tile=None, mask=None):
        """ Construct self from a single pixc tile (and associated pixcvec tile) """
        LOGGER.info('constructing scene pixel cloud from tile')

        scene_pixel_cloud = cls()

        if mask is None: # Default mask is all
            mask = np.ones(pixc_tile['pixel_cloud']['illumination_time'].shape,
                           dtype=bool)

        # Copy common pixc variables (and attributes)
        pixel_cloud_vars = set(scene_pixel_cloud.VARIABLES.keys())
        for field in pixel_cloud_vars.intersection(
                pixc_tile['pixel_cloud'].VARIABLES.keys()):
            scene_pixel_cloud.VARIABLES[field] = \
                pixc_tile['pixel_cloud'].VARIABLES[field]
            if field in ['pixc_line_qual', 'pixc_line_to_tvp']:
                scene_pixel_cloud[field] = pixc_tile['pixel_cloud'][field]
            else:
                scene_pixel_cloud[field] = pixc_tile['pixel_cloud'][field][mask]

        # Copy pixcvec variables
        # set improved llh to pixcvec llh where it exists, otherwise use pixc llh
        scene_pixel_cloud['improved_latitude'] = scene_pixel_cloud['latitude']
        scene_pixel_cloud['improved_longitude'] = scene_pixel_cloud['longitude']
        scene_pixel_cloud['improved_height'] = scene_pixel_cloud['height']

        if pixcvec_tile is not None:
            pixcvec_geoloc_valid = np.logical_not(np.logical_or.reduce((
                np.ma.getmaskarray(pixcvec_tile['latitude_vectorproc'][mask]),
                np.ma.getmaskarray(pixcvec_tile['longitude_vectorproc'][mask]),
                np.ma.getmaskarray(pixcvec_tile['height_vectorproc'][mask]))))

            scene_pixel_cloud['improved_latitude'][pixcvec_geoloc_valid] = \
                pixcvec_tile['latitude_vectorproc'][mask][pixcvec_geoloc_valid]
            scene_pixel_cloud['improved_longitude'][pixcvec_geoloc_valid] = \
                pixcvec_tile['longitude_vectorproc'][mask][pixcvec_geoloc_valid]
            scene_pixel_cloud['improved_height'][pixcvec_geoloc_valid] = \
                pixcvec_tile['height_vectorproc'][mask][pixcvec_geoloc_valid]

            scene_pixel_cloud['ice_clim_flag'] = \
                np.ma.MaskedArray(pixcvec_tile['ice_clim_f'])[mask]
            scene_pixel_cloud['ice_dyn_flag'] = \
                np.ma.MaskedArray(pixcvec_tile['ice_dyn_f'])[mask]
            ice_clim_flag_mask = np.logical_not(np.isin(
                scene_pixel_cloud['ice_clim_flag'],
                COMMON_VARIABLES['ice_clim_flag']['flag_values']))
            ice_dyn_flag_mask = np.logical_not(np.isin(
                scene_pixel_cloud['ice_dyn_flag'],
                COMMON_VARIABLES['ice_dyn_flag']['flag_values']))
            scene_pixel_cloud['ice_clim_flag'].mask = ice_clim_flag_mask
            scene_pixel_cloud['ice_dyn_flag'].mask = ice_dyn_flag_mask

        # Copy common pixc attributes
        pixel_cloud_attr = set(scene_pixel_cloud.ATTRIBUTES.keys())
        for field in pixel_cloud_attr.intersection(
                pixc_tile['pixel_cloud'].ATTRIBUTES):
            attr_val = getattr(pixc_tile['pixel_cloud'], field)
            setattr(scene_pixel_cloud, field, attr_val)

        return scene_pixel_cloud

    def __add__(self, other):
        """ Add other to self """
        klass = ScenePixelCloud()
        for key in klass.VARIABLES:
            if key in ['azimuth_index']:
                setattr(klass, key, np.ma.concatenate((
                    getattr(self, key), len(self.pixc_line_qual) + getattr(other, key))))
            else:
                setattr(klass, key, np.ma.concatenate((
                    getattr(self, key), getattr(other, key))))

        for field in self.ATTRIBUTES.keys():
            attr_val = getattr(self, field)
            setattr(klass, field, attr_val)

        return klass


class SceneTVP(Product):
    ATTRIBUTES = odict([
        ['description', {'dtype': 'str',
            'value': textjoin("""
                 Time varying parameters group
                 including spacecraft attitude, position, velocity,
                 and antenna position information""")}],
        ])
    ATTRIBUTES['description']['docstr'] = ATTRIBUTES['description']['value']
    DIMENSIONS = odict([['num_tvps', 0]])
    VARIABLES = odict([
        ['time', odict([])],
        ['velocity_heading', odict([])],
        ['x', odict([])],
        ['y', odict([])],
        ['z', odict([])],
        ['vx', odict([])],
        ['vy', odict([])],
        ['vz', odict([])],
        ['plus_y_antenna_x', odict([])],
        ['plus_y_antenna_y', odict([])],
        ['plus_y_antenna_z', odict([])],
        ['minus_y_antenna_x', odict([])],
        ['minus_y_antenna_y', odict([])],
        ['minus_y_antenna_z', odict([])],
        ['swath_side', odict([])],
    ])
    for name, reference in VARIABLES.items():
        reference['dimensions'] = DIMENSIONS

    @classmethod
    def from_tile(cls, pixc_tile):
        """ Construct self from a single pixc tile """
        LOGGER.info('constructing scene tvp from tile')

        scene_tvp = cls()

        # Copy common variables
        tvp_vars = set(scene_tvp.VARIABLES.keys())
        for field in tvp_vars.intersection(
                pixc_tile['tvp'].VARIABLES.keys()):
            scene_tvp[field] = pixc_tile['tvp'][field]

        # Copy common attributes
        tvp_attr = set(scene_tvp.ATTRIBUTES.keys())
        for field in tvp_attr.intersection(
                pixc_tile['tvp'].ATTRIBUTES):
            attr_val = getattr(pixc_tile['tvp'], field)
            setattr(scene_tvp, field, attr_val)

        # Get swath side
        scene_tvp['swath_side'] = np.full((scene_tvp.dimensions['num_tvps']),
                                          pixc_tile.swath_side)

        return scene_tvp

    def __add__(self, other):
        """ Add other to self """
        klass = SceneTVP()
        # Discard TVP overlap for each side separately
        time = np.ma.concatenate((self.time, other.time))
        swath_side = np.ma.concatenate((self.swath_side, other.swath_side))
        [junk, indx] = np.unique(
            np.column_stack((time, swath_side=='R')), axis=0, return_index=True)
        for key in klass.VARIABLES:
            setattr(klass, key, np.ma.concatenate((
                getattr(self, key), getattr(other, key)))[indx])

        for field in self.ATTRIBUTES.keys():
            attr_val = getattr(self, field)
            setattr(klass, field, attr_val)

        return klass
