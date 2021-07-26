'''
Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''
import logging
import textwrap
import raster_crs
import numpy as np

from osgeo import osr
from netCDF4 import Dataset
from datetime import datetime
from shapely.geometry import Point, Polygon
from collections import OrderedDict as odict
from SWOTWater.products.product import Product

UNIX_EPOCH = datetime(1970, 1, 1)
SWOT_EPOCH = datetime(2000, 1, 1)
TIME_FORMAT_STR = '%Y-%m-%d %H:%M:%S.%fZ'

PIXC_BAD_FLAG_VALUE = 2

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
    ['platform', {'dtype': 'str' ,'value':'SWOT','docstr': 'SWOT'}],
    ['references',
     {'dtype': 'str', 'value': 'https://github.com/SWOTAlgorithms/Raster-Processor',
      'docstr': textjoin("""
          Published or web-based references that describe
          the data or methods used to product it. Provides version number of
          software generating product.""")}],
    ['reference_document',
     {'dtype': 'str', 'value':'JPL D-56416 - Revision A (DRAFT) - January 27, 2021',
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
     {'dtype': 'str',
      'value': 'L2_HR_Raster',
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
     {'dtype': 'str',
      'docstr': 'Version identifier of this data file',
      'value': 'V0.2'}],
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
            ['valid_min', -1500],
            ['valid_max', 15000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Water surface elevation of the pixel above the geoid and after
                using models to subtract the effects of tides
                (solid_earth_tide, load_tide_fes, pole_tide).""")],
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
            ['long_name', 'surface area of water'],
            ['grid_mapping', 'crs'],
            ['units', 'm^2'],
            ['valid_min', -2000000],
            ['valid_max', 2000000000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Surface area of the water pixels.""")],
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
            ['valid_min', -1000],
            ['valid_max', 10000000],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Normalized radar cross section (sigma0) in real, linear units
                (not decibels). The value may be negative due to noise
                subtraction.""")],
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
    ['wse_qual',
     odict([['dtype', 'u1'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', 'good bad'],
            ['flag_values', np.array([0, 1]).astype('i1')],
            ['valid_min', 0],
            ['valid_max', 1],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Quality flag for the WSE quantities in the raster data.""")],
        ])],
    ['water_area_qual',
     odict([['dtype', 'u1'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', 'good bad'],
            ['flag_values', np.array([0, 1]).astype('i1')],
            ['valid_min', 0],
            ['valid_max', 1],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Quality flag for the water area quantities in the raster
                data.""")],
        ])],
    ['sig0_qual',
     odict([['dtype', 'u1'],
            ['standard_name', 'status_flag'],
            ['grid_mapping', 'crs'],
            ['flag_meanings', 'good bad'],
            ['flag_values', np.array([0, 1]).astype('i1')],
            ['valid_min', 0],
            ['valid_max', 1],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Quality flag for the sigma0 quantities in the raster data.""")],
        ])],
    ['n_wse_pix',
     odict([['dtype', 'u4'],
            ['long_name', 'number of wse pixels'],
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
            ['long_name', 'number of water area pixels'],
            ['grid_mapping', 'crs'],
            ['units', 'l'],
            ['valid_min', 0],
            ['valid_max', 999999],
            ['coordinates', '[Raster coordinates]'],
            ['comment', textjoin("""
                Number of pixel cloud samples used in water area and
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
                quantities not related to WSE, water area or sigma0.""")],
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
                Fraction of pixel water area covered by dark water.""")],
        ])],
    ['ice_clim_flag',
     odict([['dtype', 'u1'],
            ['long_name', 'climatological ice cover flag'],
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
                ['semi_major_axis', 6378137.],
                ['inverse_flattening', 298.257223563],
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
        ['wse_uncert', COMMON_VARIABLES['wse_uncert'].copy()],
        ['water_area', COMMON_VARIABLES['water_area'].copy()],
        ['water_area_uncert', COMMON_VARIABLES['water_area_uncert'].copy()],
        ['water_frac', COMMON_VARIABLES['water_frac'].copy()],
        ['water_frac_uncert', COMMON_VARIABLES['water_frac_uncert'].copy()],
        ['sig0', COMMON_VARIABLES['sig0'].copy()],
        ['sig0_uncert', COMMON_VARIABLES['sig0_uncert'].copy()],
        ['inc', COMMON_VARIABLES['inc'].copy()],
        ['cross_track', COMMON_VARIABLES['cross_track'].copy()],
        ['illumination_time', COMMON_VARIABLES['illumination_time'].copy()],
        ['illumination_time_tai', COMMON_VARIABLES['illumination_time_tai'].copy()],
        ['wse_qual', COMMON_VARIABLES['wse_qual'].copy()],
        ['water_area_qual', COMMON_VARIABLES['water_area_qual'].copy()],
        ['sig0_qual', COMMON_VARIABLES['sig0_qual'].copy()],
        ['n_wse_pix', COMMON_VARIABLES['n_wse_pix'].copy()],
        ['n_water_area_pix', COMMON_VARIABLES['n_water_area_pix'].copy()],
        ['n_sig0_pix', COMMON_VARIABLES['n_sig0_pix'].copy()],
        ['n_other_pix', COMMON_VARIABLES['n_other_pix'].copy()],
        ['dark_frac', COMMON_VARIABLES['dark_frac'].copy()],
        ['ice_clim_flag', COMMON_VARIABLES['ice_clim_flag'].copy()],
        ['ice_dyn_flag', COMMON_VARIABLES['ice_dyn_flag'].copy()],
        ['layover_impact', COMMON_VARIABLES['layover_impact'].copy()],
        ['geoid', COMMON_VARIABLES['geoid'].copy()],
        ['solid_earth_tide', COMMON_VARIABLES['solid_earth_tide'].copy()],
        ['load_tide_fes', COMMON_VARIABLES['load_tide_fes'].copy()],
        ['load_tide_got', COMMON_VARIABLES['load_tide_got'].copy()],
        ['pole_tide', COMMON_VARIABLES['pole_tide'].copy()],
        ['model_dry_tropo_cor', COMMON_VARIABLES['model_dry_tropo_cor'].copy()],
        ['model_wet_tropo_cor', COMMON_VARIABLES['model_wet_tropo_cor'].copy()],
        ['iono_cor_gim_ka', COMMON_VARIABLES['iono_cor_gim_ka'].copy()],
    ])

    for key in COMMON_VARIABLES:
        VARIABLES[key]['coordinates'] = 'x y'
        VARIABLES[key]['dimensions'] = odict([['y', 0], ['x', 0]])

    VARIABLES['latitude']['coordinates'] = 'x y'
    VARIABLES['latitude']['dimensions'] = odict([['y', 0], ['x', 0]])
    VARIABLES['longitude']['coordinates'] = 'x y'
    VARIABLES['longitude']['dimensions'] = odict([['y', 0], ['x', 0]])

    VARIABLES['x']['dimensions'] = odict([['x', 0]])
    VARIABLES['y']['dimensions'] = odict([['y', 0]])
    VARIABLES['crs']['dimensions'] = odict([])

    def get_raster_mapping(self, pixc, mask, use_improved_geoloc=True):
        """ Get the mapping of pixc points to raster bins """
        LOGGER.info('RasterUTM::get_raster_mapping')

        if use_improved_geoloc:
            lat_keyword = 'improved_latitude'
            lon_keyword = 'improved_longitude'
        else:
            lat_keyword = 'latitude'
            lon_keyword = 'longitude'

        pixc_lats = pixc['pixel_cloud'][lat_keyword]
        pixc_lons = raster_crs.lon_360to180(pixc['pixel_cloud'][lon_keyword])

        input_crs = raster_crs.wgs84_crs()
        output_crs = raster_crs.utm_crs(self.utm_zone_num,
                                        self.mgrs_latitude_band)
        transf = osr.CoordinateTransformation(input_crs, output_crs)

        x_tmp=[]
        y_tmp=[]
        for x in range(0, len(pixc_lats)):
            if mask[x]:
                u_x, u_y = transf.TransformPoint(pixc_lats[x], pixc_lons[x])[:2]
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
                i = int(round((y_tmp[x] - self.y_min) / self.resolution))
                j = int(round((x_tmp[x] - self.x_min) / self.resolution))
                # check bounds
                if (i >= 0 and i < self.dimensions['y'] and
                    j >= 0 and j < self.dimensions['x']):
                    mapping_tmp[i][j].append(x)

        return mapping_tmp

    def crop_to_bounds(self, swath_polygon_points):
        """ Crop raster to the given swath polygon """
        LOGGER.info('RasterUTM::crop_to_bounds')

        # Convert polygon points to UTM
        input_crs = raster_crs.wgs84_crs()
        output_crs = raster_crs.utm_crs(self.utm_zone_num,
                                        self.mgrs_latitude_band)
        transf = osr.CoordinateTransformation(input_crs, output_crs)
        swath_polygon_points_utm = []
        for pt in swath_polygon_points:
            swath_polygon_points_utm.append(transf.TransformPoint(pt[0],
                                                                  pt[1])[:2])

        poly = Polygon(swath_polygon_points_utm)

        # Check whether each pixel center is within the polygon
        mask = np.zeros((self.dimensions['y'], self.dimensions['x']))
        for i in range(0, self.dimensions['y']):
            for j in range(0, self.dimensions['x']):
                if Point((self.x[j], self.y[i])).within(poly):
                    mask[i][j] = True

        # Mask the datasets
        for var in self.variables:
            if var not in ['crs', 'x', 'y']: # only the 2d datasets
                self.variables[var].mask = np.logical_or(
                    self.variables[var].mask, np.logical_not(mask))

    def get_uncorrected_height(self):
        """ Get the height with wse geophysical corrections removed """
        LOGGER.info('RasterUTM::get_uncorrected_height')

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
                ['semi_major_axis', 6378137.],
                ['inverse_flattening', 298.257223563],
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
        ['wse_uncert', COMMON_VARIABLES['wse_uncert'].copy()],
        ['water_area', COMMON_VARIABLES['water_area'].copy()],
        ['water_area_uncert', COMMON_VARIABLES['water_area_uncert'].copy()],
        ['water_frac', COMMON_VARIABLES['water_frac'].copy()],
        ['water_frac_uncert', COMMON_VARIABLES['water_frac_uncert'].copy()],
        ['sig0', COMMON_VARIABLES['sig0'].copy()],
        ['sig0_uncert', COMMON_VARIABLES['sig0_uncert'].copy()],
        ['inc', COMMON_VARIABLES['inc'].copy()],
        ['cross_track', COMMON_VARIABLES['cross_track'].copy()],
        ['illumination_time', COMMON_VARIABLES['illumination_time'].copy()],
        ['illumination_time_tai', COMMON_VARIABLES['illumination_time_tai'].copy()],
        ['wse_qual', COMMON_VARIABLES['wse_qual'].copy()],
        ['water_area_qual', COMMON_VARIABLES['water_area_qual'].copy()],
        ['sig0_qual', COMMON_VARIABLES['sig0_qual'].copy()],
        ['n_wse_pix', COMMON_VARIABLES['n_wse_pix'].copy()],
        ['n_water_area_pix', COMMON_VARIABLES['n_water_area_pix'].copy()],
        ['n_sig0_pix', COMMON_VARIABLES['n_sig0_pix'].copy()],
        ['n_other_pix', COMMON_VARIABLES['n_other_pix'].copy()],
        ['dark_frac', COMMON_VARIABLES['dark_frac'].copy()],
        ['ice_clim_flag', COMMON_VARIABLES['ice_clim_flag'].copy()],
        ['ice_dyn_flag', COMMON_VARIABLES['ice_dyn_flag'].copy()],
        ['layover_impact', COMMON_VARIABLES['layover_impact'].copy()],
        ['geoid', COMMON_VARIABLES['geoid'].copy()],
        ['solid_earth_tide', COMMON_VARIABLES['solid_earth_tide'].copy()],
        ['load_tide_fes', COMMON_VARIABLES['load_tide_fes'].copy()],
        ['load_tide_got', COMMON_VARIABLES['load_tide_got'].copy()],
        ['pole_tide', COMMON_VARIABLES['pole_tide'].copy()],
        ['model_dry_tropo_cor', COMMON_VARIABLES['model_dry_tropo_cor'].copy()],
        ['model_wet_tropo_cor', COMMON_VARIABLES['model_wet_tropo_cor'].copy()],
        ['iono_cor_gim_ka', COMMON_VARIABLES['iono_cor_gim_ka'].copy()],
    ])

    for key in COMMON_VARIABLES:
        VARIABLES[key]['coordinates'] = 'longitude latitude'
        VARIABLES[key]['dimensions'] = odict([['latitude', 0], ['longitude', 0]])

    VARIABLES['longitude']['dimensions'] = odict([['longitude', 0]])
    VARIABLES['latitude']['dimensions'] = odict([['latitude', 0]])
    VARIABLES['crs']['dimensions'] = odict([])

    def get_raster_mapping(self, pixc, mask, use_improved_geoloc=True):
        """ Get the mapping of pixc points to raster bins """
        LOGGER.info('RasterGeo::get_raster_mapping')

        if use_improved_geoloc:
            lat_keyword = 'improved_latitude'
            lon_keyword = 'improved_longitude'
        else:
            lat_keyword = 'latitude'
            lon_keyword = 'longitude'

        pixc_lats = pixc['pixel_cloud'][lat_keyword]
        pixc_lons = raster_crs.lon_360to180(pixc['pixel_cloud'][lon_keyword])

        mapping_tmp = []
        for i in range(0, self.dimensions['latitude']):
            mapping_tmp.append([])
            for j in range(0, self.dimensions['longitude']):
                mapping_tmp[i].append([])

        for x in range(0, len(pixc_lats)):
            if mask[x]:
                i = int(round((pixc_lats[x] - self.latitude_min)
                              / self.resolution).astype(int))
                j = int(round((pixc_lons[x] - self.longitude_min)
                              / self.resolution).astype(int))
                # check bounds
                if (i >= 0 and i < self.dimensions['latitude'] and
                    j >= 0 and j < self.dimensions['longitude']):
                    mapping_tmp[i][j].append(x)

        return mapping_tmp

    def crop_to_bounds(self, swath_polygon_points):
        """ Crop raster to the given swath polygon """
        LOGGER.info('RasterGeo::crop_to_bounds')

        poly = Polygon(swath_polygon_points)

        # Check whether each pixel center is within the polygon
        mask = np.zeros((self.dimensions['latitude'],
                         self.dimensions['longitude']))
        for i in range(0, self.dimensions['latitude']):
            for j in range(0, self.dimensions['longitude']):
                if Point((self.latitude[i], self.longitude[j])).within(poly):
                    mask[i][j] = True

        # Mask the datasets
        for var in self.variables:
            if var not in ['crs', 'longitude', 'latitude']: # only the 2d datasets
                self.variables[var].mask = np.logical_or(
                    self.variables[var].mask, np.logical_not(mask))

        # Set the time coverage start and end
        if np.all(self.illumination_time.mask):
            start_illumination_time = np.min(self.illumination_time)
            end_illumination_time = np.max(self.illumination_time)
        else:
            start_illumination_time = 0
            end_illumination_time = 0
        start_time = datetime.fromtimestamp(
            (SWOT_EPOCH-UNIX_EPOCH).total_seconds() \
            + start_illumination_time)
        end_time = datetime.fromtimestamp(
            (SWOT_EPOCH-UNIX_EPOCH).total_seconds() \
            + end_illumination_time))

        self.time_coverage_start = start_time.strftime(TIME_FORMAT_STR)
        self.time_coverage_end = stop_time.strftime(TIME_FORMAT_STR)

    def get_uncorrected_height(self):
        """ Get the height with wse geophysical corrections removed """
        LOGGER.info('RasterGeo::get_uncorrected_height')

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
         odict([['dtype', 'i1']])],
    ]))
    for key in COMMON_VARIABLES:
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
         odict([['dtype', 'i1']])],
    ]))
    for key in COMMON_VARIABLES:
        VARIABLES[key]['coordinates'] = 'longitude latitude'
        VARIABLES[key]['dimensions'] = odict([['latitude', 0], ['longitude', 0]])

    VARIABLES['longitude']['dimensions'] = odict([['longitude', 0]])
    VARIABLES['latitude']['dimensions'] = odict([['latitude', 0]])
    VARIABLES['crs']['dimensions'] = odict([])
    VARIABLES['classification']['dimensions'] = \
        odict([['latitude', 0], ['longitude', 0]])


class RasterPixc(Product):
    ATTRIBUTES = odict([
        ['cycle_number', odict([])],
        ['pass_number', odict([])],
        ['scene_number', odict([])],
        ['tile_numbers', odict([])],
        ['tile_names', odict([])],
        ['tile_polarizations', odict([])],
        ['time_granule_start', odict([])],
        ['time_granule_end', odict([])],
        ['time_coverage_start', odict([])],
        ['time_coverage_end', odict([])],
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
    ])
    GROUPS = odict([
        ['pixel_cloud', 'RasterPixelCloud'],
        ['tvp', 'RasterTVP'],
    ])

    @classmethod
    def from_tile(cls, pixc_tile, pixcvec_tile=None):
        """ Construct self from a single pixc tile (and associated pixcvec tile) """
        LOGGER.info('RasterPixc::from_tile')

        raster_pixc = cls()

        # Copy over attributes
        raster_pixc.cycle_number = pixc_tile.cycle_number
        raster_pixc.pass_number = pixc_tile.pass_number
        raster_pixc.tile_numbers = [pixc_tile.tile_number]
        raster_pixc.tile_names = [pixc_tile.tile_name]
        raster_pixc.tile_polarizations = [pixc_tile.polarization]
        raster_pixc.scene_number = np.ceil(pixc_tile.tile_number/2).astype('i2')
        raster_pixc.time_granule_start = pixc_tile.time_granule_start
        raster_pixc.time_granule_end = pixc_tile.time_granule_end
        raster_pixc.time_coverage_start = pixc_tile.time_coverage_start
        raster_pixc.time_coverage_end = pixc_tile.time_coverage_end
        raster_pixc.wavelength = pixc_tile.wavelength
        raster_pixc.near_range = pixc_tile.near_range
        raster_pixc.nominal_slant_range_spacing = \
            pixc_tile.nominal_slant_range_spacing

        swath_side = pixc_tile.swath_side

        if swath_side.lower() == 'l':
            raster_pixc.left_first_longitude = pixc_tile.outer_first_longitude
            raster_pixc.left_last_longitude = pixc_tile.outer_last_longitude
            raster_pixc.left_first_latitude = pixc_tile.outer_first_latitude
            raster_pixc.left_last_latitude = pixc_tile.outer_last_latitude
            raster_pixc.right_first_longitude = pixc_tile.inner_first_longitude
            raster_pixc.right_last_longitude = pixc_tile.inner_last_longitude
            raster_pixc.right_first_latitude = pixc_tile.inner_first_latitude
            raster_pixc.right_last_latitude = pixc_tile.inner_last_latitude
        elif swath_side.lower() == 'r':
            raster_pixc.left_first_longitude = pixc_tile.inner_first_longitude
            raster_pixc.left_last_longitude = pixc_tile.inner_last_longitude
            raster_pixc.left_first_latitude = pixc_tile.inner_first_latitude
            raster_pixc.left_last_latitude = pixc_tile.inner_last_latitude
            raster_pixc.right_first_longitude = pixc_tile.outer_first_longitude
            raster_pixc.right_last_longitude = pixc_tile.outer_last_longitude
            raster_pixc.right_first_latitude = pixc_tile.outer_first_latitude
            raster_pixc.right_last_latitude = pixc_tile.outer_last_latitude

        lats = [raster_pixc.left_first_latitude,
                 raster_pixc.right_first_latitude,
                 raster_pixc.left_last_latitude,
                 raster_pixc.right_last_latitude]
        lons = [raster_pixc.left_first_longitude,
                   raster_pixc.right_first_longitude,
                   raster_pixc.left_last_longitude,
                   raster_pixc.right_last_longitude]
        raster_pixc.geospatial_lat_min = min(lats)
        raster_pixc.geospatial_lat_max = max(lats)
        raster_pixc.geospatial_lon_min = min(lons)
        raster_pixc.geospatial_lon_max = max(lons)

        # Copy over groups
        raster_pixc['pixel_cloud'] = RasterPixelCloud.from_tile(
            pixc_tile['pixel_cloud'], pixcvec_tile)
        raster_pixc['tvp'] = RasterTVP.from_tile(pixc_tile['tvp'])

        return raster_pixc

    @classmethod
    def from_tiles(cls, pixc_tiles, swath_edges, swath_polygon_points,
                   granule_start_time, granule_end_time, cycle_number,
                   pass_number, scene_number, pixcvec_tiles=None):
        """ Constructs self from a list of pixc tiles (and associated pixcvec
           tiles). Pixcvec_tiles must either have a one-to-one correspondence
           with pixc_tiles or be None. """
        LOGGER.info('RasterPixc::from_tiles')

        num_tiles = len(pixc_tiles)
        if pixcvec_tiles is None:
            pixcvec_tiles = [None]*num_tiles

        tile_objs = []
        for tile_idx in range(num_tiles):
            tile_objs.append(cls.from_tile(pixc_tiles[tile_idx],
                                           pixcvec_tiles[tile_idx]))

        # Add all of the pixel_cloud/tvp data
        raster_pixc = np.array(tile_objs).sum()
        granule_start_times = [datetime.strptime(
            tile.time_granule_start, TIME_FORMAT_STR) for tile in tile_objs]
        granule_end_times = [datetime.strptime(
            tile.time_granule_end, TIME_FORMAT_STR) for tile in tile_objs]
        coverage_start_times = [datetime.strptime(
            tile.time_coverage_start, TIME_FORMAT_STR) for tile in tile_objs]
        coverage_end_times = [datetime.strptime(
            tile.time_coverage_end, TIME_FORMAT_STR) for tile in tile_objs]
        raster_pixc.cycle_number = np.short(cycle_number)
        raster_pixc.pass_number = np.short(pass_number)
        raster_pixc.scene_number = np.short(scene_number)

        swath_sides = [tile_name[-1] for tile in tile_objs
                       for tile_name in tile.tile_names]
        sort_indices = np.argsort(
            [swath_side + str(start_time)
             for swath_side, start_time in zip(swath_sides, granule_start_times)])

        raster_pixc.tile_numbers = [tile_num for i in sort_indices
                                    for tile_num in tile_objs[i].tile_numbers]
        raster_pixc.tile_names = ', '.join(
            [tile_name for i in sort_indices
             for tile_name in tile_objs[i].tile_names])
        raster_pixc.tile_polarizations =', '.join(
            [tile_pol for i in sort_indices
             for tile_pol in tile_objs[i].tile_polarizations])
        raster_pixc.time_granule_start = \
            granule_start_time.strftime(TIME_FORMAT_STR)
        raster_pixc.time_granule_end = \
            granule_end_time.strftime(TIME_FORMAT_STR)
        raster_pixc.time_coverage_start = \
            min(coverage_start_times).strftime(TIME_FORMAT_STR)
        raster_pixc.time_coverage_end = \
            max(coverage_end_times).strftime(TIME_FORMAT_STR)

        # Copy most attributes from one of the central tiles
        # Central tile is one with the median time
        central_tile_index = granule_start_times.index(
            np.percentile(granule_start_times, 50, interpolation='nearest'))
        raster_pixc.wavelength = tile_objs[central_tile_index].wavelength
        raster_pixc.near_range = tile_objs[central_tile_index].near_range
        raster_pixc.nominal_slant_range_spacing = \
            tile_objs[central_tile_index].nominal_slant_range_spacing

        # Set the first/last lats/lons from the swath edges
        # swath_edges = ((left_first_lat, left_first_lon),
        #                (right_first_lat, right_first_lon),
        #                (left_last_lat, left_last_lon),
        #                (right_last_lat, right_last_lon))
        raster_pixc.left_first_latitude = swath_edges[0][0]
        raster_pixc.left_first_longitude = swath_edges[0][1]
        raster_pixc.right_first_latitude = swath_edges[1][0]
        raster_pixc.right_first_longitude = swath_edges[1][1]
        raster_pixc.left_last_latitude = swath_edges[2][0]
        raster_pixc.left_last_longitude = swath_edges[2][1]
        raster_pixc.right_last_latitude = swath_edges[3][0]
        raster_pixc.right_last_longitude = swath_edges[3][1]

        lats = [latlon[0] for latlon in swath_polygon_points]
        lons = [latlon[1] for latlon in swath_polygon_points]
        raster_pixc.geospatial_lat_min = min(lats)
        raster_pixc.geospatial_lat_max = max(lats)
        raster_pixc.geospatial_lon_min = min(lons)
        raster_pixc.geospatial_lon_max = max(lons)
        return raster_pixc

    def get_mask(self, valid_classes, qual_flags=[],
                 use_improved_geoloc=True):
        """ Get mask of valid pixc points for raster aggregation """
        LOGGER.info('RasterPixc::get_valid_masks')

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

        # Use valid classes to mask
        classif_mask = np.zeros_like(mask)
        for classif_val in valid_classes:
            classif_mask = np.logical_or(
                classif_mask, pixc_classif==classif_val)
        mask[classif_mask] = 0

        # Use qual flags to mask
        qual_mask = np.zeros_like(mask)
        for qual_flag in qual_flags:
            if qual_flag == 'pixc_line_qual':
                per_pixel_line_qual = \
                    self.pixel_cloud['pixc_line_qual'][self.pixel_cloud['azimuth_index']]
                qual_mask = np.logical_or(
                    qual_mask, per_pixel_line_qual==PIXC_BAD_FLAG_VALUE)
            else:
                qual_mask = np.logical_or(
                    qual_mask, self.pixel_cloud[qual_flag]==PIXC_BAD_FLAG_VALUE)
        mask[pixc_qual] == 0

        return mask==1

    def __add__(self, other):
        """ Add other to self """

        klass = RasterPixc()
        klass.tvp = self.tvp + other.tvp
        klass.pixel_cloud = self.pixel_cloud + other.pixel_cloud
        return klass


class RasterPixelCloud(Product):
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
        ['layover_impact', odict([])],
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
        ['height_qual', odict([])],
        ['pixc_line_qual', odict([])],
    ])
    for name, reference in VARIABLES.items():
        reference['dimensions'] = odict([['points', 0]])
    VARIABLES['pixc_line_qual']['dimensions'] = odict([['num_pixc_lines',0],])

    @classmethod
    def from_tile(cls, pixc_tile, pixcvec_tile=None):
        """ Construct self from a single pixc tile (and associated pixcvec tile) """
        LOGGER.info('RasterPixelCloud::from_tile')

        raster_pixel_cloud = cls()

        # Copy common pixc variables
        pixel_cloud_vars = set(raster_pixel_cloud.VARIABLES.keys())
        for field in pixel_cloud_vars.intersection(
                pixc_tile.VARIABLES.keys()):
            raster_pixel_cloud[field] = pixc_tile[field]

        # Copy pixcvec variables (set improved llh to pixcvec llh here)
        if pixcvec_tile is not None:
            raster_pixel_cloud['improved_latitude'] = \
                pixcvec_tile.latitude_vectorproc
            raster_pixel_cloud['improved_longitude'] = \
                pixcvec_tile.longitude_vectorproc
            raster_pixel_cloud['improved_height'] = \
                pixcvec_tile.height_vectorproc

            raster_pixel_cloud['ice_clim_flag'] = pixcvec_tile.ice_clim_f
            raster_pixel_cloud['ice_dyn_flag'] = pixcvec_tile.ice_dyn_f

        # Copy common pixc attributes
        pixel_cloud_attr = set(raster_pixel_cloud.ATTRIBUTES.keys())
        for field in pixel_cloud_attr.intersection(
                pixc_tile.ATTRIBUTES):
            attr_val = getattr(pixc_tile, field)
            setattr(raster_pixel_cloud, field, attr_val)
        return raster_pixel_cloud

    def __add__(self, other):
        """ Add other to self """
        klass = RasterPixelCloud()
        klass.looks_to_efflooks = self.looks_to_efflooks
        for key in klass.VARIABLES:
            setattr(klass, key, np.concatenate((
                getattr(self, key), getattr(other, key))))
        return klass

class RasterTVP(Product):
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
    ])
    for name, reference in VARIABLES.items():
        reference['dimensions'] = DIMENSIONS

    @classmethod
    def from_tile(cls, pixc_tile):
        """ Construct self from a single pixc tile """
        LOGGER.info('RasterTVP::from_tile')

        raster_tvp = cls()

        # Copy common variables
        tvp_vars = set(raster_tvp.VARIABLES.keys())
        for field in tvp_vars.intersection(
                pixc_tile.VARIABLES.keys()):
            raster_tvp[field] = pixc_tile[field]

        # Copy common attributes
        tvp_attr = set(raster_tvp.ATTRIBUTES.keys())
        for field in tvp_attr.intersection(
                pixc_tile.ATTRIBUTES):
            attr_val = getattr(pixc_tile, field)
            setattr(raster_tvp, field, attr_val)

        return raster_tvp

    def __add__(self, other):
        """ Add other to self """

        # discard overlapping TVP records
        time = np.concatenate((self.time, other.time))
        [junk, indx] = np.unique(time, return_index=True)
        klass = RasterTVP()
        for key in klass.VARIABLES:
            setattr(klass, key, np.concatenate((
                getattr(self, key), getattr(other, key)))[indx])
        return klass
