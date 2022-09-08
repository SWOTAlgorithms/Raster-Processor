'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''

import logging
import argparse
import warnings
import numpy as np

from osgeo import osr

WGS84_ID = 4326
UTM_NUM_ZONES = 60
MGRS_VALID_BANDS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

ELLIPSOID_SEMI_MAJOR_AXIS = 6378137.0
ELLIPSOID_INVERSE_FLATTENING = 298.257223563
ELLIPSOID_FLATTENING = 1.0/ELLIPSOID_INVERSE_FLATTENING
ELLIPSOID_SEMI_MINOR_AXIS = ELLIPSOID_SEMI_MAJOR_AXIS*(1-ELLIPSOID_FLATTENING)

LOGGER = logging.getLogger(__name__)

def wgs84_px_area(center_lat, px_size):
    # Calculates the area of a pixel by getting the total area between
    # the lat bounds and taking the fraction of that area between the lon bounds
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(WGS84_ID)
    semi_maj = spatial_ref.GetSemiMajor()
    semi_min = spatial_ref.GetSemiMinor()
    e = np.sqrt(1 - (semi_min/semi_maj)**2)
    area_list = []
    for f in [center_lat + px_size/2, center_lat - px_size/2]:
        zm = 1 - e*np.sin(np.deg2rad(f))
        zp = 1 + e*np.sin(np.deg2rad(f))
        area_list.append(
            np.pi * semi_min**2 * (
                2 * np.arctanh(e*np.sin(np.deg2rad(f))) / (2*e) +
                np.sin(np.deg2rad(f)) / (zp*zm)))
    return px_size / 360. * (area_list[0] - area_list[1])


def is_utm_zone_valid(utm_zone):
    # Checks in a UTM zone is valid
    return (1 <= utm_zone <= UTM_NUM_ZONES)


def is_mgrs_band_valid(mgrs_band):
    # Checks if an MGRS band is valid
    return (mgrs_band.lower() in MGRS_VALID_BANDS.lower())


def utm_zone_from_latlon(latitude, longitude):
    # Gets the utm zone for a given lat/lon
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            return 31
        elif longitude < 21:
            return 33
        elif longitude < 33:
            return 35
        elif longitude < 42:
            return 37

    return int((longitude + 180) / 6) + 1


def mgrs_band_from_latlon(latitude, longitude):
    # Gets the MGRS band for a given lat/lon
    if latitude < -80:
        if longitude < 0:
            return 'A'
        else:
            return 'B'
    elif latitude < 80:
        return MGRS_VALID_BANDS[2 + (int(latitude + 80) >> 3)]
    elif latitude < 84:
        return 'X'
    else:
        if longitude < 0:
            return 'Y'
        else:
            return 'Z'


def mgrs_band_shift(mgrs_band, shift, longitude):
    # Shifts an mgrs band by a given number of steps
    band_num = MGRS_VALID_BANDS.find(mgrs_band) + shift
    if band_num < MGRS_VALID_BANDS.find('C'):
        if longitude < 0:
            return 'A'
        else:
            return 'B'
    elif band_num > MGRS_VALID_BANDS.find('X'):
        if longitude < 0:
            return 'Y'
        else:
            return 'Z'
    else:
        return MGRS_VALID_BANDS[band_num]


def hemisphere_from_mgrs_band(mgrs_band):
    # Gets the hemisphere from an MGRS band
    if not is_mgrs_band_valid(mgrs_band):
        raise ValueError("Invalid MGRS Band: {}".format(mgrs_band))
    if mgrs_band >= 'N':
        return 'N'
    else:
        return 'S'


def utm_zone_identifier(utm_zone, hemisphere):
    # Gets the EPGS identifier for the utm zone
    if not is_utm_zone_valid(utm_zone):
        raise ValueError("Invalid UTM Zone: {}".format(utm_zone))

    if hemisphere == 'N':
        hemisphere_id = "6"
    elif hemisphere == 'S':
        hemisphere_id = "7"
    else:
        raise ValueError("Invalid hemisphere: {}".format(hemisphere))

    identifier = "32" + hemisphere_id + str(utm_zone).zfill(2)
    return int(identifier)


def utm_crs(utm_zone, mgrs_band):
    # Gets a UTM Coordinate Reference System
    if not is_utm_zone_valid(utm_zone):
        raise ValueError("Invalid UTM Zone: {}".format(utm_zone))
    if not is_mgrs_band_valid(mgrs_band):
        raise ValueError("Invalid MGRS Band: {}".format(mgrs_band))

    hemisphere = hemisphere_from_mgrs_band(mgrs_band)
    utm_zone_id = utm_zone_identifier(utm_zone, hemisphere)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(utm_zone_id)
    return spatial_ref


def wgs84_crs():
    # Gets the WGS84 Coordinate Reference System
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(WGS84_ID)
    return spatial_ref


def lon_360to180(longitude):
    # Converts 0-360 degree longitude to -180-180 degree longitude
    return np.mod(longitude + 180, 360) - 180


def xyz2llh(xyz):
    # Converts ECEF XYZ coordinates into LLH coordinates (in radians)
    warnings.simplefilter("ignore")
    llh = np.zeros_like(xyz)
    def cbrt(x):
        '''The cube root with negative handling.'''
        sign = x/np.abs(x)
        cbrt = sign*np.power(np.abs(x), 1.0/3)
        return cbrt
    # Use the WGS84 ellipsoid
    f = ELLIPSOID_FLATTENING
    e2 = f * (2.0 - f)
    e4 = e2**2
    a2 = (ELLIPSOID_SEMI_MAJOR_AXIS)**2

    # This is the algorithm described in
    # Vermeille H (2002) Direct transformation from geocentric
    # coordinates to geodetic coordinates. J Geodesy 76: 451-454
    # which is an exact solution and stable at the poles.
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    hxy = np.hypot(x, y)
    p = hxy**2 / a2
    q = (1.0 - e2) * (z**2) / a2
    r = (p + q - e4) / 6.0
    s = e4 * p * q / (4.0 * r**3)
    t = cbrt(1.0 + s + np.sqrt(s * (2.0 + s)))
    u = r * (1.0 + t + 1.0/t)
    v = np.sqrt(u**2 + e4*q)
    w = e2 * (u + v - q) / (2.0 * v)
    k = np.sqrt(u + v + w**2) - w
    d = k * hxy / (k + e2)
    hdz = np.hypot(d, z)

    llh[0] = 2.0 * np.arctan(z / (d + hdz))
    llh[1] = 2.0 * np.arctan2(y, (x + hxy))
    llh[2] = (k + e2 - 1.0) / k * hdz

    warnings.resetwarnings()
    return llh


def llh2xyz(llh):
    # Converts LLH coordinates (in radians) into ECEF XYZ coordinates
    warnings.simplefilter("ignore")
    xyz = np.zeros_like(llh)
    radius = ELLIPSOID_SEMI_MAJOR_AXIS
    f = ELLIPSOID_FLATTENING
    e2 = f * (2.0 - f)
    sin_lat = np.sin(llh[0]);
    cos_lat = np.cos(llh[0]);
    radius_east = radius/np.sqrt(1.0 - e2*sin_lat**2)
    xyz[0] = (radius_east + llh[2])*cos_lat*np.cos(llh[1])
    xyz[1] = (radius_east + llh[2])*cos_lat*np.sin(llh[1])
    xyz[2] = (radius_east*(1.0-e2) + llh[2])*sin_lat
    warnings.resetwarnings()
    return xyz


def llh2bearing(llh0, llh1):
    # Returns the bearing from one set of LLH coordinates to another set of LLH
    # coordinates
    x = np.cos(llh1[0])*np.sin(llh1[1]-llh0[1])
    y = np.cos(llh0[0])*np.sin(llh1[0]) \
        - np.sin(llh0[0])*np.cos(llh1[0])*np.cos(llh1[1]-llh0[1])
    return np.arctan2(x,y)


def terminal_loc_spherical(latitude, longitude, distance, bearing):
    # Returns the latitude and longitude of a location at a given distance and
    # bearing from the original point using a local spherical approximation to
    # the ellipsoid

    llh = np.row_stack((latitude, longitude, np.zeros_like(latitude)))
    xyz = llh2xyz(llh)

    alpha = distance/ELLIPSOID_SEMI_MAJOR_AXIS
    ueast = np.row_stack((-np.sin(llh[1]), np.cos(llh[1]), 0))
    unorth = np.row_stack((-np.sin(llh[0])*np.cos(llh[1]),
                           -np.sin(llh[0])*np.sin(llh[1]),
                           np.cos(llh[0])))
    ub = unorth*np.cos(bearing)+ueast*np.sin(bearing)
    uh = np.row_stack((np.cos(llh[0])*np.cos(llh[1]),
                       np.cos(llh[0])*np.sin(llh[1]),
                       np.sin(llh[0])))
    sphere_center_pos = xyz-ELLIPSOID_SEMI_MAJOR_AXIS*uh
    new_xyz = sphere_center_pos + ELLIPSOID_SEMI_MAJOR_AXIS \
              * (uh*np.cos(alpha)+ub*np.sin(alpha))
    new_llh = xyz2llh(new_xyz)
    return new_llh[0][0], new_llh[1][0]

