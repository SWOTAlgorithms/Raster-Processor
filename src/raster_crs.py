#!/usr/bin/env python
'''
Copyright (c) 2020-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''

import logging
import argparse
import numpy as np
from osgeo import osr

LOGGER = logging.getLogger(__name__)

EARTH_RAD = 6378137 # As defined in granule/sampling doc
WGS84_ID = 4326
UTM_NUM_ZONES = 60
MGRS_VALID_BANDS = "CDEFGHJKLMNPQRSTUVWXX"

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


def mgrs_band_from_lat(latitude):
    # Gets the MGRS band for a given lat (Doesn't handle polar bands)
    if -80 <= latitude <= 84:
        return MGRS_VALID_BANDS[int(latitude + 80) >> 3]
    else:
        return None


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


def terminal_loc_spherical(latitude, longitude, distance, bearing):
    # Gets the lat/lon of a location a given distance and bearing from the
    # original point using a spherical approximation
    new_latitude = np.arcsin(
        np.sin(latitude) * np.cos(distance/EARTH_RAD) \
        + np.cos(latitude) * np.sin(distance/EARTH_RAD) * np.cos(bearing))
    new_longitude = longitude + np.arctan2(
        np.sin(bearing) * np.sin(distance/EARTH_RAD) * np.cos(latitude), \
        np.cos(distance/EARTH_RAD) - np.sin(latitude) * np.sin(new_latitude))

    return new_latitude, new_longitude


def lon_360to180(longitude):
    return np.mod(longitude + 180, 360) - 180
