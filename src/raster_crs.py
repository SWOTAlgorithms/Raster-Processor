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

# UTM Constants
UTM_NUM_ZONES = 60
UTM_VALID_BANDS = "CDEFGHJKLMNPQRSTUVWX"

# WGS84 Constants
WGS84_ID = 4326

def wgs84_px_area(center_lat, px_size):
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(WGS84_ID)
    semi_maj = spatial_ref.GetSemiMajor()
    semi_min = spatial_ref.GetSemiMinor()
    e = np.sqrt(1 - (semi_min/semi_maj)**2)
    area_list = []
    for f in [center_lat + px_size/2, center_lat - px_size/2]:
        zm = 1 - e*np.sin(np.radians(f))
        zp = 1 + e*np.sin(np.radians(f))
        area_list.append(
            np.pi * semi_min**2 * (
                2 * np.arctanh(e*np.sin(np.radians(f))) / (2*e) +
                np.sin(np.radians(f)) / (zp*zm)))
    return px_size / 360. * (area_list[0] - area_list[1])


def utm_hemisphere(latitude_band):
    if latitude_band not in UTM_VALID_BANDS:
        raise ValueError("Invalid Latitude Band: {}".format(latitude_band))
    if latitude_band >= 'N':
        return 'N'
    else:
        return 'S'


def utm_zone_identifier(utm_zone, hemisphere):
    if utm_zone > UTM_NUM_ZONES or utm_zone < 1:
        raise ValueError("Invalid UTM Zone: {}".format(utm_zone))

    if hemisphere == 'N':
        hemisphere_id = "6"
    elif hemisphere == 'S':
        hemisphere_id = "7"
    else:
        raise ValueError("Invalid hemisphere: {}".format(hemisphere))

    identifier = "32" + hemisphere_id + str(utm_zone).zfill(2)
    return int(identifier)


def utm_crs(utm_zone, latitude_band):
    hemisphere = utm_hemisphere(latitude_band)
    utm_zone_id = utm_zone_identifier(utm_zone, hemisphere)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(utm_zone_id)
    return spatial_ref


def wgs84_crs():
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(WGS84_ID)
    return spatial_ref


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("proj_type", type=str)
    parser.add_argument("--utm_zone", type=int, default=None)
    parser.add_argument("--utm_lat_band", type=str, default=None)
    args = parser.parse_args()

    if args.proj_type == 'geo':
        proj = wgs84_crs()
    elif args.proj_type == 'utm':
        if args.utm_zone is not None and args.utm_lat_band is not None:
            proj = utm_crs(args.utm_zone, args.utm_lat_band)
        else:
            LOGGER.error('Must define zone and lat band for utm projection')
    else:
        LOGGER.error('Invalid projection: {}'.format(args.proj_type))

    try:
        print(proj.ExportToWkt())
    except:
        pass
