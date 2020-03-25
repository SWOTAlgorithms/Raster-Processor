#!/usr/bin/env python
'''
Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''

import utm
import logging
import argparse
import numpy as np

LOGGER = logging.getLogger(__name__)

# CR Constants
DEGREE_SCALE_FACTOR = 0.0174532925199433

# WGS84 Constants
WGS84_SEMI_MAJOR_AXIS = 6378137
WGS84_INVERSE_FLATTENING = 298.257223563

# UTM Constants
UTM_NUM_ZONES = 60
UTM_FALSE_EASTING = 500000.
UTM_FALSE_NORTHING_N = 0.
UTM_FALSE_NORTHING_S = 10000000.
UTM_SCALE_FACTOR = 0.9996


class Projection(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        wkt = "PROJECTION[\"{0}\"]".format(self.name)
        return wkt


class Parameter(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        wkt = "PARAMETER[\"{0}\",{1}]".format(self.name, self.value)
        return wkt


class Identifier(object):
    def __init__(self, name, identifier):
        self.name = name
        self.identifier = identifier

    def __str__(self):
        wkt = "ID[\"{0}\",\"{1}\"]".format(self.name, self.identifier)
        return wkt


class Axis(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        wkt = "AXIS[\"{0}\",{1}]".format(self.name, self.value)
        return wkt


class Unit(object):
    def __init__(self, name, conversion_factor, identifier=None):
        self.name = name
        self.conversion_factor = conversion_factor
        self.identifier = identifier

    def __str__(self):
        wkt = "UNIT[\"{0}\",{1}".format(
            self.name, self.conversion_factor)
        if self.identifier is None:
            wkt = wkt + "]"
        else:
            wkt = wkt + ",{0}]".format(self.identifier)
        return wkt


class PrimeM(object):
    def __init__(self, name, longitude, identifier=None):
        self.name = name
        self.longitude = longitude
        self.identifier = identifier

    def __str__(self):
        wkt = "PRIMEM[\"{0}\",{1}".format(
            self.name, self.longitude)
        if self.identifier is None:
            wkt = wkt + "]"
        else:
            wkt = wkt + ",{0}]".format(self.identifier)
        return wkt


class Spheroid(object):
    def __init__(self, name, semi_major_axis, inverse_flattening, identifier=None):
        self.name = name
        self.semi_major_axis = semi_major_axis
        self.inverse_flattening = inverse_flattening
        self.identifier = identifier

    def __str__(self):
        wkt = "SPHEROID[\"{0}\",{1},{2}".format(
            self.name, self.semi_major_axis, self.inverse_flattening)
        if self.identifier is None:
            wkt = wkt + "]"
        else:
            wkt = wkt + ",{1}]".format(wkt, self.identifier)
        return wkt


class Datum(object):
    def __init__(self, name, spheroid, identifier=None):
        self.name = name
        self.spheroid = spheroid
        self.identifier = identifier

    def __str__(self):
        wkt = "DATUM[\"{0}\",{1}".format(
            self.name, self.spheroid)
        if self.identifier is None:
            wkt = wkt + "]"
        else:
            wkt = wkt + ",{0}]".format(self.identifier)
        return wkt


class GeogCRS(object):
    def __init__(self, name, datum, primem, unit, identifier=None):
        self.name = name
        self.datum = datum
        self.primem = primem
        self.unit = unit
        self.identifier = identifier

    def __str__(self):
        wkt = "GEOGCRS[\"{0}\",{1},{2},{3}".format(
            self.name, self.datum, self.primem, self.unit)
        if self.identifier is None:
            wkt = wkt + "]"
        else:
            wkt = wkt + ",{0}]".format(self.identifier)
        return wkt


class ProjCRS(object):
    def __init__(self, name, geogcs, projection, params, unit, axes, identifier=None):
        self.name = name
        self.geogcs = geogcs
        self.projection = projection
        for param in params:
            setattr(self, param.name, param)
        self.unit = unit
        for axis in axes:
            setattr(self, axis.name, axis)
        self.identifier = identifier

        self._param_keys = [param.name for param in params]
        self._axis_keys = [axis.name for axis in axes]

    def __str__(self):
        wkt = "PROJCRS[\"{0}\",{1},{2}".format(
            self.name, self.geogcs, self.projection)
        for key in self._param_keys:
            wkt = wkt + ",{0}".format(getattr(self, key))

        wkt = wkt + ",{0}".format(self.unit)

        for key in self._axis_keys:
            wkt = wkt + ",{0}".format(getattr(self, key))

        if self.identifier is None:
            wkt = wkt + "]"
        else:
            wkt = wkt + ",{0}]".format(self.identifier)
        return wkt


def wgs84_crs():
    wgs84_spheroid = Spheroid("WGS 84", WGS84_SEMI_MAJOR_AXIS, WGS84_INVERSE_FLATTENING,
                              Identifier("EPSG", "7030"))
    wgs84_datum = Datum("WGS_1984", wgs84_spheroid, Identifier("EPSG", "6326"))
    greenwich_primem = PrimeM("Greenwich", 0, Identifier("EPSG", "8901"))
    degree_unit = Unit("degree", DEGREE_SCALE_FACTOR, Identifier("EPSG", "9122"))

    name = "WGS 84"
    return GeogCRS(name, wgs84_datum, greenwich_primem, degree_unit,
                   Identifier("EPSG", "4326"))


def get_utm_zone_central_meridian(utm_zone):
    if utm_zone > UTM_NUM_ZONES or utm_zone < 0:
        raise ValueError("Invalid UTM Zone: {}".format(utm_zone))

    return utm_zone*(360/UTM_NUM_ZONES) - 180 - 180/UTM_NUM_ZONES


def get_utm_zone_identifier(utm_zone, hemisphere):
    if utm_zone > UTM_NUM_ZONES or utm_zone < 0:
        raise ValueError("Invalid UTM Zone: {}".format(utm_zone))

    if hemisphere == 'N':
        hemisphere_id = "6"
    elif hemisphere == 'S':
        hemisphere_id = "7"
    else:
        raise ValueError("Invalid hemisphere: {}".format(hemisphere))

    identifier = "32" + hemisphere_id + str(utm_zone)
    return Identifier("EPSG", identifier)


def utm_crs(utm_zone, hemisphere):
    zone_id = get_utm_zone_identifier(utm_zone, hemisphere)
    central_meridian = get_utm_zone_central_meridian(utm_zone)

    if hemisphere == 'N':
        false_northing = UTM_FALSE_NORTHING_N
    elif hemisphere == 'S':
        false_northing = UTM_FALSE_NORTHING_S

    geogcrs = wgs84_crs()
    utm_projection = Projection("Transverse_Mercator")
    params = [Parameter("latitude_of_origin", 0),
              Parameter("central_meridian", central_meridian),
              Parameter("scale_factor", UTM_SCALE_FACTOR),
              Parameter("false_easting", UTM_FALSE_EASTING),
              Parameter("false_northing", false_northing)]
    axes = [Axis("Easting", "EAST"),
            Axis("Northing", "NORTH")]
    unit = Unit("metre", 1, Identifier("EPSG", "9001"))
    name = "WGS 84 / UTM zone {0}{1}".format(utm_zone, hemisphere)
    return ProjCRS(name, geogcrs, utm_projection, params, unit, axes, zone_id)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("proj_type", type=str)
    parser.add_argument("--utm_zone", type=int, default=None)
    parser.add_argument("--utm_hemisphere", type=str, default=None)
    args = parser.parse_args()

    if args.proj_type == 'geo':
        proj = wgs84_crs()
        print(proj)
    elif args.proj_type == 'utm':
        if args.utm_zone is not None and args.utm_hemisphere is not None:
            proj = utm_crs(args.utm_zone, args.utm_hemisphere)
            print(proj)
        else:
            LOGGER.error('Must define zone and hemisphere for utm projection')
    else:
        LOGGER.error('Invalid projection: {}'.format(args.proj_type))
