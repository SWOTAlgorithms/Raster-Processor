#!/usr/bin/env python
'''
Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben (adapted from geoloc_river)

'''

import os
import raster
import logging
import argparse
import numpy as np
import raster_products
import SWOTWater.aggregate as ag
import cnes.modules.geoloc.lib.geoloc as geoloc
import cnes.common.service_error as service_error

from raster_products import RasterPixc
from pixc_to_raster import load_raster_configs
from SWOTWater.products.product import MutableProduct
from cnes.common.lib.my_variables import GEN_RAD_EARTH_EQ, GEN_RAD_EARTH_POLE

LOGGER = logging.getLogger(__name__)

class GeolocRaster(object):
    def __init__(self, pixc, raster, algorithmic_config):
        self.pixc = pixc
        self.raster = raster
        self.algorithmic_config = algorithmic_config

    def process(self):
        """ Do improved raster geolocation """
        LOGGER.info("processing")

        self.update_heights_from_raster()
        self.apply_improved_geoloc()

        return (self.out_lat_corr,
                self.out_lon_corr,
                self.out_height_corr)

    def update_heights_from_raster(self):
        """ Update pixelcloud heights from raster """
        LOGGER.info("updating heights from raster")

        self.new_height = self.pixc['pixel_cloud']['height'].copy()

        all_classes = np.concatenate(
            (self.algorithmic_config['interior_water_classes'],
             self.algorithmic_config['water_edge_classes'],
             self.algorithmic_config['land_edge_classes'],
             self.algorithmic_config['dark_water_classes']))
        common_qual_flags = ['pixc_line_qual', 'interferogram_qual',
                             'classification_qual', 'geolocation_qual']
        all_mask = self.pixc.get_mask(
            all_classes, common_qual_flags, use_improved_geoloc=False)

        proj_mapping = self.raster.get_raster_mapping(self.pixc, all_mask,
                                                      use_improved_geoloc=False)

        raster_uncorrected_height = self.raster.get_uncorrected_height()

        for i in range(0, len(proj_mapping)):
            for j in range(0, len(proj_mapping[0])):
                if not np.ma.is_masked(raster_uncorrected_height[i][j]):
                    for k in proj_mapping[i][j]:
                        self.new_height[k] = raster_uncorrected_height[i][j]

    def apply_improved_geoloc(self):
        """ Compute the new lat, lon, height using the new heights """
        LOGGER.info("applying improved geolocation")

        method = self.algorithmic_config[ \
            'lowres_raster_height_constrained_geoloc_method']
        if method == 'taylor':
            self.taylor_improved_geoloc()
        else:
            message = "the method " + str(method) + " is undefined"
            raise service_error.ParameterError("apply_improved_geoloc", message)

    def taylor_improved_geoloc(self):
        """ Improve the height of noisy point (in object sensor) """
        LOGGER.info("doing taylor improved geolocation")

        nb_pix = self.pixc['pixel_cloud']['height'].size
        # Convert geodetic coordinates (lat, lon, height) to cartesian coordinates (x, y, z)
        x, y, z = geoloc.convert_llh2ecef(self.pixc['pixel_cloud']['latitude'],
                                          self.pixc['pixel_cloud']['longitude'],
                                          self.pixc['pixel_cloud']['height'],
                                          GEN_RAD_EARTH_EQ, GEN_RAD_EARTH_POLE)

        # Get position of associated along-track pixels (in cartesian coordinates)
        nadir_x = self.pixc['tvp']['x']
        nadir_y = self.pixc['tvp']['y']
        nadir_z = self.pixc['tvp']['z']

        # Get velocity of associated along-track pixels (in cartesian coordinates)
        nadir_vx = self.pixc['tvp']['vx']
        nadir_vy = self.pixc['tvp']['vy']
        nadir_vz = self.pixc['tvp']['vz']

        # Get distance from satellite to target point
        ri = self.pixc.near_range + (self.pixc['pixel_cloud']['range_index']
                                     * self.pixc.nominal_slant_range_spacing)

        # Init output vectors
        self.out_lat_corr = np.zeros(nb_pix)  # Improved latitudes
        self.out_lon_corr = np.zeros(nb_pix)  # Improved longitudes
        self.out_height_corr = np.zeros(nb_pix)  # Improved heights

        # Remap illumnation time to nearest sensor index
        sensor_s = ag.get_sensor_index(self.pixc)

        # Loop over each pixel (could be vectorized)
        h_noisy = self.pixc['pixel_cloud']['height']
        nadir_x_vect = np.zeros(nb_pix)
        nadir_y_vect = np.zeros(nb_pix)
        nadir_z_vect = np.zeros(nb_pix)
        nadir_vx_vect = np.zeros(nb_pix)
        nadir_vy_vect = np.zeros(nb_pix)
        nadir_vz_vect = np.zeros(nb_pix)

        for i in np.arange(nb_pix):
            ind_sensor = sensor_s[i]
            nadir_x_vect[i] = nadir_x[ind_sensor]
            nadir_y_vect[i] = nadir_y[ind_sensor]
            nadir_z_vect[i] = nadir_z[ind_sensor]
            nadir_vx_vect[i] = nadir_vx[ind_sensor]
            nadir_vy_vect[i] = nadir_vy[ind_sensor]
            nadir_vz_vect[i] = nadir_vz[ind_sensor]

        # improve height with vectorised pixel
        p_final, p_final_llh, h_mu, (iter_grad, nfev_minimize_scalar) = \
            geoloc.pointcloud_height_geoloc_vect(np.transpose(np.array([x, y, z])),
                                                 h_noisy,
                                                 np.transpose(np.array(
                                                     [nadir_x_vect, nadir_y_vect,
                                                      nadir_z_vect])),
                                                 np.transpose(np.array(
                                                     [nadir_vx_vect, nadir_vy_vect,
                                                      nadir_vz_vect])),
                                                 ri, self.new_height,
                                                 recompute_doppler=True,
                                                 recompute_range=True, verbose=False,
                                                 max_iter_grad=1, height_goal=1.e-3)

        self.out_lat_corr, self.out_lon_corr, self.out_height_corr = np.transpose(p_final_llh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pixc_file')
    parser.add_argument('raster_file')
    parser.add_argument('alg_config_file')
    parser.add_argument('runtime_config_file')
    parser.add_argument('out_pixc_file')
    args = parser.parse_args()

    alg_cfg, rt_cfg = load_raster_configs(args.alg_config_file,
                                          args.runtime_config_file)

    pixc_tile = MutableProduct.from_ncfile(args.pixc_file)
    pixc_prod = RasterPixc.from_tile(pixc_tile, None)

    if rt_cfg['output_sampling_grid_type'] == 'utm':
        if alg_cfg['debug_flag']:
            raster_prod = \
                raster_products.RasterUTMDebug.from_ncfile(args.raster_file)
        else:
            raster_prod = \
                raster_products.RasterUTM.from_ncfile(args.raster_file)
    elif rt_cfg['output_sampling_grid_type'] == 'geo':
        if alg_cfg['debug_flag']:
            raster_prod = \
                raster_products.RasterGeoDebug.from_ncfile(args.raster_file)
        else:
            raster_prod = \
                raster_products.RasterGeo.from_ncfile(args.raster_file)

    geolocator = Geoloc_raster(pixc_prod, raster_prod, alg_cfg)
    out_lat, out_lon, out_height = geolocator.process()

    pixc_prod['pixel_cloud']['height'][:] = out_height
    pixc_prod['pixel_cloud']['latitude'][:] = out_lat
    pixc_prod['pixel_cloud']['longitude'][:] = out_lon
    pixc_prod.to_ncfile(args.out_pixc_file)
