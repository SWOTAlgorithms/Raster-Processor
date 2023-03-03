'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben (adapted from geoloc_river)
'''

import logging
import numpy as np
import SWOTWater.aggregate as ag
import cnes.modules.geoloc.lib.geoloc as geoloc
import cnes.common.service_error as service_error

from SWOTRaster.products import RasterUTM
from cnes.common.lib.my_variables import GEN_RAD_EARTH_EQ, GEN_RAD_EARTH_POLE

LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 100000

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

        all_classes_mask = self.pixc.get_mask(
            all_classes, use_improved_geoloc=False)

        if isinstance(self.raster, RasterUTM):
            try:
                chunk_size = self.algorithmic_config['utm_conversion_chunk_size']
            except KeyError:
                chunk_size = DEFAULT_CHUNK_SIZE

            proj_mapping = self.raster.get_raster_mapping(
                self.pixc, all_classes_mask, False, chunk_size)
        else:
            proj_mapping = self.raster.get_raster_mapping(
                self.pixc, all_classes_mask, False)

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

        mask = np.logical_not(np.logical_or.reduce((
            np.ma.getmaskarray(self.pixc['pixel_cloud']['height']),
            np.ma.getmaskarray(self.pixc['pixel_cloud']['latitude']),
            np.ma.getmaskarray(self.pixc['pixel_cloud']['longitude']))))

        mask_indices = np.where(mask)[0]

        nb_pix = np.sum(mask)

        # Convert geodetic coordinates (lat, lon, height) to
        # cartesian coordinates (x, y, z)
        x, y, z = geoloc.convert_llh2ecef(
            self.pixc['pixel_cloud']['latitude'][mask],
            self.pixc['pixel_cloud']['longitude'][mask],
            self.pixc['pixel_cloud']['height'][mask],
            GEN_RAD_EARTH_EQ, GEN_RAD_EARTH_POLE)

        # Get position of associated along-track pixels
        # (in cartesian coordinates)
        nadir_x = self.pixc['tvp']['x']
        nadir_y = self.pixc['tvp']['y']
        nadir_z = self.pixc['tvp']['z']

        # Get velocity of associated along-track pixels
        # (in cartesian coordinates)
        nadir_vx = self.pixc['tvp']['vx']
        nadir_vy = self.pixc['tvp']['vy']
        nadir_vz = self.pixc['tvp']['vz']

        # Get distance from satellite to target point
        ri = self.pixc.near_range \
             + (self.pixc['pixel_cloud']['range_index'][mask]
                * self.pixc.nominal_slant_range_spacing)

        # Remap illumination time to nearest sensor index
        sensor_s = ag.get_sensor_index(self.pixc)[mask]

        # Loop over each pixel (could be vectorized)
        h_noisy = self.pixc['pixel_cloud']['height'][mask]
        h_new = self.new_height[mask]
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

        # Split the data into more manageable chunks and geolocate
        # Init output vectors
        self.out_lat_corr = np.ma.masked_all(
            len(self.pixc['pixel_cloud']['latitude']))
        self.out_lon_corr = np.ma.masked_all(
            len(self.pixc['pixel_cloud']['longitude']))
        self.out_height_corr = np.ma.masked_all(
            len(self.pixc['pixel_cloud']['height']))

        try:
            chunk_size = self.algorithmic_config['height_constrained_geoloc_chunk_size']
        except KeyError:
            chunk_size = DEFAULT_CHUNK_SIZE

        for start_idx in np.arange(0, nb_pix, chunk_size):
            end_idx = min(start_idx+chunk_size, nb_pix)
            p_final, p_final_llh, h_mu, (iter_grad, nfev_minimize_scalar) = \
                geoloc.pointcloud_height_geoloc_vect(
                    np.transpose(np.array([x[start_idx:end_idx],
                                           y[start_idx:end_idx],
                                           z[start_idx:end_idx]])),
                    h_noisy[start_idx:end_idx],
                    np.transpose(np.array([nadir_x_vect[start_idx:end_idx],
                                           nadir_y_vect[start_idx:end_idx],
                                           nadir_z_vect[start_idx:end_idx]])),
                    np.transpose(np.array([nadir_vx_vect[start_idx:end_idx],
                                           nadir_vy_vect[start_idx:end_idx],
                                           nadir_vz_vect[start_idx:end_idx]])),
                    ri[start_idx:end_idx],
                    h_new[start_idx:end_idx],
                    recompute_doppler=True, recompute_range=True, verbose=False,
                    max_iter_grad=1, height_goal=1.e-3)

            self.out_lat_corr[mask_indices[start_idx:end_idx]] = p_final_llh[:,0]
            self.out_lon_corr[mask_indices[start_idx:end_idx]] = p_final_llh[:,1]
            self.out_height_corr[mask_indices[start_idx:end_idx]] = p_final_llh[:,2]
