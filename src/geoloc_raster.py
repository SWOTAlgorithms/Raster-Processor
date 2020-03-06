#!/usr/bin/env python
'''
Improved geolocation for raster processor
Adapted from geoloc_river from CNES swot-hydrology-toolbox

Author (s): Alexander Corben
'''
import os
import rdf
import raster
import logging
import argparse
import numpy as np
import cnes.modules.geoloc.lib.geoloc as geoloc
import cnes.common.service_error as service_error

from scipy import interpolate
from SWOTWater.constants import PIXC_CLASSES
from SWOTWater.products.product import MutableProduct
from cnes.common.lib.my_variables import GEN_RAD_EARTH_EQ, GEN_RAD_EARTH_POLE

downsamp_rate = 1

class GeolocRaster(object):
    """
        class GeolocRaster
    """
    def __init__(self, pixc, raster, raster_config):
        logger = logging.getLogger(self.__class__.__name__)
        logger.info("GeolocRaster initialization")

        self.pixc = pixc
        self.raster = raster
        self.raster_config = raster_config

    def update_heights_from_raster(self):
        """
        Update pixelcloud heights from raster
        """
        self.new_height = np.ma.all_masked_like(self.pixc['pixel_cloud'].height)

        pixc_mask = raster.get_pixc_mask(self.pixc)
        pixc_mask = np.logical_and(
            pixc_mask,
            np.isin(self.pixc['pixel_cloud']['classification'],
                    np.concatenate((self.raster_config['interior_water_classes'],
                                    self.raster_config['water_edge_classes'],
                                    self.raster_config['land_edge_classes'],
                                    self.raster_config['dark_water_classes']))))
        proj_mapping = raster.get_raster_mapping(self.pixc, pixc_mask)

        for i in range(0, self.raster.size_y):
            for j in range(0, self.raster.size_x):
                for k in proj_mapping[i][j]:
                    self.new_height[k] = self.raster.height[i][j]

    def apply_improved_geoloc(self):
        """ Compute the new lat, lon, height using the new heights """
        method = self.raster_config['improved_geolocation_method']
        if method == 'taylor':
            self.taylor_improved_geoloc()
        else:
            message = "the method " + str(method) + " is undefined"
            raise service_error.ParameterError("apply_improved_geoloc", message)

    def taylor_improved_geoloc(self):
        """
        Improve the height of noisy point (in object sensor)
        """
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
        ri = self.pixc.near_range + self.pixc['pixel_cloud']['range_index'] * self.pixc.nominal_slant_range_spacing

        # Init output vectors
        self.out_lat_corr = np.zeros(nb_pix)  # Improved latitudes
        self.out_lon_corr = np.zeros(nb_pix)  # Improved longitudes
        self.out_height_corr = np.zeros(nb_pix)  # Improved heights

        # need to remap illumnation time to nearest sensor index
        # TODO replace this by a call to a get_sensor_index or equivalent function
        # that either interpolates the sensor or does something more efficient
        f = interpolate.interp1d(self.pixc['tvp']['time'], range(len(self.pixc['tvp']['time'])))
        illumination_time = self.pixc['pixel_cloud']['illumination_time'].data[~self.pixc['pixel_cloud']['illumination_time'].mask]
        sensor_s = (np.rint(f(illumination_time))).astype(int).T

        # Loop over each pixel (could be vectorized)
        # vectorisation
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
        p_final, p_final_llh, h_mu, (iter_grad, nfev_minimize_scalar) = geoloc.pointcloud_height_geoloc_vect(np.transpose(np.array([x, y, z])),
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


def geoloc_raster(pixc_prod, raster_prod, raster_config):
    """ Improved raster geolocation """
    geoloc_raster = GeolocRaster(pixc_prod, raster_prod, raster_config)
    # Do the improved raster geolocation
    logger = logging.getLogger()
    logger.info("Improved geolocation")
    geoloc_raster.update_heights_from_raster()
    geoloc_raster.apply_improved_geoloc()

    return geoloc_raster.out_lat_corr, geoloc_raster.out_lon_corr, geoloc_raster.out_height_corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pixc_file')
    parser.add_argument('raster_file')
    parser.add_argument('raster_config')
    parser.add_argument('out_pixc_file')
    args = parser.parse_args()

    pixc_prod = MutableProduct.from_ncfile(args.pixc_file)
    raster_prod = MutableProduct.from_ncfile(args.raster_file)
    raster_config = rdf.parse(os.path.abspath(args.raster_config), comment='!')
    out_lat, out_lon, out_height = geoloc_raster(pixc_prod,
                                                 raster_prod,
                                                 raster_config)
    pixc_prod['pixel_cloud']['latitude'][:] = out_lat
    pixc_prod['pixel_cloud']['longitude'][:] = out_lon
    pixc_prod.to_ncfile(args.out_pixc_file)