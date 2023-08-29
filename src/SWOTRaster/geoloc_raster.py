'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben (adapted from geoloc_river)
'''

import logging
import numpy as np
import multiprocessing
import SWOTWater.aggregate as ag
import cnes.modules.geoloc.lib.geoloc as geoloc
import cnes.common.service_error as service_error

from itertools import chain
from functools import partial
from SWOTRaster.raster_agg import fn_star, fn_it, chunk_it
from SWOTRaster.products import RasterUTM, DEFAULT_MAX_CHUNK_SIZE
from cnes.common.lib.my_variables import GEN_RAD_EARTH_EQ, GEN_RAD_EARTH_POLE

LOGGER = logging.getLogger(__name__)

class GeolocRaster(object):
    def __init__(self, pixc, raster, algorithmic_config, max_worker_processes=1):
        self.pixc = pixc
        self.raster = raster
        self.algorithmic_config = algorithmic_config
        self.max_worker_processes = max_worker_processes

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
                max_chunk_size = self.algorithmic_config[
                    'utm_conversion_max_chunk_size']
            except KeyError:
                max_chunk_size = DEFAULT_MAX_CHUNK_SIZE

            proj_mapping = self.raster.get_raster_mapping(
                self.pixc, all_classes_mask, False, max_chunk_size)
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

        # Init output vectors
        self.out_lat_corr = np.ma.masked_all(
            len(self.pixc['pixel_cloud']['latitude']))
        self.out_lon_corr = np.ma.masked_all(
            len(self.pixc['pixel_cloud']['longitude']))
        self.out_height_corr = np.ma.masked_all(
            len(self.pixc['pixel_cloud']['height']))

        # If there are no input pixc samples, return
        if len(self.pixc['pixel_cloud']['height']) == 0:
            return

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
        xyz = np.transpose(np.array([x, y, z]))

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

        nadir_xyz = np.transpose(
            np.array([nadir_x_vect, nadir_y_vect, nadir_z_vect]))
        nadir_v_xyz = np.transpose(
            np.array([nadir_vx_vect, nadir_vy_vect, nadir_vz_vect]))

        # Split the data into more manageable chunks and geolocate
        try:
            max_chunk_size = self.algorithmic_config[
                'height_constrained_geoloc_max_chunk_size']
        except KeyError:
            max_chunk_size = DEFAULT_MAX_CHUNK_SIZE

        args = (xyz, h_noisy, nadir_xyz, nadir_v_xyz, ri, h_new)
        geoloc_fn = partial(geoloc.pointcloud_height_geoloc_vect,
                            recompute_doppler=True, recompute_range=True,
                            verbose=False, max_iter_grad=1, height_goal=1.e-3)
        _geoloc_fn = partial(fn_star, geoloc_fn)
        if self.max_worker_processes > 1:
            chunk_size = int(max(1, min(
                np.ceil(len(h_new)/(self.max_worker_processes*4)), max_chunk_size)))
            with multiprocessing.get_context('spawn').Pool(
                    processes=self.max_worker_processes) as pool:
                result_chunks = list(
                    pool.imap(_geoloc_fn, zip(*(
                        fn_it(chunk_it(arg, chunk_size), np.array) for arg in args))))
        else:
            chunk_size = max_chunk_size
            result_chunks = [
                _geoloc_fn(arglist) for arglist in zip(*(
                    fn_it(chunk_it(arg, chunk_size), np.array) for arg in args))]

        # Merge chunks
        (self.out_lat_corr[mask], self.out_lon_corr[mask],
         self.out_height_corr[mask]) = np.transpose(list(chain.from_iterable(
             result[1] for result in result_chunks)))
