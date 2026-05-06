'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author (s): Alexander Corben
'''

import os
import logging
import numpy as np
import SWOTRaster.geoloc_raster
import SWOTRaster.raster_proc

from SWOTRaster.errors import RasterUsageException
from SWOTRaster.products import DEFAULT_MAX_CHUNK_SIZE
from SWOTRaster.smooth_slant_plane import smooth_slant_plane, \
    DEFAULT_MAX_CHUNK_SHAPE

LOGGER = logging.getLogger(__name__)

LOWRES_RASTER_FILENAME = 'lowres_wse_raster.nc'
INTERNAL_SCENE_PIXC_FILENAME = 'internal_scene_pixc.nc'


class L2PixcToRaster():
    """ Top level L2 PIXC to Raster processor. """
    def __init__(self, pixc, algorithmic_config, runtime_config,
                 polygon_points=None, data_centroid=None,
                 max_worker_processes=1, scratch_dir=None):
        self.pixc = pixc
        self.algorithmic_config = algorithmic_config
        self.runtime_config = runtime_config
        self.polygon_points = polygon_points
        self.data_centroid = data_centroid
        self.max_worker_processes = max_worker_processes
        self.scratch_dir = scratch_dir

        # Add default optional values to configs
        if 'slant_plane_smoothing_max_chunk_shape' \
           not in self.algorithmic_config:
            self.algorithmic_config[
                'slant_plane_smoothing_max_chunk_shape'] = \
                    DEFAULT_MAX_CHUNK_SHAPE
        if 'height_constrained_geoloc_max_chunk_size' \
           not in self.algorithmic_config:
            self.algorithmic_config[
                'height_constrained_geoloc_max_chunk_size'] = \
                    DEFAULT_MAX_CHUNK_SIZE
        if 'utm_conversion_max_chunk_size' not in self.algorithmic_config:
            self.algorithmic_config['utm_conversion_max_chunk_size'] = \
                DEFAULT_MAX_CHUNK_SIZE
        if 'aggregator_max_chunk_size' not in self.algorithmic_config:
            self.algorithmic_config['aggregator_max_chunk_size'] = \
                DEFAULT_MAX_CHUNK_SIZE
        if 'debug_flag' not in self.algorithmic_config:
            self.algorithmic_config['debug_flag'] = False
        if 'write_internal_files' not in self.algorithmic_config:
            self.algorithmic_config['write_internal_files'] = False
        if 'utm_zone_adjust' not in self.runtime_config:
            self.runtime_config['utm_zone_adjust'] = 0
        if 'mgrs_band_adjust' not in self.runtime_config:
            self.runtime_config['mgrs_band_adjust'] = 0

        # Use default geo qual values if not overridden
        if 'wse_geo_qual_suspect' not in self.algorithmic_config:
            self.algorithmic_config['wse_geo_qual_suspect'] = \
                self.algorithmic_config['geo_qual_suspect']
        if 'wse_geo_qual_degraded' not in self.algorithmic_config:
            self.algorithmic_config['wse_geo_qual_degraded'] = \
                self.algorithmic_config['geo_qual_degraded']
        if 'wse_geo_qual_bad' not in self.algorithmic_config:
            self.algorithmic_config['wse_geo_qual_bad'] = \
                self.algorithmic_config['geo_qual_bad']
        if 'area_geo_qual_suspect' not in self.algorithmic_config:
            self.algorithmic_config['area_geo_qual_suspect'] = \
                self.algorithmic_config['geo_qual_suspect']
        if 'area_geo_qual_degraded' not in self.algorithmic_config:
            self.algorithmic_config['area_geo_qual_degraded'] = \
                self.algorithmic_config['geo_qual_degraded']
        if 'area_geo_qual_bad' not in self.algorithmic_config:
            self.algorithmic_config['area_geo_qual_bad'] = \
                self.algorithmic_config['geo_qual_bad']
        if 'sig0_geo_qual_suspect' not in self.algorithmic_config:
            self.algorithmic_config['sig0_geo_qual_suspect'] = \
                self.algorithmic_config['geo_qual_suspect']
        if 'sig0_geo_qual_degraded' not in self.algorithmic_config:
            self.algorithmic_config['sig0_geo_qual_degraded'] = \
                self.algorithmic_config['geo_qual_degraded']
        if 'sig0_geo_qual_bad' not in self.algorithmic_config:
            self.algorithmic_config['sig0_geo_qual_bad'] = \
                self.algorithmic_config['geo_qual_bad']

        # Use default class qual values if not overridden
        if 'wse_class_qual_suspect' not in self.algorithmic_config:
            self.algorithmic_config['wse_class_qual_suspect'] = \
                self.algorithmic_config['class_qual_suspect']
        if 'wse_class_qual_degraded' not in self.algorithmic_config:
            self.algorithmic_config['wse_class_qual_degraded'] = \
                self.algorithmic_config['class_qual_degraded']
        if 'wse_class_qual_bad' not in self.algorithmic_config:
            self.algorithmic_config['wse_class_qual_bad'] = \
                self.algorithmic_config['class_qual_bad']
        if 'area_class_qual_suspect' not in self.algorithmic_config:
            self.algorithmic_config['area_class_qual_suspect'] = \
                self.algorithmic_config['class_qual_suspect']
        if 'area_class_qual_degraded' not in self.algorithmic_config:
            self.algorithmic_config['area_class_qual_degraded'] = \
                self.algorithmic_config['class_qual_degraded']
        if 'area_class_qual_bad' not in self.algorithmic_config:
            self.algorithmic_config['area_class_qual_bad'] = \
                self.algorithmic_config['class_qual_bad']
        if 'sig0_class_qual_suspect' not in self.algorithmic_config:
            self.algorithmic_config['sig0_class_qual_suspect'] = \
                self.algorithmic_config['class_qual_suspect']
        if 'sig0_class_qual_degraded' not in self.algorithmic_config:
            self.algorithmic_config['sig0_class_qual_degraded'] = \
                self.algorithmic_config['class_qual_degraded']
        if 'sig0_class_qual_bad' not in self.algorithmic_config:
            self.algorithmic_config['sig0_class_qual_bad'] = \
                self.algorithmic_config['class_qual_bad']

    def process(self):
        """ Process L2Pixc to Raster """
        LOGGER.info("processing l2pixc to raster")

        # Get height-constrained geolocation as specified in config:
        # "none" - we want to use non-improved geoloc
        # "lowres_raster" - we want to get height constrained geolocation using
        #                   a lowres raster for improved geoloc
        # "pixcvec" - we want to keep pixcvec geoloc as improved geoloc

        hc_geoloc_source = \
            self.algorithmic_config['height_constrained_geoloc_source']
        if hc_geoloc_source is None or hc_geoloc_source.lower() == "none":
            new_height = self.get_smoothed_height()
            self.pixc['pixel_cloud']['improved_height'] = new_height
            self.use_improved_geoloc = False
        elif hc_geoloc_source.lower() == "lowres_raster":
            new_lat, new_lon, new_height = self.do_lowres_raster_hc_geoloc()
            self.pixc['pixel_cloud']['improved_latitude'] = new_lat
            self.pixc['pixel_cloud']['improved_longitude'] = new_lon
            self.pixc['pixel_cloud']['improved_height'] = new_height
            self.use_improved_geoloc = True
        elif hc_geoloc_source.lower() == "smoothed_slant_plane":
            new_lat, new_lon, new_height = \
                self.do_smoothed_slant_plane_hc_geoloc()
            self.pixc['pixel_cloud']['improved_latitude'] = new_lat
            self.pixc['pixel_cloud']['improved_longitude'] = new_lon
            self.pixc['pixel_cloud']['improved_height'] = new_height
            self.use_improved_geoloc = True
        elif hc_geoloc_source.lower() == "pixcvec":
            self.use_improved_geoloc = True
        else:
            raise RasterUsageException(
                f'Invalid height_constrained_geoloc_source: {hc_geoloc_source}')

        if self.algorithmic_config['write_internal_files']:
            if self.scratch_dir is None:
                self.pixc.to_ncfile(INTERNAL_SCENE_PIXC_FILENAME)
            else:
                self.pixc.to_ncfile(
                    os.path.join(self.scratch_dir,
                                 INTERNAL_SCENE_PIXC_FILENAME))

        product = self.do_raster_processing()

        return product

    def do_lowres_raster_hc_geoloc(self):
        """ Do lowres raster height constrained geolocation """
        LOGGER.info("doing lowres raster height constrained geoloc")

        # Normally land edges wouldn't get raster heights, but we are forcing
        # the land edges to be processed as water edges here. Only side effect
        # is in water_area aggregation, which improved geolocation does not use
        tmp_water_edge_classes = np.concatenate(
            (self.algorithmic_config['water_edge_classes'],
             self.algorithmic_config['land_edge_classes']))
        tmp_land_edge_classes = []

        height_constrained_geoloc_raster_proc = \
            SWOTRaster.raster_proc.RasterProcessor(
                self.runtime_config['output_sampling_grid_type'],
                self.runtime_config['raster_resolution']
                / self.algorithmic_config['lowres_raster_scale_factor'],
                self.algorithmic_config['padding'],
                self.algorithmic_config['height_agg_method'],
                self.algorithmic_config['area_agg_method'],
                self.algorithmic_config['sig0_agg_method'],
                self.algorithmic_config['interior_water_classes'],
                tmp_water_edge_classes,
                tmp_land_edge_classes,
                self.algorithmic_config['dark_water_classes'],
                self.algorithmic_config['low_coh_water_classes'],
                self.algorithmic_config['use_bright_land'],
                self.algorithmic_config['use_specular_not_intersecting_prior'],
                self.algorithmic_config[
                    'specular_not_intersecting_prior_thresh'],
                self.algorithmic_config['use_all_classes_for_wse'],
                self.algorithmic_config['use_all_classes_for_sig0'],
                self.algorithmic_config['wse_geo_qual_suspect'],
                self.algorithmic_config['wse_geo_qual_degraded'],
                self.algorithmic_config['wse_geo_qual_bad'],
                self.algorithmic_config['area_geo_qual_suspect'],
                self.algorithmic_config['area_geo_qual_degraded'],
                self.algorithmic_config['area_geo_qual_bad'],
                self.algorithmic_config['sig0_geo_qual_suspect'],
                self.algorithmic_config['sig0_geo_qual_degraded'],
                self.algorithmic_config['sig0_geo_qual_bad'],
                self.algorithmic_config['wse_class_qual_suspect'],
                self.algorithmic_config['wse_class_qual_degraded'],
                self.algorithmic_config['wse_class_qual_bad'],
                self.algorithmic_config['area_class_qual_suspect'],
                self.algorithmic_config['area_class_qual_degraded'],
                self.algorithmic_config['area_class_qual_bad'],
                self.algorithmic_config['sig0_class_qual_suspect'],
                self.algorithmic_config['sig0_class_qual_degraded'],
                self.algorithmic_config['sig0_class_qual_bad'],
                self.algorithmic_config['sig0_qual_suspect'],
                self.algorithmic_config['sig0_qual_degraded'],
                self.algorithmic_config['sig0_qual_bad'],
                self.algorithmic_config['num_good_sus_pix_thresh_wse'],
                self.algorithmic_config['num_good_sus_pix_thresh_water_area'],
                self.algorithmic_config['num_good_sus_pix_thresh_sig0'],
                self.algorithmic_config['pixc_water_frac_suspect_thresh'],
                self.algorithmic_config['num_wse_pix_suspect_thresh'],
                self.algorithmic_config['num_water_area_pix_suspect_thresh'],
                self.algorithmic_config['num_sig0_pix_suspect_thresh'],
                self.algorithmic_config['near_range_suspect_thresh'],
                self.algorithmic_config['far_range_suspect_thresh'],
                self.algorithmic_config['wse_uncert_suspect_thresh'],
                self.algorithmic_config['water_frac_uncert_suspect_thresh'],
                self.algorithmic_config['sig0_uncert_suspect_thresh'],
                self.algorithmic_config['wse_bad_thresh_min'],
                self.algorithmic_config['wse_bad_thresh_max'],
                self.algorithmic_config['water_frac_bad_thresh_min'],
                self.algorithmic_config['water_frac_bad_thresh_max'],
                self.algorithmic_config['sig0_bad_thresh_min'],
                self.algorithmic_config['sig0_bad_thresh_max'],
                self.algorithmic_config['inner_swath_distance_thresh'],
                self.algorithmic_config['missing_karin_data_time_thresh'],
                utm_zone_adjust=self.runtime_config['utm_zone_adjust'],
                mgrs_band_adjust=self.runtime_config['mgrs_band_adjust'],
                utm_conversion_max_chunk_size=self.algorithmic_config[
                    'utm_conversion_max_chunk_size'],
                aggregator_max_chunk_size=self.algorithmic_config[
                    'aggregator_max_chunk_size'],
                skip_area=True, skip_sig0=True,
                max_worker_processes=self.max_worker_processes,
                debug_flag=self.algorithmic_config['debug_flag'])

        height_constrained_geoloc_raster = \
            height_constrained_geoloc_raster_proc.rasterize(
                self.pixc, self.polygon_points, self.data_centroid,
                use_improved_geoloc=False)

        if self.algorithmic_config['write_internal_files']:
            if self.scratch_dir is None:
                height_constrained_geoloc_raster.to_ncfile(
                    LOWRES_RASTER_FILENAME)
            else:
                height_constrained_geoloc_raster.to_ncfile(
                    os.path.join(self.scratch_dir, LOWRES_RASTER_FILENAME))

        # if the height-constrained geoloc raster is empty, return fully masked
        # output
        if height_constrained_geoloc_raster.is_empty():
            return (
                np.ma.masked_all_like(self.pixc['pixel_cloud']['latitude']),
                np.ma.masked_all_like(self.pixc['pixel_cloud']['longitude']),
                np.ma.masked_all_like(self.pixc['pixel_cloud']['height']))

        geolocator = SWOTRaster.geoloc_raster.GeolocRaster(
            self.pixc, self.algorithmic_config,
            max_worker_processes=self.max_worker_processes)
        geolocator.set_new_height_from_raster(height_constrained_geoloc_raster)
        out_lat, out_lon, out_height = geolocator.process()

        return out_lat, out_lon, out_height

    def do_smoothed_slant_plane_hc_geoloc(self):
        """ Do smoothed slant plane height constrained geolocation """
        LOGGER.info("doing smoothed slant plane height constrained geoloc")

        smoothed_slant_plane_height = smooth_slant_plane(
            self.pixc, 'height',
            self.algorithmic_config['slant_plane_smoothing_filter_shape'],
            self.algorithmic_config['slant_plane_smoothing_good_classes'],
            self.algorithmic_config['slant_plane_smoothing_sus_classes'],
            self.algorithmic_config['wse_class_qual_suspect'],
            self.algorithmic_config['wse_class_qual_degraded'],
            self.algorithmic_config['wse_class_qual_bad'],
            self.algorithmic_config['wse_geo_qual_suspect'],
            self.algorithmic_config['wse_geo_qual_degraded'],
            self.algorithmic_config['wse_geo_qual_bad'],
            self.algorithmic_config['use_bright_land'],
            self.algorithmic_config['specular_not_intersecting_prior_thresh'],
            self.algorithmic_config['slant_plane_smoothing_method'],
            self.algorithmic_config['slant_plane_smoothing_max_chunk_shape'],
            max_worker_processes=self.max_worker_processes)

        geolocator = SWOTRaster.geoloc_raster.GeolocRaster(
            self.pixc, self.algorithmic_config,
            max_worker_processes=self.max_worker_processes)
        geolocator.set_new_height(smoothed_slant_plane_height)
        out_lat, out_lon, out_height = geolocator.process()

        return out_lat, out_lon, out_height

    def get_smoothed_height(self):
        """ Get smoothed raster height for ifgram flattening """
        LOGGER.info("getting smoothed height")

        smoothed_slant_plane_height = smooth_slant_plane(
            self.pixc, 'height',
            self.algorithmic_config['slant_plane_smoothing_filter_shape'],
            self.algorithmic_config['slant_plane_smoothing_good_classes'],
            self.algorithmic_config['slant_plane_smoothing_sus_classes'],
            self.algorithmic_config['wse_class_qual_suspect'],
            self.algorithmic_config['wse_class_qual_degraded'],
            self.algorithmic_config['wse_class_qual_bad'],
            self.algorithmic_config['wse_geo_qual_suspect'],
            self.algorithmic_config['wse_geo_qual_degraded'],
            self.algorithmic_config['wse_geo_qual_bad'],
            self.algorithmic_config['use_bright_land'],
            self.algorithmic_config['specular_not_intersecting_prior_thresh'],
            self.algorithmic_config['slant_plane_smoothing_method'],
            self.algorithmic_config['slant_plane_smoothing_max_chunk_shape'],
            max_worker_processes=self.max_worker_processes)

        return smoothed_slant_plane_height

    def do_raster_processing(self):
        """ Do raster processing """
        LOGGER.info("doing raster processing")

        raster_proc = SWOTRaster.raster_proc.RasterProcessor(
            self.runtime_config['output_sampling_grid_type'],
            self.runtime_config['raster_resolution'],
            self.algorithmic_config['padding'],
            self.algorithmic_config['height_agg_method'],
            self.algorithmic_config['area_agg_method'],
            self.algorithmic_config['sig0_agg_method'],
            self.algorithmic_config['interior_water_classes'],
            self.algorithmic_config['water_edge_classes'],
            self.algorithmic_config['land_edge_classes'],
            self.algorithmic_config['dark_water_classes'],
            self.algorithmic_config['low_coh_water_classes'],
            self.algorithmic_config['use_bright_land'],
            self.algorithmic_config['use_specular_not_intersecting_prior'],
            self.algorithmic_config['specular_not_intersecting_prior_thresh'],
            self.algorithmic_config['use_all_classes_for_wse'],
            self.algorithmic_config['use_all_classes_for_sig0'],
            self.algorithmic_config['wse_geo_qual_suspect'],
            self.algorithmic_config['wse_geo_qual_degraded'],
            self.algorithmic_config['wse_geo_qual_bad'],
            self.algorithmic_config['area_geo_qual_suspect'],
            self.algorithmic_config['area_geo_qual_degraded'],
            self.algorithmic_config['area_geo_qual_bad'],
            self.algorithmic_config['sig0_geo_qual_suspect'],
            self.algorithmic_config['sig0_geo_qual_degraded'],
            self.algorithmic_config['sig0_geo_qual_bad'],
            self.algorithmic_config['wse_class_qual_suspect'],
            self.algorithmic_config['wse_class_qual_degraded'],
            self.algorithmic_config['wse_class_qual_bad'],
            self.algorithmic_config['area_class_qual_suspect'],
            self.algorithmic_config['area_class_qual_degraded'],
            self.algorithmic_config['area_class_qual_bad'],
            self.algorithmic_config['sig0_class_qual_suspect'],
            self.algorithmic_config['sig0_class_qual_degraded'],
            self.algorithmic_config['sig0_class_qual_bad'],
            self.algorithmic_config['sig0_qual_suspect'],
            self.algorithmic_config['sig0_qual_degraded'],
            self.algorithmic_config['sig0_qual_bad'],
            self.algorithmic_config['num_good_sus_pix_thresh_wse'],
            self.algorithmic_config['num_good_sus_pix_thresh_water_area'],
            self.algorithmic_config['num_good_sus_pix_thresh_sig0'],
            self.algorithmic_config['pixc_water_frac_suspect_thresh'],
            self.algorithmic_config['num_wse_pix_suspect_thresh'],
            self.algorithmic_config['num_water_area_pix_suspect_thresh'],
            self.algorithmic_config['num_sig0_pix_suspect_thresh'],
            self.algorithmic_config['near_range_suspect_thresh'],
            self.algorithmic_config['far_range_suspect_thresh'],
            self.algorithmic_config['wse_uncert_suspect_thresh'],
            self.algorithmic_config['water_frac_uncert_suspect_thresh'],
            self.algorithmic_config['sig0_uncert_suspect_thresh'],
            self.algorithmic_config['wse_bad_thresh_min'],
            self.algorithmic_config['wse_bad_thresh_max'],
            self.algorithmic_config['water_frac_bad_thresh_min'],
            self.algorithmic_config['water_frac_bad_thresh_max'],
            self.algorithmic_config['sig0_bad_thresh_min'],
            self.algorithmic_config['sig0_bad_thresh_max'],
            self.algorithmic_config['inner_swath_distance_thresh'],
            self.algorithmic_config['missing_karin_data_time_thresh'],
            utm_zone_adjust=self.runtime_config['utm_zone_adjust'],
            mgrs_band_adjust=self.runtime_config['mgrs_band_adjust'],
            utm_conversion_max_chunk_size=self.algorithmic_config[
                'utm_conversion_max_chunk_size'],
            aggregator_max_chunk_size=self.algorithmic_config[
                'aggregator_max_chunk_size'],
            max_worker_processes=self.max_worker_processes,
            debug_flag=self.algorithmic_config['debug_flag'])

        out_raster = raster_proc.rasterize(
            self.pixc, self.polygon_points, self.data_centroid,
            use_improved_geoloc=self.use_improved_geoloc)
        return out_raster
