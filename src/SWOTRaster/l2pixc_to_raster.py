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

LOGGER = logging.getLogger(__name__)

LOWRES_RASTER_FILENAME = 'lowres_wse_raster.nc'

class L2PixcToRaster(object):
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
        if 'utm_conversion_max_chunk_size' not in self.algorithmic_config:
            self.algorithmic_config['utm_conversion_max_chunk_size'] = \
                DEFAULT_MAX_CHUNK_SIZE
        if 'aggregator_max_chunk_size' not in self.algorithmic_config:
            self.algorithmic_config['aggregator_max_chunk_size'] = \
                DEFAULT_MAX_CHUNK_SIZE
        if 'write_internal_files' not in self.algorithmic_config:
            self.algorithmic_config['write_internal_files'] = False
        if 'debug_flag' not in self.algorithmic_config:
            self.algorithmic_config['debug_flag'] = False
        if 'utm_zone_adjust' not in self.runtime_config:
            self.runtime_config['utm_zone_adjust'] = 0
        if 'mgrs_band_adjust' not in self.runtime_config:
            self.runtime_config['mgrs_band_adjust'] = 0

        # Add default values for low coherence classes
        if 'low_coh_water_classes' not in self.algorithmic_config:
            self.algorithmic_config['low_coh_water_classes'] = []

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
        # "pixcvec" - we want to keep pixcvec improved geoloc as improved geoloc

        if self.algorithmic_config['height_constrained_geoloc_source'] is None \
           or self.algorithmic_config['height_constrained_geoloc_source'].lower() \
           == "none":
            new_height = self.get_smoothed_height()
            self.pixc['pixel_cloud']['improved_height'] = new_height
            self.use_improved_geoloc = False
        elif self.algorithmic_config['height_constrained_geoloc_source'].lower() \
             == "lowres_raster":
            new_lat, new_lon, new_height = self.do_height_constrained_geolocation()
            self.pixc['pixel_cloud']['improved_latitude'] = new_lat
            self.pixc['pixel_cloud']['improved_longitude'] = new_lon
            self.pixc['pixel_cloud']['improved_height'] = new_height
            self.use_improved_geoloc = True
        elif self.algorithmic_config['height_constrained_geoloc_source'].lower() \
             == "pixcvec":
            self.use_improved_geoloc = True
        else:
            raise RasterUsageException(
                'Invalid height_constrained_geoloc_source: {}'.format(
                    self.algorithmic_config['height_constrained_geoloc_source']))

        product = self.do_raster_processing()

        return product

    def do_height_constrained_geolocation(self):
        """ Do raster height constrained geolocation """
        LOGGER.info("doing height constrained geolocation")

        # Normally land edges wouldn't get raster heights, but we are forcing
        # the land edges to be processed as water edges here. Only side effect
        # is in water_area aggregation, which improved geolocation does not use.
        tmp_water_edge_classes = np.concatenate(
            (self.algorithmic_config['water_edge_classes'],
             self.algorithmic_config['land_edge_classes']))
        tmp_land_edge_classes = []

        height_constrained_geoloc_raster_proc = \
            SWOTRaster.raster_proc.RasterProcessor(
                self.runtime_config['output_sampling_grid_type'],
                self.runtime_config['raster_resolution'] \
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
                utm_conversion_max_chunk_size=\
                    self.algorithmic_config['utm_conversion_max_chunk_size'],
                aggregator_max_chunk_size=\
                    self.algorithmic_config['aggregator_max_chunk_size'],
                skip_area=True, skip_sig0=True,
                max_worker_processes=self.max_worker_processes,
                debug_flag=self.algorithmic_config['debug_flag'])

        height_constrained_geoloc_raster = \
            height_constrained_geoloc_raster_proc.rasterize(
                self.pixc, self.polygon_points, self.data_centroid,
                use_improved_geoloc=False)

        if self.algorithmic_config['write_internal_files']:
            if self.scratch_dir is None:
                height_constrained_geoloc_raster.to_ncfile(LOWRES_RASTER_FILENAME)
            else:
                height_constrained_geoloc_raster.to_ncfile(
                    os.path.join(self.scratch_dir, LOWRES_RASTER_FILENAME))

        # if the height-constrained geoloc raster is empty, return fully masked
        # output
        if height_constrained_geoloc_raster.is_empty():
            return (np.ma.masked_all_like(self.pixc['pixel_cloud']['latitude']),
                    np.ma.masked_all_like(self.pixc['pixel_cloud']['longitude']),
                    np.ma.masked_all_like(self.pixc['pixel_cloud']['height']))

        geolocator = SWOTRaster.geoloc_raster.GeolocRaster(
            self.pixc, height_constrained_geoloc_raster, self.algorithmic_config,
            max_worker_processes=self.max_worker_processes)
        out_lat, out_lon, out_height = geolocator.process()

        return out_lat, out_lon, out_height

    def get_smoothed_height(self):
        """ Get smoothed raster height for ifgram flattening """
        LOGGER.info("getting smoothed height")

        height_constrained_geoloc_raster_proc = \
            SWOTRaster.raster_proc.RasterProcessor(
                self.runtime_config['output_sampling_grid_type'],
                self.runtime_config['raster_resolution'] \
                / self.algorithmic_config['lowres_raster_scale_factor'],
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
                utm_conversion_max_chunk_size=\
                    self.algorithmic_config['utm_conversion_max_chunk_size'],
                aggregator_max_chunk_size=\
                    self.algorithmic_config['aggregator_max_chunk_size'],
                skip_area=True, skip_sig0=True,
                max_worker_processes=self.max_worker_processes,
                debug_flag=self.algorithmic_config['debug_flag'])

        height_constrained_geoloc_raster = \
            height_constrained_geoloc_raster_proc.rasterize(
                self.pixc, self.polygon_points, self.data_centroid,
                use_improved_geoloc=False)

        if self.algorithmic_config['write_internal_files']:
            if self.scratch_dir is None:
                height_constrained_geoloc_raster.to_ncfile(LOWRES_RASTER_FILENAME)
            else:
                height_constrained_geoloc_raster.to_ncfile(
                    os.path.join(self.scratch_dir, LOWRES_RASTER_FILENAME))

        # if the height-constrained geoloc raster is empty, return fully masked
        # output
        if height_constrained_geoloc_raster.is_empty():
            return np.ma.masked_all_like(self.pixc['pixel_cloud']['height'])

        geolocator = SWOTRaster.geoloc_raster.GeolocRaster(
            self.pixc, height_constrained_geoloc_raster, self.algorithmic_config,
            max_worker_processes=self.max_worker_processes)
        geolocator.update_heights_from_raster()

        return geolocator.new_height

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
            utm_conversion_max_chunk_size=\
                self.algorithmic_config['utm_conversion_max_chunk_size'],
            aggregator_max_chunk_size=\
                    self.algorithmic_config['aggregator_max_chunk_size'],
            max_worker_processes=self.max_worker_processes,
            debug_flag=self.algorithmic_config['debug_flag'])

        out_raster = raster_proc.rasterize(
            self.pixc, self.polygon_points, self.data_centroid,
            use_improved_geoloc=self.use_improved_geoloc)
        return out_raster
