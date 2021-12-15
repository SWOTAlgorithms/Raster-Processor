'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author (s): Alexander Corben
'''

import logging
import numpy as np
import SWOTRaster.geoloc_raster
import SWOTRaster.raster_proc

LOGGER = logging.getLogger(__name__)

class L2PixcToRaster(object):
    def __init__(self, pixc=None, polygon_points=None,
                 algorithmic_config=None, runtime_config=None):
        self.pixc = pixc
        self.polygon_points = polygon_points
        self.algorithmic_config = algorithmic_config
        self.runtime_config = runtime_config

        # Add default zone adjusts to config
        if 'utm_zone_adjust' not in self.runtime_config:
            self.runtime_config['utm_zone_adjust'] = 0
        if 'mgrs_band_adjust' not in self.runtime_config:
            self.runtime_config['mgrs_band_adjust'] = 0

    def process(self):
        """ Process L2Pixc to Raster """
        LOGGER.info("processing l2pixc to raster")

        # Get height-constrained geolocation as specified in config:
        # "none" - we want to use non-improved geoloc
        # "lowres_raster" - we want to get height constrained geolocation using
        #                   a lowres raster for improved geoloc
        # "pixcvec" - we want to keep pixcvec improved geoloc as improved geoloc

        if self.algorithmic_config['height_constrained_geoloc_source'].lower() \
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
            raise ValueError('Invalid height_constrained_geoloc_source: {}'.format( \
                self.algorithmic_config['height_constrained_geoloc_source']))

        product = self.do_raster_processing()

        return product

    def do_height_constrained_geolocation(self):
        """ Do raster height constrained geolocation """
        LOGGER.info("doing height constrained geolocation")

        # TODO: Handle land edges better in improved geolocation
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
                self.algorithmic_config['interior_water_classes'],
                tmp_water_edge_classes,
                tmp_land_edge_classes,
                self.algorithmic_config['dark_water_classes'],
                self.runtime_config['utm_zone_adjust'],
                self.runtime_config['mgrs_band_adjust'],
                self.algorithmic_config['debug_flag'])

        height_constrained_geoloc_raster = \
            height_constrained_geoloc_raster_proc.rasterize(
                self.pixc, self.polygon_points, use_improved_geoloc=False)

        # if the height-constrained geoloc raster is empty, return fully masked
        # output
        if height_constrained_geoloc_raster.is_empty():
            return (np.ma.masked_all_like(self.pixc['pixel_cloud']['latitude']),
                    np.ma.masked_all_like(self.pixc['pixel_cloud']['longitude']),
                    np.ma.masked_all_like(self.pixc['pixel_cloud']['height']))

        geolocator = SWOTRaster.geoloc_raster.GeolocRaster(
            self.pixc, height_constrained_geoloc_raster, self.algorithmic_config)
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
                self.algorithmic_config['interior_water_classes'],
                self.algorithmic_config['water_edge_classes'],
                self.algorithmic_config['land_edge_classes'],
                self.algorithmic_config['dark_water_classes'],
                self.runtime_config['utm_zone_adjust'],
                self.runtime_config['mgrs_band_adjust'],
                self.algorithmic_config['debug_flag'])

        height_constrained_geoloc_raster = \
            height_constrained_geoloc_raster_proc.rasterize(
                self.pixc, self.polygon_points, use_improved_geoloc=False)

        # if the height-constrained geoloc raster is empty, return fully masked
        # output
        if height_constrained_geoloc_raster.is_empty():
            return np.ma.masked_all_like(self.pixc['pixel_cloud']['height'])

        geolocator = SWOTRaster.geoloc_raster.GeolocRaster(
            self.pixc, height_constrained_geoloc_raster, self.algorithmic_config)
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
            self.algorithmic_config['interior_water_classes'],
            self.algorithmic_config['water_edge_classes'],
            self.algorithmic_config['land_edge_classes'],
            self.algorithmic_config['dark_water_classes'],
            self.runtime_config['utm_zone_adjust'],
            self.runtime_config['mgrs_band_adjust'],
            self.algorithmic_config['debug_flag'])

        out_raster = raster_proc.rasterize(
            self.pixc, self.polygon_points,
            use_improved_geoloc=self.use_improved_geoloc)
        return out_raster
