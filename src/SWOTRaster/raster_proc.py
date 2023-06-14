'''
Copyright (c) 2023-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author (s): Alexander Corben (JPL)
'''

import logging
import numpy as np
import collections.abc
import multiprocessing
import rasterio.features
import rasterio.transform
import SWOTWater.aggregate as ag
import SWOTRaster.products as products
import SWOTRaster.raster_crs as raster_crs
import SWOTRaster.raster_agg as raster_agg

from osgeo import osr
from datetime import datetime
from functools import partial
from itertools import groupby, chain, compress
from more_itertools import chunked
from shapely import affinity
from shapely.geometry import Polygon
from SWOTRaster.errors import RasterUsageException

LOGGER = logging.getLogger(__name__)

class RasterProcessor(object):
    def __init__(self, projection_type, resolution, padding,
                 height_agg_method, area_agg_method, sig0_agg_method,
                 interior_water_classes, water_edge_classes, land_edge_classes,
                 dark_water_classes, use_bright_land, use_all_classes_for_wse,
                 use_all_classes_for_sig0, geo_qual_suspect, geo_qual_degraded,
                 geo_qual_bad, class_qual_suspect, class_qual_degraded,
                 class_qual_bad, sig0_qual_suspect, sig0_qual_degraded,
                 sig0_qual_bad, num_good_sus_pix_thresh_wse,
                 num_good_sus_pix_thresh_water_area,
                 num_good_sus_pix_thresh_sig0, pixc_water_frac_suspect_thresh,
                 num_wse_pix_suspect_thresh, num_water_area_pix_suspect_thresh,
                 num_sig0_pix_suspect_thresh,
                 near_range_suspect_thresh, far_range_suspect_thresh,
                 wse_uncert_suspect_thresh, water_frac_uncert_suspect_thresh,
                 sig0_uncert_suspect_thresh,
                 wse_bad_thresh_min, wse_bad_thresh_max,
                 water_frac_bad_thresh_min, water_frac_bad_thresh_max,
                 sig0_bad_thresh_min, sig0_bad_thresh_max,
                 inner_swath_distance_thresh, missing_karin_data_time_thresh,
                 utm_zone_adjust=0, mgrs_band_adjust=0,
                 utm_conversion_max_chunk_size=products.DEFAULT_MAX_CHUNK_SIZE,
                 aggregator_max_chunk_size=products.DEFAULT_MAX_CHUNK_SIZE,
                 skip_wse=False, skip_area=False, skip_sig0=False,
                 max_worker_processes=0, debug_flag=False):
        self.projection_type = projection_type
        if self.projection_type=='geo':
            # Geodetic resolution is given in arcsec
            self.resolution = np.float(resolution/(60*60))
        elif self.projection_type=='utm':
            self.resolution = np.float(resolution)
            self.utm_zone_adjust = utm_zone_adjust
            self.mgrs_band_adjust = mgrs_band_adjust
            self.utm_conversion_max_chunk_size = utm_conversion_max_chunk_size
        else:
            raise RasterUsageException(
                'Unknown projection type: {}'.format(self.projection_type))

        self.padding = padding
        self.height_agg_method = height_agg_method
        self.area_agg_method = area_agg_method
        self.sig0_agg_method = sig0_agg_method
        self.interior_water_classes = interior_water_classes
        self.water_edge_classes = water_edge_classes
        self.land_edge_classes = land_edge_classes
        self.dark_water_classes = dark_water_classes
        self.use_bright_land = use_bright_land
        self.use_all_classes_for_wse = use_all_classes_for_wse
        self.use_all_classes_for_sig0 = use_all_classes_for_sig0

        self.geo_qual_suspect = geo_qual_suspect
        self.geo_qual_degraded = geo_qual_degraded
        self.geo_qual_bad = geo_qual_bad
        self.class_qual_suspect = class_qual_suspect
        self.class_qual_degraded = class_qual_degraded
        self.class_qual_bad = class_qual_bad
        self.sig0_qual_suspect = sig0_qual_suspect
        self.sig0_qual_degraded = sig0_qual_degraded
        self.sig0_qual_bad = sig0_qual_bad

        self.num_good_sus_pix_thresh_wse = num_good_sus_pix_thresh_wse
        self.num_good_sus_pix_thresh_water_area = num_good_sus_pix_thresh_water_area
        self.num_good_sus_pix_thresh_sig0 = num_good_sus_pix_thresh_sig0

        self.pixc_water_frac_suspect_thresh = pixc_water_frac_suspect_thresh
        self.num_wse_pix_suspect_thresh = num_wse_pix_suspect_thresh
        self.num_water_area_pix_suspect_thresh = num_water_area_pix_suspect_thresh
        self.num_sig0_pix_suspect_thresh = num_sig0_pix_suspect_thresh
        self.near_range_suspect_thresh = near_range_suspect_thresh
        self.far_range_suspect_thresh = far_range_suspect_thresh
        self.wse_uncert_suspect_thresh = wse_uncert_suspect_thresh
        self.water_frac_uncert_suspect_thresh = water_frac_uncert_suspect_thresh
        self.sig0_uncert_suspect_thresh = sig0_uncert_suspect_thresh

        self.wse_bad_thresh_min = wse_bad_thresh_min
        self.wse_bad_thresh_max = wse_bad_thresh_max
        self.water_frac_bad_thresh_min = water_frac_bad_thresh_min
        self.water_frac_bad_thresh_max = water_frac_bad_thresh_max
        self.sig0_bad_thresh_min = sig0_bad_thresh_min
        self.sig0_bad_thresh_max = sig0_bad_thresh_max

        self.inner_swath_distance_thresh = inner_swath_distance_thresh
        self.missing_karin_data_time_thresh = missing_karin_data_time_thresh

        self.skip_wse = skip_wse
        self.skip_area = skip_area
        self.skip_sig0 = skip_sig0

        self.aggregator_max_chunk_size = aggregator_max_chunk_size
        self.max_worker_processes = max_worker_processes
        self.debug_flag = debug_flag

    def rasterize(self, pixc, polygon_points=None, use_improved_geoloc=True):
        """ Rasterize pixc to raster """
        LOGGER.info("rasterizing")
        self.input_crs = raster_crs.wgs84_crs()
        self.cycle_number = pixc.scene_cycle_number
        self.pass_number = pixc.scene_pass_number
        self.tile_numbers = pixc.tile_numbers
        self.tile_names = pixc.tile_names
        self.tile_polarizations = pixc.tile_polarizations
        self.scene_number = pixc.scene_number
        self.time_granule_start = pixc.time_granule_start
        self.time_granule_end = pixc.time_granule_end
        self.time_coverage_start = pixc.time_coverage_start
        self.time_coverage_end = pixc.time_coverage_end
        self.geospatial_lon_min = pixc.geospatial_lon_min
        self.geospatial_lon_max = pixc.geospatial_lon_max
        self.geospatial_lat_min = pixc.geospatial_lat_min
        self.geospatial_lat_max = pixc.geospatial_lat_max
        self.left_first_longitude = pixc.left_first_longitude
        self.left_first_latitude = pixc.left_first_latitude
        self.left_last_longitude = pixc.left_last_longitude
        self.left_last_latitude = pixc.left_last_latitude
        self.right_first_longitude = pixc.right_first_longitude
        self.right_first_latitude = pixc.right_first_latitude
        self.right_last_longitude = pixc.right_last_longitude
        self.right_last_latitude = pixc.right_last_latitude

        if polygon_points is None:
            LOGGER.info("creating projection from swath corner points")
            swath_corners = \
                [(pixc.left_first_latitude, pixc.left_first_longitude),
                 (pixc.right_first_latitude, pixc.right_first_longitude),
                 (pixc.right_last_latitude, pixc.right_last_longitude),
                 (pixc.left_last_latitude, pixc.left_last_longitude)]
            self.create_projection_from_polygon_points(swath_corners)
        else:
            LOGGER.info("creating projection from polygon points")
            self.create_projection_from_polygon_points(polygon_points)

        # Return empty product if pixc is empty
        empty_product = self.build_product(populate_values=False)
        if len(pixc['pixel_cloud']['height'])==0:
            LOGGER.warning('Empty Pixel Cloud: returning empty raster')
            return empty_product

        # Get pixc classification masks
        water_classes = np.concatenate((self.interior_water_classes,
                                        self.water_edge_classes,
                                        self.dark_water_classes))
        all_classes = np.concatenate((water_classes, self.land_edge_classes))

        water_classes_mask = pixc.get_mask(water_classes, use_improved_geoloc)
        all_classes_mask = pixc.get_mask(all_classes, use_improved_geoloc)

        bright_land_pixc_flag = pixc['pixel_cloud']['bright_land_flag']
        if not self.use_bright_land:
            water_classes_mask = np.logical_and(
                water_classes_mask, np.logical_not(bright_land_pixc_flag))
            all_classes_mask = np.logical_and(
                all_classes_mask, np.logical_not(bright_land_pixc_flag))

        # Get pixc summary quality flags
        LOGGER.info("getting pixc summary quality flags")
        geo_qual_pixc_flag = pixc.get_summary_qual_flag(
            'geolocation_qual', self.geo_qual_suspect,
            self.geo_qual_degraded,self.geo_qual_bad)
        class_qual_pixc_flag = pixc.get_summary_qual_flag(
            'classification_qual', self.class_qual_suspect,
            self.class_qual_degraded, self.class_qual_bad)
        sig0_qual_pixc_flag = pixc.get_summary_qual_flag(
            'sig0_qual', self.sig0_qual_suspect,
            self.sig0_qual_degraded, self.sig0_qual_bad)

        # Get raster mapping
        if self.projection_type=='utm':
            self.proj_mapping = empty_product.get_raster_mapping(
                pixc, all_classes_mask, use_improved_geoloc,
                self.utm_conversion_max_chunk_size)
        else:
            self.proj_mapping = empty_product.get_raster_mapping(
                pixc, all_classes_mask, use_improved_geoloc)

        # Get rasterization masks
        LOGGER.info('getting rasterization masks for wse/water area/sig0')
        wse_classes_mask = water_classes_mask
        if self.use_all_classes_for_wse:
            wse_classes_mask = all_classes_mask

        wse_pixc_mask, wse_raster_mask = self.get_rasterization_masks(
            wse_classes_mask, (geo_qual_pixc_flag, class_qual_pixc_flag),
            self.num_good_sus_pix_thresh_wse)

        water_area_pixc_mask, water_area_raster_mask = \
            self.get_rasterization_masks(
                all_classes_mask,(geo_qual_pixc_flag, class_qual_pixc_flag),
                self.num_good_sus_pix_thresh_water_area)

        sig0_classes_mask = water_classes_mask
        if self.use_all_classes_for_wse:
            sig0_classes_mask = all_classes_mask

        sig0_pixc_mask, sig0_raster_mask = self.get_rasterization_masks(
            sig0_classes_mask,
            (geo_qual_pixc_flag, class_qual_pixc_flag, sig0_qual_pixc_flag),
            self.num_good_sus_pix_thresh_sig0)

        all_pixc_mask = np.logical_or.reduce((
            wse_pixc_mask, water_area_pixc_mask, sig0_pixc_mask))
        all_raster_mask = np.logical_or.reduce((
            wse_raster_mask, water_area_raster_mask, sig0_raster_mask))

        # Aggregate variables
        LOGGER.info('aggregating cross track and incidence angle')
        self.cross_track, self.inc, self.n_other_pix = self.call_aggregator(
            raster_agg.aggregate_cross_track_and_incidence_angle,
            pixc['pixel_cloud']['cross_track'], pixc['pixel_cloud']['inc'],
            all_pixc_mask, mask=all_raster_mask)

        LOGGER.info('aggregating illumination time')
        self.illumination_time, self.illumination_time_tai = self.call_aggregator(
            raster_agg.aggregate_illumination_time,
            pixc['pixel_cloud']['illumination_time'],
            pixc['pixel_cloud']['illumination_time_tai'],
            all_pixc_mask, mask=all_raster_mask)

        LOGGER.info('aggregating latitude and longitude')
        x_mesh = np.tile(self.x_vec, (self.size_y, 1))
        y_mesh = np.tile(self.y_vec, (self.size_x, 1)).T
        if self.projection_type=='geo':
            self.latitude = np.ma.masked_array(
                y_mesh, mask=np.logical_not(all_raster_mask))
            self.longitude = np.ma.masked_array(
                x_mesh, mask=np.logical_not(all_raster_mask))
        else:
            self.latitude, self.longitude = self.call_aggregator(
                partial(raster_agg.aggregate_px_latlon,
                        crs_wkt=self.output_crs.ExportToWkt()),
                x_mesh, y_mesh, all_pixc_mask, mask=all_raster_mask)

        if not self.skip_wse:
            LOGGER.info('aggregating wse corrections')
            (self.height_cor_xover, self.geoid, self.solid_earth_tide,
             self.load_tide_fes, self.load_tide_got, self.pole_tide,
             self.model_dry_tropo_cor, self.model_wet_tropo_cor,
             self.iono_cor_gim_ka) = self.call_aggregator(
                 partial(raster_agg.aggregate_wse_corrections,
                         height_agg_method=self.height_agg_method),
                 pixc['pixel_cloud']['height_cor_xover'],
                 pixc['pixel_cloud']['geoid'],
                 pixc['pixel_cloud']['solid_earth_tide'],
                 pixc['pixel_cloud']['load_tide_fes'],
                 pixc['pixel_cloud']['load_tide_got'],
                 pixc['pixel_cloud']['pole_tide'],
                 pixc['pixel_cloud']['model_dry_tropo_cor'],
                 pixc['pixel_cloud']['model_wet_tropo_cor'],
                 pixc['pixel_cloud']['iono_cor_gim_ka'],
                 pixc['pixel_cloud']['dheight_dphase'],
                 pixc['pixel_cloud']['phase_noise_std'],
                 wse_pixc_mask, mask=wse_raster_mask)

            LOGGER.info('flattening interferogram')
            tvp_plus_y_antenna_xyz = (pixc['tvp']['plus_y_antenna_x'],
                                      pixc['tvp']['plus_y_antenna_y'],
                                      pixc['tvp']['plus_y_antenna_z'])
            tvp_minus_y_antenna_xyz = (pixc['tvp']['minus_y_antenna_x'],
                                       pixc['tvp']['minus_y_antenna_y'],
                                       pixc['tvp']['minus_y_antenna_z'])

            if use_improved_geoloc:
                # Flatten ifgram with improved geoloc and height
                target_xyz = raster_crs.llh2xyz((
                    np.deg2rad(pixc['pixel_cloud']['improved_latitude']),
                    np.deg2rad(pixc['pixel_cloud']['improved_longitude']),
                    pixc['pixel_cloud']['improved_height']))
            else:
                # Flatten ifgram with original geoloc and improved height
                target_xyz = raster_crs.llh2xyz((
                    np.deg2rad(pixc['pixel_cloud']['latitude']),
                    np.deg2rad(pixc['pixel_cloud']['longitude']),
                    pixc['pixel_cloud']['improved_height']))

            flat_ifgram = ag.flatten_interferogram(
                pixc['pixel_cloud']['interferogram'],
                tvp_plus_y_antenna_xyz, tvp_minus_y_antenna_xyz, target_xyz,
                ag.get_sensor_index(pixc), pixc.wavelength)

            LOGGER.info('aggregating height')
            height, self.wse_u = self.call_aggregator(
                partial(raster_agg.aggregate_height,
                        looks_to_efflooks=pixc['pixel_cloud'].looks_to_efflooks,
                        height_agg_method=self.height_agg_method),
                pixc['pixel_cloud']['height'],
                pixc['pixel_cloud']['eff_num_rare_looks'],
                pixc['pixel_cloud']['eff_num_medium_looks'],
                pixc['pixel_cloud']['power_plus_y'],
                pixc['pixel_cloud']['power_minus_y'],
                pixc['pixel_cloud']['dheight_dphase'],
                pixc['pixel_cloud']['dlatitude_dphase'],
                pixc['pixel_cloud']['dlongitude_dphase'],
                pixc['pixel_cloud']['phase_noise_std'], flat_ifgram,
                wse_pixc_mask, mask=wse_raster_mask)

            LOGGER.info('applying wse corrections')
            self.wse = raster_agg.apply_wse_corrections(
                height, self.geoid, self.solid_earth_tide,
                self.load_tide_fes, self.pole_tide)

            LOGGER.info('aggregating wse qual')
            (self.wse_qual, self.wse_qual_bitwise,
             self.n_wse_pix) = self.call_aggregator(
                 partial(raster_agg.aggregate_wse_qual,
                         wse_uncert_suspect_thresh=self.wse_uncert_suspect_thresh,
                         num_wse_pix_suspect_thresh=self.num_wse_pix_suspect_thresh,
                         near_range_suspect_thresh=self.near_range_suspect_thresh,
                         far_range_suspect_thresh=self.far_range_suspect_thresh,
                         wse_bad_thresh_min=self.wse_bad_thresh_min,
                         wse_bad_thresh_max=self.wse_bad_thresh_max),
                 self.wse, self.wse_u, self.cross_track, class_qual_pixc_flag,
                 geo_qual_pixc_flag, bright_land_pixc_flag,
                 wse_pixc_mask, mask=wse_raster_mask)

            LOGGER.info('aggregating layover impact')
            self.layover_impact = self.call_aggregator(
                partial(raster_agg.aggregate_layover_impact,
                        height_agg_method=self.height_agg_method),
                pixc['pixel_cloud']['layover_impact'],
                pixc['pixel_cloud']['dheight_dphase'],
                pixc['pixel_cloud']['phase_noise_std'],
                wse_pixc_mask, mask=wse_raster_mask)

        if not self.skip_area:
            LOGGER.info('aggregating water area')
            (self.water_area, self.water_area_u, self.water_frac,
             self.water_frac_u) = self.call_aggregator(
                 partial(raster_agg.aggregate_water_area,
                         projection_type=self.projection_type,
                         resolution=self.resolution,
                         interior_water_klasses=self.interior_water_classes,
                         water_edge_klasses=self.water_edge_classes,
                         land_edge_klasses=self.land_edge_classes,
                         dark_water_klasses=self.dark_water_classes,
                         area_agg_method=self.area_agg_method),
                 pixc['pixel_cloud']['pixel_area'],
                 pixc['pixel_cloud']['water_frac'],
                 pixc['pixel_cloud']['water_frac_uncert'],
                 pixc['pixel_cloud']['darea_dheight'],
                 pixc['pixel_cloud']['false_detection_rate'],
                 pixc['pixel_cloud']['missed_detection_rate'],
                 pixc['pixel_cloud']['classification'], self.latitude,
                 water_area_pixc_mask, mask=water_area_raster_mask)

            LOGGER.info('aggregating water area qual')
            (self.water_area_qual, self.water_area_qual_bitwise,
             self.n_water_area_pix) = self.call_aggregator(
                 partial(raster_agg.aggregate_water_area_qual,
                         pixc_water_frac_suspect_thresh=\
                             self.pixc_water_frac_suspect_thresh,
                         water_frac_uncert_suspect_thresh=\
                             self.water_frac_uncert_suspect_thresh,
                         num_water_area_pix_suspect_thresh=\
                             self.num_water_area_pix_suspect_thresh,
                         near_range_suspect_thresh=\
                             self.near_range_suspect_thresh,
                         far_range_suspect_thresh=\
                             self.far_range_suspect_thresh,
                         water_frac_bad_thresh_min=\
                             self.water_frac_bad_thresh_min,
                         water_frac_bad_thresh_max=\
                             self.water_frac_bad_thresh_max),
                 self.water_frac, self.water_frac_u, self.cross_track,
                 class_qual_pixc_flag, geo_qual_pixc_flag, bright_land_pixc_flag,
                 pixc['pixel_cloud']['water_frac'],
                 water_area_pixc_mask, mask=water_area_raster_mask)

            LOGGER.info('aggregating dark water fraction')
            self.dark_frac = self.call_aggregator(
                partial(raster_agg.aggregate_dark_frac,
                        interior_water_klasses=self.interior_water_classes,
                        water_edge_klasses=self.water_edge_classes,
                        land_edge_klasses=self.land_edge_classes,
                        dark_water_klasses=self.dark_water_classes,
                        area_agg_method=self.area_agg_method),
                pixc['pixel_cloud']['classification'],
                pixc['pixel_cloud']['pixel_area'],
                pixc['pixel_cloud']['water_frac'],
                water_area_pixc_mask, mask=water_area_raster_mask)

        if not self.skip_sig0:
            LOGGER.info('aggregating sigma0 corrections')
            self.sig0_cor_atmos_model = self.call_aggregator(
                raster_agg.aggregate_sig0_corrections,
                pixc['pixel_cloud']['sig0_cor_atmos_model'],
                sig0_pixc_mask, mask=sig0_raster_mask)

            LOGGER.info('aggregating sigma0')
            self.sig0, self.sig0_u = self.call_aggregator(
                partial(raster_agg.aggregate_sig0,
                        sig0_agg_method=self.sig0_agg_method),
                pixc['pixel_cloud']['sig0'], pixc['pixel_cloud']['sig0_uncert'],
                sig0_pixc_mask, mask=sig0_raster_mask)

            LOGGER.info('aggregating sigma0 qual')
            (self.sig0_qual, self.sig0_qual_bitwise,
             self.n_sig0_pix) = self.call_aggregator(
                 partial(raster_agg.aggregate_sig0_qual,
                         sig0_uncert_suspect_thresh=\
                             self.sig0_uncert_suspect_thresh,
                         num_sig0_pix_suspect_thresh=\
                            self.num_sig0_pix_suspect_thresh,
                         near_range_suspect_thresh=\
                             self.near_range_suspect_thresh,
                         far_range_suspect_thresh=self.far_range_suspect_thresh,
                         sig0_bad_thresh_min=self.sig0_bad_thresh_min,
                         sig0_bad_thresh_max=self.sig0_bad_thresh_max),
                 self.sig0, self.sig0_u, self.cross_track, sig0_qual_pixc_flag,
                 class_qual_pixc_flag, geo_qual_pixc_flag, bright_land_pixc_flag,
                 sig0_pixc_mask, mask=sig0_raster_mask)

        if self.debug_flag:
            LOGGER.info('aggregating classification')
            self.classification = self.call_aggregator(
                raster_agg.aggregate_classification,
                pixc['pixel_cloud']['classification'],
                all_pixc_mask, mask=all_raster_mask)

        LOGGER.info('aggregating ice flags')
        self.ice_clim_flag = self.call_aggregator(
            raster_agg.aggregate_ice_flag, pixc['pixel_cloud']['ice_clim_flag'],
            all_pixc_mask, mask=all_raster_mask)

        self.ice_dyn_flag = self.call_aggregator(
            raster_agg.aggregate_ice_flag, pixc['pixel_cloud']['ice_dyn_flag'],
            all_pixc_mask, mask=all_raster_mask)

        LOGGER.info("flagging missing karin data")
        self.flag_missing_karin_data(pixc)

        LOGGER.info("flagging inner swath")
        self.flag_inner_swath(pixc)

        # Set the time coverage start and end based on illumination time
        if np.all(self.illumination_time.mask):
            self.time_coverage_start = products.EMPTY_DATETIME
            self.time_coverage_end = products.EMPTY_DATETIME
        else:
            start_illumination_time = np.nanmin(self.illumination_time)
            end_illumination_time = np.nanmax(self.illumination_time)
            start_time = datetime.utcfromtimestamp(
                (products.SWOT_EPOCH - products.UNIX_EPOCH).total_seconds()
                + start_illumination_time)
            stop_time = datetime.utcfromtimestamp(
                (products.SWOT_EPOCH - products.UNIX_EPOCH).total_seconds()
                + end_illumination_time)
            self.time_coverage_start = start_time.strftime(
                products.DATETIME_FORMAT_STR)
            self.time_coverage_end = stop_time.strftime(
                products.DATETIME_FORMAT_STR)

        # Set tai_utc_difference
        min_illumination_time_index = np.unravel_index(
            np.nanargmin(self.illumination_time), self.illumination_time.shape)
        self.tai_utc_difference = \
            self.illumination_time_tai[min_illumination_time_index] \
            - self.illumination_time[min_illumination_time_index]

        # Set leap second
        if pixc.leap_second==products.EMPTY_LEAPSEC:
            self.leap_second = products.EMPTY_LEAPSEC
        else:
            leap_second = datetime.strptime(
                pixc.leap_second, products.LEAPSEC_FORMAT_STR)
            if leap_second < start_time or leap_second > end_time:
                self.leap_second = products.EMPTY_LEAPSEC
            else:
                self.leap_second = leap_second.strftime(
                    products.LEAPSEC_FORMAT_STR)

        LOGGER.info("building product")
        return self.build_product(polygon_points=polygon_points)

    def create_projection_from_polygon_points(self, polygon_points):
        """ Create projection given points defining a bounding polygon """
        if self.projection_type=='geo':
            # Set output crs
            self.output_crs = raster_crs.wgs84_crs()

            # Handle longitude wrap
            poly_edge_x = raster_crs.shift_wrapped_longitude(
                [point[1] for point in polygon_points])
            poly_edge_y = [point[0] for point in polygon_points]
            proj_center_x = 0
            proj_center_y = 0

        elif self.projection_type=='utm':
            # Set output crs
            self.output_crs, utm_zone, mgrs_band = \
                raster_crs.utm_crs_from_points(
                    polygon_points, self.utm_zone_adjust, self.mgrs_band_adjust)
            self.utm_zone = np.short(utm_zone)
            self.utm_hemisphere = raster_crs.hemisphere_from_mgrs_band(mgrs_band)
            self.mgrs_band = mgrs_band

            # Transform to UTM
            transf = osr.CoordinateTransformation(self.input_crs, self.output_crs)
            polygon_points = [(transf.TransformPoint(point[0], point[1])[:2])
                              for point in polygon_points]
            poly_edge_y = [point[1] for point in polygon_points]
            poly_edge_x = [point[0] for point in polygon_points]
            proj_center_x = self.output_crs.GetProjParm('false_easting')
            proj_center_y = self.output_crs.GetProjParm('false_northing')

        else:
            raise RasterUsageException(
                'Unknown projection type: {}'.format(self.projection_type))

        # Get the coordinate limits
        x_min = np.min(poly_edge_x)
        x_max = np.max(poly_edge_x)
        y_min = np.min(poly_edge_y)
        y_max = np.max(poly_edge_y)

        # Round limits to the nearest bin (centered at proj center with pad)
        x_min = int((round((x_min - proj_center_x) / self.resolution))
                    - self.padding) * self.resolution + proj_center_x
        x_max = int((round((x_max - proj_center_x) / self.resolution)) \
                    + self.padding) * self.resolution + proj_center_x
        y_min = int((round((y_min - proj_center_y) / self.resolution)) \
                    - self.padding) * self.resolution + proj_center_y
        y_max = int((round((y_max - proj_center_y) / self.resolution)) \
                    + self.padding) * self.resolution + proj_center_y

        self.size_x = int(round((x_max - x_min) / self.resolution)) + 1
        self.size_y = int(round((y_max - y_min) / self.resolution)) + 1

        # Wrap longitude to between -180 to 180 degrees longitude if lat/lon
        if self.projection_type=='geo':
            self.x_min = raster_crs.lon_360to180(x_min)
            self.x_max = raster_crs.lon_360to180(x_max)
            self.x_vec = raster_crs.lon_360to180(
                np.linspace(x_min, x_max, self.size_x))
        else:
            self.x_min = x_min
            self.x_max = x_max
            self.x_vec = np.linspace(x_min, x_max, self.size_x)

        self.y_min = y_min
        self.y_max = y_max
        self.y_vec = np.linspace(y_min, y_max, self.size_y)

        LOGGER.info({'proj': self.output_crs.ExportToWkt(),
                     'res': self.resolution,
                     'x_min': self.x_min,
                     'x_max': self.x_max,
                     'size_x': self.size_x,
                     'y_min': self.y_min,
                     'y_max': self.y_max,
                     'size_y': self.size_y})

    def get_rasterization_masks(self, valid_classes_mask,
                                pixc_summary_qual_flags, num_good_sus_pix_thresh):
        """ Get masks of pixels to rasterize """
        common_qual_flag = np.maximum.reduce((pixc_summary_qual_flags))
        good_qual_mask = [x==products.QUAL_IND_GOOD for x in common_qual_flag]
        sus_qual_mask = [x==products.QUAL_IND_SUSPECT for x in common_qual_flag]
        deg_qual_mask = [x==products.QUAL_IND_DEGRADED for x in common_qual_flag]

        good_mask = np.logical_and(valid_classes_mask, good_qual_mask)
        suspect_mask = np.logical_and(valid_classes_mask, sus_qual_mask)
        degraded_mask = np.logical_and(valid_classes_mask, deg_qual_mask)

        good_sus_mask = np.logical_or(good_mask, suspect_mask)
        good_sus_degraded_mask = np.logical_or(good_sus_mask, degraded_mask)

        pixc_mask = np.ma.zeros(good_mask.shape, dtype=bool)
        raster_mask = np.ma.zeros((self.size_y, self.size_x), dtype=bool)

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = good_sus_mask[self.proj_mapping[i][j]]
                if np.sum(mask) < num_good_sus_pix_thresh:
                    mask = good_sus_degraded_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    mapping_idxs = np.array(self.proj_mapping[i][j])[mask]
                    pixc_mask[mapping_idxs] = True
                    raster_mask[i][j] = True

        return pixc_mask, raster_mask

    def call_aggregator(self, agg_fn, *args, mask=None):
        """ Calls aggregator function with iterable arguments """
        def get_agg_arg(arg, mask, chunk_size=None):
            """ Get generator of an aggregator input argument, with chunking """
            if arg.shape==(self.size_y, self.size_x):
                if chunk_size is None:
                    return (el for el in arg[mask])
                else:
                    return ([el for el in chunk]
                            for chunk in chunked(arg[mask], chunk_size))
            else:
                compressed_mapping = compress(
                    chain.from_iterable(self.proj_mapping), mask.flatten())
                if chunk_size is None:
                    return (arg[inds] for inds in compressed_mapping)
                else:
                    return ([arg[inds] for inds in chunk]
                            for chunk in chunked(compressed_mapping, chunk_size))

        def get_agg_output(result, mask, fill_value=np.nan):
            """ Get aggregator output on raster grid, with fill_value """
            out = fill_value*np.ma.ones((self.size_y, self.size_x),
                                        dtype=type(fill_value))
            out[mask] = result
            return np.ma.masked_invalid(out)

        # Call aggregator with multiprocessing if commanded
        if self.max_worker_processes > 1:
            chunk_size = int(min(
                np.ceil(np.sum(mask) / (self.max_worker_processes*4)),
                self.aggregator_max_chunk_size))

            _agg_fn = partial(raster_agg.fn_map, agg_fn)
            with multiprocessing.get_context('spawn').Pool(
                    processes=self.max_worker_processes) as pool:
                result_chunks = pool.imap(_agg_fn,
                    zip(*(get_agg_arg(arg, mask, chunk_size) for arg in args)))
                results = list(chain.from_iterable(result_chunks))
        else:
            _agg_fn = partial(raster_agg.fn_star, agg_fn)
            results = [_agg_fn(arglist) for arglist
                       in zip(*(get_agg_arg(arg, mask) for arg in args))]

        # Call aggregator with empty inputs to get fill values
        empty_results = agg_fn(*[[]]*len(args))
        if not results:
            results = [empty_results]

        # Reshape results
        if isinstance(empty_results, collections.abc.Iterable):
            return tuple(get_agg_output(result, mask, empty_result)
                         for result, empty_result in zip(zip(*results),
                                                         empty_results))
        else:
            return get_agg_output(results, mask, empty_results)

    def flag_missing_karin_data(self, pixc):
        """ Flag missing karin data"""
        pixc_line_qual_meanings = \
            pixc['pixel_cloud'].VARIABLES['pixc_line_qual']['flag_meanings'].split()
        pixc_line_qual_masks = \
            pixc['pixel_cloud'].VARIABLES['pixc_line_qual']['flag_masks']
        pixc_line_qual_ind_large_karin_gap = pixc_line_qual_masks[
            pixc_line_qual_meanings.index('large_karin_gap')]

        # Define grouping function
        def _group_by_diff(data, diff, key=None):
            if key is None: key = data
            split_idxs = [i+1 for x, y, i in zip(key[:-1], key[1:], range(len(key)))
                          if abs(y-x) > diff]
            split_idxs = [0] + split_idxs + [len(key)]
            groups = [data[i:j] for i, j in zip(split_idxs[:-1], split_idxs[1:])]
            idxs = [np.arange(i, j) for i, j in zip(split_idxs[:-1], split_idxs[1:])]
            return zip(groups, idxs)

        # Create polygons for areas that don't have missing data
        # Handle the different sides separately
        extant_data_polygons_points = []
        for swath_side in ['L', 'R']:
            tvp_side_mask = pixc['tvp']['swath_side']==swath_side
            pixc_tvp_index = pixc['pixel_cloud']['pixc_line_to_tvp'].astype(int)
            pixc_side_mask = tvp_side_mask[pixc_tvp_index]
            pixc_line_qual = pixc['pixel_cloud']['pixc_line_qual'][pixc_side_mask]
            pixc_tvp_index = pixc_tvp_index[pixc_side_mask]

            tvp_time = pixc['tvp']['time']
            tvp_velocity_heading = pixc['tvp']['velocity_heading']
            tvp_xyz = np.row_stack((
                pixc['tvp']['x'], pixc['tvp']['y'], pixc['tvp']['z']))

            pixc_extant_data_mask = \
                (pixc_line_qual & pixc_line_qual_ind_large_karin_gap)==0

            for k, g in groupby(np.arange(len(pixc_extant_data_mask)),
                                lambda x: pixc_extant_data_mask[x]):
                if k:
                    group_idxs = pixc_tvp_index[list(g)]
                    group_times = tvp_time[group_idxs]
                    for idxs, _ in _group_by_diff(
                            group_idxs, self.missing_karin_data_time_thresh,
                            key=group_times):
                        group_tvp_xyz = tvp_xyz[:,idxs]
                        group_tvp_velocity_heading = tvp_velocity_heading[idxs]
                        extant_data_polygons_points.append(
                            self.get_swath_polygon_points_from_tvp(
                                group_tvp_xyz,
                                group_tvp_velocity_heading,
                                swath_side,
                                products.POLYGON_EXTENT_DIST))

        polys = []
        for this_polygon_points in extant_data_polygons_points:
            # If polygon points are in geodetic coordinates, swap lat/lon
            if self.projection_type=='geo':
                this_poly = Polygon(
                    [[point[1], point[0]] for point in this_polygon_points])
            else:
                this_poly = Polygon(this_polygon_points)
            polys.append(this_poly)

        # Handle longitude wrap
        x_max = self.x_max
        if self.projection_type=='geo' and self.x_min > x_max:
            x_max = x_max + 360
            shifted_polys = []
            for poly in polys:
                shifted_poly = raster_crs.shift_wrapped_longitude_polygon(poly)
                (min_x, _, max_x, _) = shifted_poly.bounds
                if max_x < self.x_min:
                    shifted_poly = affinity.translate(shifted_poly, xoff=360)
                shifted_polys.append(shifted_poly)
            polys = shifted_polys

        if len(polys) > 0:
            raster_transform = rasterio.transform.from_bounds(
                self.x_min, self.y_min, x_max, self.y_max, self.size_x,
                self.size_y)
            mask = np.flipud(rasterio.features.geometry_mask(
                polys, out_shape=(self.size_y, self.size_x),
                transform=raster_transform, all_touched=True))
        else:
            mask = np.ones((self.size_y, self.size_x), dtype=bool)

        # Mask the datasets and flag
        if not self.skip_wse:
            wse_mask = np.logical_and(self.wse.mask, mask)
            self.wse_qual_bitwise[wse_mask] += \
                products.QUAL_IND_MISSING_KARIN_DATA
        if not self.skip_area:
            water_area_mask = np.logical_and(self.water_area.mask, mask)
            self.water_area_qual_bitwise[water_area_mask] += \
                products.QUAL_IND_MISSING_KARIN_DATA
        if not self.skip_sig0:
            sig0_mask = np.logical_and(self.sig0.mask, mask)
            self.sig0_qual_bitwise[sig0_mask] += \
                products.QUAL_IND_MISSING_KARIN_DATA

    def flag_inner_swath(self, pixc):
        """ Flag inner swath"""
        # Create polygon for inner swath area (full swath)
        swath_side = 'F'
        tvp_velocity_heading = pixc['tvp']['velocity_heading']
        tvp_xyz = np.row_stack((
            pixc['tvp']['x'], pixc['tvp']['y'], pixc['tvp']['z']))

        inner_swath_polygon_points = self.get_swath_polygon_points_from_tvp(
            tvp_xyz, tvp_velocity_heading, swath_side,
            self.inner_swath_distance_thresh,
            products.POLYGON_EXTENT_DIST,
            products.POLYGON_EXTENT_DIST)

        # If polygon points are in geodetic coordinates, swap lat/lon
        if self.projection_type=='geo':
            poly = Polygon(
                [[point[1], point[0]] for point in inner_swath_polygon_points])
        else:
            poly = Polygon(inner_swath_polygon_points)

        # Handle longitude wrap
        x_max = self.x_max
        if self.projection_type=='geo' and self.x_min > x_max:
            x_max = x_max + 360
            poly = raster_crs.shift_wrapped_longitude_polygon(poly)
            (min_x, _, max_x, _) = poly.bounds
            if max_x < self.x_min:
                poly = affinity.translate(poly, xoff=360)

        raster_transform = rasterio.transform.from_bounds(
            self.x_min, self.y_min, x_max, self.y_max, self.size_x,
            self.size_y)
        mask = np.flipud(rasterio.features.geometry_mask(
            [poly], out_shape=(self.size_y, self.size_x),
            transform=raster_transform, all_touched=True, invert=True))

        # Mask the datasets and flag
        if not self.skip_wse:
            wse_mask = np.logical_and(self.wse.mask, mask)
            self.wse_qual_bitwise[wse_mask] += \
                products.QUAL_IND_INNER_SWATH
        if not self.skip_area:
            water_area_mask = np.logical_and(self.water_area.mask, mask)
            self.water_area_qual_bitwise[water_area_mask] += \
                products.QUAL_IND_INNER_SWATH
        if not self.skip_sig0:
            sig0_mask = np.logical_and(self.sig0.mask, mask)
            self.sig0_qual_bitwise[sig0_mask] += \
                products.QUAL_IND_INNER_SWATH

    def get_swath_polygon_points_from_tvp(
            self, sc_xyz, sc_velocity_heading,
            swath_side, crosstrack_dist, alongtrack_start_buffer_dist=None,
            alongtrack_end_buffer_dist=None, downsample_rate=None):
        """ Get swath polygon points from tvp points """
        # If there is only one line, repeat it to make a polygon
        if len(sc_velocity_heading)==1:
            sc_xyz = np.column_stack((sc_xyz, sc_xyz))
            sc_velocity_heading = np.append(
                sc_velocity_heading, sc_velocity_heading)

        transf = osr.CoordinateTransformation(self.input_crs, self.output_crs)

        if downsample_rate is not None:
            idx_vec = np.arange(0, sc_xyz.shape[1], downsample_rate)
            if idx_vec[-1] != sc_xyz.shape[1]-1:
                idx_vec = np.append(idx_vec, sc_xyz.shape[1]-1)
        else:
            idx_vec = np.arange(sc_xyz.shape[1])

        polygon = []
        for polygon_side in [0, 1]:
            reverse_side = polygon_side*2 - 1
            if swath_side=='L':
                this_side_crosstrack_angle = np.deg2rad(
                    np.mod(sc_velocity_heading-90, 360))
                this_side_crosstrack_dist = polygon_side*crosstrack_dist
            elif swath_side=='R':
                this_side_crosstrack_angle = np.deg2rad(
                    np.mod(sc_velocity_heading+90, 360))
                this_side_crosstrack_dist = polygon_side*crosstrack_dist
            elif swath_side=='F':
                if polygon_side==0:
                    this_side_crosstrack_angle = np.deg2rad(
                        np.mod(sc_velocity_heading-90, 360))
                    this_side_crosstrack_dist = crosstrack_dist
                else:
                    this_side_crosstrack_angle = np.deg2rad(
                        np.mod(sc_velocity_heading+90, 360))
                    this_side_crosstrack_dist = crosstrack_dist
            else:
                raise ValueError("Invalid Swath Side: {}".format(swath_side))

            this_side_polygon_points = []
            for idx in idx_vec:
                sc_llh = raster_crs.xyz2llh(sc_xyz[:,idx])
                this_side_ll = raster_crs.terminal_loc_spherical(
                    sc_llh[0], sc_llh[1], this_side_crosstrack_dist,
                    this_side_crosstrack_angle[idx])
                this_side_points_deg = [[np.rad2deg(this_side_ll[0]),
                                         np.rad2deg(this_side_ll[1]),
                                         sc_llh[2]]]

                if idx==0 and alongtrack_start_buffer_dist is not None:
                    this_side_ll_buffer = raster_crs.terminal_loc_spherical(
                        this_side_ll[0], this_side_ll[1],
                        alongtrack_start_buffer_dist,
                        np.deg2rad(np.mod(sc_velocity_heading[idx]-180, 360)))
                    this_side_point_buffer_deg = [[
                        np.rad2deg(this_side_ll_buffer[0]),
                        np.rad2deg(this_side_ll_buffer[1]),
                        sc_llh[2]]]
                    this_side_points_deg = this_side_point_buffer_deg \
                                           + this_side_points_deg

                if idx==sc_xyz.shape[1]-1 and alongtrack_end_buffer_dist is not None:
                    this_side_ll_buffer = raster_crs.terminal_loc_spherical(
                        this_side_ll[0], this_side_ll[1],
                        alongtrack_start_buffer_dist,
                        np.deg2rad(np.mod(sc_velocity_heading[idx], 360)))
                    this_side_point_buffer_deg = [[
                        np.rad2deg(this_side_ll_buffer[0]),
                        np.rad2deg(this_side_ll_buffer[1]),
                        sc_llh[2]]]
                    this_side_points_deg = this_side_points_deg \
                                           + this_side_point_buffer_deg

                this_side_polygon_points.extend(
                    [point[:2] for point in
                     transf.TransformPoints(this_side_points_deg)])

            polygon.extend(this_side_polygon_points[::reverse_side])

        return polygon

    def build_product(self, populate_values=True, polygon_points=None):
        """ Assemble the product """
        if self.projection_type=='utm':
            if self.debug_flag:
                product = products.RasterUTMDebug()
            else:
                product = products.RasterUTM()
        elif self.projection_type=='geo':
            if self.debug_flag:
                product = products.RasterGeoDebug()
            else:
                product = products.RasterGeo()
        else:
            raise RasterUsageException(
                'Unknown projection type: {}'.format(self.projection_type))

        current_datetime = datetime.utcnow()
        product.history = \
            "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z : Creation".format(
                current_datetime.year, current_datetime.month,
                current_datetime.day, current_datetime.hour,
                current_datetime.minute, current_datetime.second)
        product.cycle_number = self.cycle_number
        product.pass_number = self.pass_number
        product.tile_numbers = self.tile_numbers
        product.tile_names = self.tile_names
        product.tile_polarizations = self.tile_polarizations
        product.scene_number = self.scene_number
        product.resolution = self.resolution
        product.time_granule_start = self.time_granule_start
        product.time_granule_end = self.time_granule_end
        product.time_coverage_start = self.time_coverage_start
        product.time_coverage_end = self.time_coverage_end
        product.geospatial_lon_min = self.geospatial_lon_min
        product.geospatial_lon_max = self.geospatial_lon_max
        product.geospatial_lat_min = self.geospatial_lat_min
        product.geospatial_lat_max = self.geospatial_lat_max
        product.left_first_longitude = self.left_first_longitude
        product.left_first_latitude = self.left_first_latitude
        product.left_last_longitude = self.left_last_longitude
        product.left_last_latitude = self.left_last_latitude
        product.right_first_longitude = self.right_first_longitude
        product.right_first_latitude = self.right_first_latitude
        product.right_last_longitude = self.right_last_longitude
        product.right_last_latitude = self.right_last_latitude

        coordinate_system = self.output_crs

        if self.projection_type=='utm':
            product.utm_zone_num = self.utm_zone
            product.mgrs_latitude_band = self.mgrs_band
            product.x_min = self.x_min
            product.x_max = self.x_max
            product.y_min = self.y_min
            product.y_max = self.y_max
            product['x'] = self.x_vec
            product['y'] = self.y_vec
            product.VARIABLES['crs']['projected_crs_name'] = \
                coordinate_system.GetName()
            product.VARIABLES['crs']['false_northing'] = \
                coordinate_system.GetProjParm('false_northing')
            product.VARIABLES['crs']['longitude_of_central_meridian'] = \
                coordinate_system.GetProjParm('central_meridian')
        elif self.projection_type=='geo':
            product.longitude_min = self.x_min
            product.longitude_max = self.x_max
            product.latitude_min = self.y_min
            product.latitude_max = self.y_max
            product['longitude'] = self.x_vec
            product['latitude'] = self.y_vec
        else:
            raise RasterUsageException(
                'Unknown projection type: {}'.format(self.projection_type))

        product.VARIABLES['crs']['crs_wkt'] = coordinate_system.ExportToWkt()
        product.VARIABLES['crs']['spatial_ref'] = \
            product.VARIABLES['crs']['crs_wkt']

        if populate_values:
            if self.projection_type=='utm':
                product['longitude'] = self.longitude
                product['latitude'] = self.latitude

            product['illumination_time'] = self.illumination_time
            product['illumination_time_tai'] = self.illumination_time_tai
            product.VARIABLES['illumination_time']['tai_utc_difference'] = \
                self.tai_utc_difference
            product.VARIABLES['illumination_time']['leap_second'] = \
                self.leap_second

            product['inc'] = self.inc
            product['cross_track'] = self.cross_track
            product['n_other_pix'] = self.n_other_pix
            product['ice_clim_flag'] = self.ice_clim_flag
            product['ice_dyn_flag'] = self.ice_dyn_flag

            if not self.skip_wse:
                product['wse'] = self.wse
                product['wse_qual_bitwise'] = self.wse_qual_bitwise
                product['wse_qual'] = self.wse_qual
                product['wse_uncert'] = self.wse_u
                product['n_wse_pix'] = self.n_wse_pix
                product['layover_impact'] = self.layover_impact
                product['height_cor_xover'] = self.height_cor_xover
                product['geoid'] = self.geoid
                product['solid_earth_tide'] = self.solid_earth_tide
                product['load_tide_fes'] = self.load_tide_fes
                product['load_tide_got'] = self.load_tide_got
                product['pole_tide'] = self.pole_tide
                product['model_dry_tropo_cor'] = self.model_dry_tropo_cor
                product['model_wet_tropo_cor'] = self.model_wet_tropo_cor
                product['iono_cor_gim_ka'] = self.iono_cor_gim_ka

            if not self.skip_area:
                product['water_area'] = self.water_area
                product['water_area_qual_bitwise'] = self.water_area_qual_bitwise
                product['water_area_qual'] = self.water_area_qual
                product['water_area_uncert'] = self.water_area_u
                product['water_frac'] = self.water_frac
                product['water_frac_uncert'] = self.water_frac_u
                product['n_water_area_pix'] = self.n_water_area_pix
                product['dark_frac'] = self.dark_frac

            if not self.skip_sig0:
                product['sig0'] = self.sig0
                product['sig0_qual_bitwise'] = self.sig0_qual_bitwise
                product['sig0_qual'] = self.sig0_qual
                product['sig0_uncert'] = self.sig0_u
                product['sig0_cor_atmos_model'] = self.sig0_cor_atmos_model
                product['n_sig0_pix'] = self.n_sig0_pix

            if self.debug_flag:
                product['classification'] = self.classification

        # Crop the product to the desired bounds
        if polygon_points is not None:
            product.crop_to_bounds(polygon_points)

        return product
