'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author (s): Shuai Zhang (UNC) and Alexander Corben (JPL)
'''

import logging
import numpy as np
import SWOTWater.aggregate as ag
import SWOTRaster.products as products
import SWOTRaster.raster_crs as raster_crs

from osgeo import osr
from datetime import datetime
from itertools import groupby
from shapely.geometry import Point, Polygon
from SWOTRaster.errors import RasterUsageException

LOGGER = logging.getLogger(__name__)

class RasterProcessor(object):
    def __init__(self, projection_type, resolution, padding,
                 height_agg_method, area_agg_method, interior_water_classes,
                 water_edge_classes, land_edge_classes, dark_water_classes,
                 use_bright_land,
                 geo_qual_suspect, geo_qual_degraded, geo_qual_bad,
                 class_qual_suspect, class_qual_degraded, class_qual_bad,
                 sig0_qual_suspect, sig0_qual_degraded, sig0_qual_bad,
                 num_good_sus_pix_thresh_wse, num_good_sus_pix_thresh_water_area,
                 num_good_sus_pix_thresh_sig0, pixc_water_frac_suspect_thresh,
                 num_wse_pix_suspect_thresh, num_water_area_pix_suspect_thresh,
                 num_sig0_pix_suspect_thresh,
                 near_range_suspect_thresh, far_range_suspect_thresh,
                 wse_uncert_suspect_thresh, water_frac_uncert_suspect_thresh,
                 sig0_uncert_suspect_thresh,
                 wse_bad_thresh_min, wse_bad_thresh_max,
                 water_frac_bad_thresh_min, water_frac_bad_thresh_max,
                 sig0_bad_thresh_min, sig0_bad_thresh_max,
                 utm_zone_adjust=0, mgrs_band_adjust=0, debug_flag=False):

        self.projection_type = projection_type
        if self.projection_type=='geo':
            # Geodetic resolution is given in arcsec
            self.resolution = np.float(resolution/(60*60))
        elif self.projection_type=='utm':
            self.resolution = np.float(resolution)
            self.utm_zone_adjust = utm_zone_adjust
            self.mgrs_band_adjust = mgrs_band_adjust
        else:
            raise RasterUsageException(
                'Unknown projection type: {}'.format(self.projection_type))

        self.padding = padding
        self.height_agg_method = height_agg_method
        self.area_agg_method = area_agg_method
        self.interior_water_classes = interior_water_classes
        self.water_edge_classes = water_edge_classes
        self.land_edge_classes = land_edge_classes
        self.dark_water_classes = dark_water_classes
        self.use_bright_land = use_bright_land

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

        self.debug_flag = debug_flag

    def rasterize(self, pixc, polygon_points=None, use_improved_geoloc=True):
        """ Rasterize pixc to raster """
        LOGGER.info("rasterizing")

        self.input_crs = raster_crs.wgs84_crs()
        self.cycle_number = pixc.cycle_number
        self.pass_number = pixc.pass_number
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
            swath_corners = \
                [(pixc.left_first_latitude, pixc.left_first_longitude),
                 (pixc.right_first_latitude, pixc.right_first_longitude),
                 (pixc.right_last_latitude, pixc.right_last_longitude),
                 (pixc.left_last_latitude, pixc.left_last_longitude)]
            self.create_projection_from_polygon(swath_corners)
        else:
            self.create_projection_from_polygon(polygon_points)

        # Get pixc classification masks
        water_classes = np.concatenate((self.interior_water_classes,
                                        self.water_edge_classes,
                                        self.dark_water_classes))
        all_classes = np.concatenate((water_classes, self.land_edge_classes))

        water_classes_mask = pixc.get_mask(water_classes, use_improved_geoloc)
        all_classes_mask = pixc.get_mask(all_classes, use_improved_geoloc)

        bright_land_flag = pixc['pixel_cloud']['bright_land_flag']
        if not self.use_bright_land:
            water_classes_mask = np.logical_and(water_classes_mask,
                                                np.logical_not(bright_land_flag))
            all_classes_mask = np.logical_and(all_classes_mask,
                                              np.logical_not(bright_land_flag))

        # Get pixc summary quality flags
        geo_qual_pixc_flag = pixc.get_summary_qual_flag(
            'geolocation_qual', self.geo_qual_suspect,
            self.geo_qual_degraded,self.geo_qual_bad)
        class_qual_pixc_flag = pixc.get_summary_qual_flag(
            'classification_qual', self.class_qual_suspect,
            self.class_qual_degraded, self.class_qual_bad)
        sig0_qual_pixc_flag = pixc.get_summary_qual_flag(
            'sig0_qual', self.sig0_qual_suspect,
            self.sig0_qual_degraded, self.sig0_qual_bad)

        # Create an empty Raster
        empty_product = self.build_product(populate_values=False)
        # Return empty product if pixc is empty
        if len(pixc['pixel_cloud']['height'])==0:
            LOGGER.warn('Empty Pixel Cloud: returning empty raster')
            return empty_product

        self.proj_mapping = empty_product.get_raster_mapping(
            pixc, all_classes_mask, use_improved_geoloc)

        # Get rasterization masks for wse/water_area/sig0/all
        wse_mask, water_area_mask, sig0_mask, all_mask = \
            self.get_rasterization_masks(
                water_classes_mask, all_classes_mask,
                geo_qual_pixc_flag, class_qual_pixc_flag,
                sig0_qual_pixc_flag)

        self.aggregate_sig0_corrections(pixc, sig0_mask)
        self.aggregate_wse_corrections(pixc, wse_mask)
        self.aggregate_wse(pixc, wse_mask, use_improved_geoloc)
        self.aggregate_water_area(pixc, water_area_mask)
        self.aggregate_sig0(pixc, sig0_mask)
        self.aggregate_cross_track(pixc, all_mask)
        self.aggregate_inc(pixc, all_mask)
        self.aggregate_dark_frac(pixc, all_mask)
        self.aggregate_illumination_time(pixc, all_mask)
        self.aggregate_ice_flags(pixc, all_mask)
        self.aggregate_layover_impact(pixc, wse_mask)
        self.aggregate_wse_qual(
            wse_mask, geo_qual_pixc_flag, class_qual_pixc_flag,
            bright_land_flag)
        self.aggregate_water_area_qual(
            water_area_mask, geo_qual_pixc_flag, class_qual_pixc_flag,
            bright_land_flag, pixc['pixel_cloud']['water_frac'])
        self.aggregate_sig0_qual(
            sig0_mask, geo_qual_pixc_flag, class_qual_pixc_flag,
            sig0_qual_pixc_flag, bright_land_flag)
        self.flag_large_karin_gaps(pixc)
        self.flag_inner_swath(pixc)

        if self.projection_type == 'utm':
            self.aggregate_lat_lon(all_mask)

        if self.debug_flag:
            self.aggregate_classification(pixc, all_mask)

        return self.build_product(polygon_points=polygon_points)

    def create_projection_from_polygon(self, polygon_points):
        """ Create the output projection given a bounding polygon """
        LOGGER.info("creating projection from polygon")

        poly_edge_y = [point[0] for point in polygon_points]
        poly_edge_x = [point[1] for point in polygon_points]

        if self.projection_type=='geo':
            self.output_crs = raster_crs.wgs84_crs()
            proj_center_x = 0
            proj_center_y = 0
        elif self.projection_type=='utm':
            lat_mid = np.mean(poly_edge_y)
            lon_mid = np.mean(poly_edge_x)
            utm_zone = raster_crs.utm_zone_from_latlon(lat_mid, lon_mid)
            mgrs_band = raster_crs.mgrs_band_from_latlon(lat_mid, lon_mid)
            # adjust the utm zone (-1 and +1 as zone numbers are 1 indexed)
            utm_zone = np.mod(utm_zone + self.utm_zone_adjust - 1,
                              raster_crs.UTM_NUM_ZONES) + 1

            # adjust/shift the mgrs band
            mgrs_band = raster_crs.mgrs_band_shift(mgrs_band,
                                                   self.mgrs_band_adjust,
                                                   lon_mid)

            self.output_crs = raster_crs.utm_crs(utm_zone, mgrs_band)

            transf = osr.CoordinateTransformation(self.input_crs,
                                                  self.output_crs)

            polygon_points = [(transf.TransformPoint(point[0], point[1])[:2])
                              for point in polygon_points]
            poly_edge_y = [point[1] for point in polygon_points]
            poly_edge_x = [point[0] for point in polygon_points]

            proj_center_x = self.output_crs.GetProjParm('false_easting')
            proj_center_y = self.output_crs.GetProjParm('false_northing')
        else:
            raise RasterUsageException(
                'Unknown projection type: {}'.format(self.projection_type))

        # get the coordinate limits
        x_min = np.min(poly_edge_x)
        x_max = np.max(poly_edge_x)
        y_min = np.min(poly_edge_y)
        y_max = np.max(poly_edge_y)

        # round limits to the nearest bin (centered at proj_center_x with pad)
        x_min = int(round((x_min - proj_center_x) / self.resolution)) \
                * self.resolution + proj_center_x - self.padding
        x_max = int(round((x_max - proj_center_x) / self.resolution)) \
                * self.resolution + proj_center_x + self.padding
        y_min = int(round((y_min - proj_center_y) / self.resolution)) \
                * self.resolution + proj_center_y - self.padding
        y_max = int(round((y_max - proj_center_y) / self.resolution)) \
                * self.resolution + proj_center_y + self.padding

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.size_x = int(round((x_max - x_min) / self.resolution)) + 1
        self.size_y = int(round((y_max - y_min) / self.resolution)) + 1
        if self.projection_type=='utm':
            self.utm_zone = np.short(utm_zone)
            self.utm_hemisphere = raster_crs.hemisphere_from_mgrs_band(mgrs_band)
            self.mgrs_band = mgrs_band

        LOGGER.info({'proj': self.output_crs.ExportToWkt(),
                     'res': self.resolution,
                     'x_min': self.x_min,
                     'x_max': self.x_max,
                     'size_x': self.size_x,
                     'y_min': self.y_min,
                     'y_max': self.y_max,
                     'size_y': self.size_y})

    def get_rasterization_masks(self, water_classes_mask, all_classes_mask,
                                geo_qual_pixc_flag, class_qual_pixc_flag,
                                sig0_qual_pixc_flag):
        """ Get masks of pixels to rasterize for wse/water_area/sig0/all"""
        LOGGER.info('getting rasterization masks for wse/water_area/sig0/all')

        common_qual_flag = [max(x, y) for x, y
                            in zip(geo_qual_pixc_flag, class_qual_pixc_flag)]
        common_good_qual_mask = \
            [x==products.QUAL_IND_GOOD for x in common_qual_flag]
        common_suspect_qual_mask = \
            [x==products.QUAL_IND_SUSPECT for x in common_qual_flag]
        common_degraded_qual_mask = \
            [x==products.QUAL_IND_DEGRADED for x in common_qual_flag]

        sig0_qual_flag = [max(x, y) for x,y \
                          in zip(common_qual_flag, sig0_qual_pixc_flag)]
        sig0_good_qual_mask = \
            [x==products.QUAL_IND_GOOD for x in sig0_qual_flag]
        sig0_suspect_qual_mask = \
            [x==products.QUAL_IND_SUSPECT for x in sig0_qual_flag]
        sig0_degraded_qual_mask = \
            [x==products.QUAL_IND_DEGRADED for x in sig0_qual_flag]

        wse_good_mask = np.logical_and(
            water_classes_mask, common_good_qual_mask)
        wse_suspect_mask = np.logical_and(
            water_classes_mask, common_suspect_qual_mask)
        wse_degraded_mask = np.logical_and(
            water_classes_mask, common_degraded_qual_mask)

        water_area_good_mask = np.logical_and(
            all_classes_mask, common_good_qual_mask)
        water_area_suspect_mask = np.logical_and(
            all_classes_mask, common_suspect_qual_mask)
        water_area_degraded_mask = np.logical_and(
            all_classes_mask, common_degraded_qual_mask)

        sig0_good_mask = np.logical_and(
            water_classes_mask, sig0_good_qual_mask)
        sig0_suspect_mask = np.logical_and(
            water_classes_mask, sig0_suspect_qual_mask)
        sig0_degraded_mask = np.logical_and(
            water_classes_mask, sig0_degraded_qual_mask)

        wse_mask = self.get_rasterization_mask(
            wse_good_mask, wse_suspect_mask, wse_degraded_mask,
            self.num_good_sus_pix_thresh_wse)
        water_area_mask = self.get_rasterization_mask(
            water_area_good_mask, water_area_suspect_mask, water_area_degraded_mask,
            self.num_good_sus_pix_thresh_water_area)
        sig0_mask = self.get_rasterization_mask(
            sig0_good_mask, sig0_suspect_mask, sig0_degraded_mask,
            self.num_good_sus_pix_thresh_sig0)
        all_mask = np.logical_or.reduce((wse_mask, water_area_mask, sig0_mask))

        return wse_mask, water_area_mask, sig0_mask, all_mask

    def get_rasterization_mask(self, good_mask, suspect_mask, degraded_mask,
                               num_good_sus_pix_thresh):
        """ Get mask of pixels to rasterize """
        LOGGER.info('getting rasterization mask')
        good_sus_mask = np.logical_or(good_mask, suspect_mask)
        good_sus_degraded_mask = np.logical_or(good_sus_mask, degraded_mask)

        rasterization_mask = np.ma.zeros(good_mask.shape, dtype=bool)

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = good_sus_mask[self.proj_mapping[i][j]]
                if np.sum(mask) < num_good_sus_pix_thresh:
                    mask = good_sus_degraded_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    these_idxs = [idx for idx,valid
                                  in zip(self.proj_mapping[i][j], mask) if valid]
                    rasterization_mask[these_idxs] = True

        return rasterization_mask

    def aggregate_sig0_corrections(self, pixc, rasterization_mask):
        """ Aggregate sig0 geophysical corrections """
        LOGGER.info("aggregating sig0 corrections")

        pixc_sig0_cor_atmos_model = pixc['pixel_cloud']['sig0_cor_atmos_model']

        self.sig0_cor_atmos_model = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    self.sig0_cor_atmos_model[i][j] = ag.simple(
                        pixc_sig0_cor_atmos_model[self.proj_mapping[i][j]][mask],
                        metric='mean')

    def aggregate_wse_corrections(self, pixc, rasterization_mask):
        """ Aggregate wse geophysical corrections """
        LOGGER.info("aggregating wse corrections")

        pixc_height_cor_xover = pixc['pixel_cloud']['height_cor_xover']
        pixc_geoid = pixc['pixel_cloud']['geoid']
        pixc_solid_earth_tide = pixc['pixel_cloud']['solid_earth_tide']
        pixc_load_tide_fes = pixc['pixel_cloud']['load_tide_fes']
        pixc_load_tide_got = pixc['pixel_cloud']['load_tide_got']
        pixc_pole_tide = pixc['pixel_cloud']['pole_tide']
        pixc_model_dry_tropo_cor = pixc['pixel_cloud']['model_dry_tropo_cor']
        pixc_model_wet_tropo_cor = pixc['pixel_cloud']['model_wet_tropo_cor']
        pixc_iono_cor_gim_ka = pixc['pixel_cloud']['iono_cor_gim_ka']
        pixc_dh_dphi = pixc['pixel_cloud']['dheight_dphase']
        pixc_phase_noise_std = pixc['pixel_cloud']['phase_noise_std']

        pixc_height_std = np.abs(pixc_phase_noise_std * pixc_dh_dphi)
        # set bad pix height std to high number to deweight
        # instead of giving infs/nans
        bad_num = 1.0e5
        pixc_height_std[pixc_height_std<=0] = bad_num
        pixc_height_std[np.isinf(pixc_height_std)] = bad_num
        pixc_height_std[np.isnan(pixc_height_std)] = bad_num

        self.height_cor_xover = np.ma.masked_all((self.size_y, self.size_x))
        self.geoid = np.ma.masked_all((self.size_y, self.size_x))
        self.solid_earth_tide = np.ma.masked_all((self.size_y, self.size_x))
        self.load_tide_fes = np.ma.masked_all((self.size_y, self.size_x))
        self.load_tide_got = np.ma.masked_all((self.size_y, self.size_x))
        self.pole_tide = np.ma.masked_all((self.size_y, self.size_x))
        self.model_dry_tropo_cor = np.ma.masked_all((self.size_y, self.size_x))
        self.model_wet_tropo_cor = np.ma.masked_all((self.size_y, self.size_x))
        self.iono_cor_gim_ka = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    self.height_cor_xover[i][j] = ag.height_only(
                        pixc_height_cor_xover[self.proj_mapping[i][j]],
                        mask,
                        pixc_height_std[self.proj_mapping[i][j]],
                        method=self.height_agg_method)[0]
                    self.geoid[i][j] = ag.simple(
                        pixc_geoid[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.solid_earth_tide[i][j] = ag.simple(
                        pixc_solid_earth_tide[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.load_tide_fes[i][j] = ag.simple(
                        pixc_load_tide_fes[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.load_tide_got[i][j] = ag.simple(
                        pixc_load_tide_got[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.pole_tide[i][j] = ag.simple(
                        pixc_pole_tide[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.model_dry_tropo_cor[i][j] = ag.simple(
                        pixc_model_dry_tropo_cor[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.model_wet_tropo_cor[i][j] = ag.simple(
                        pixc_model_wet_tropo_cor[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.iono_cor_gim_ka[i][j] = ag.simple(
                        pixc_iono_cor_gim_ka[self.proj_mapping[i][j]][mask],
                        metric='mean')

    def apply_wse_corrections(self):
        """ Apply geophysical corrections to wse """
        LOGGER.info("applying wse corrections")

        self.wse -= (
            self.geoid +
            self.solid_earth_tide +
            self.load_tide_fes +
            self.pole_tide)

    def aggregate_wse(self, pixc, rasterization_mask, use_improved_geoloc=True):
        """ Aggregate water surface elevation and associated uncertainties """
        LOGGER.info("aggregating wse")

        pixc_height = pixc['pixel_cloud']['height']
        pixc_num_rare_looks = pixc['pixel_cloud']['eff_num_rare_looks']
        pixc_num_med_looks = pixc['pixel_cloud']['eff_num_medium_looks']

        pixc_power_plus_y = pixc['pixel_cloud']['power_plus_y']
        pixc_power_minus_y = pixc['pixel_cloud']['power_minus_y']

        pixc_dh_dphi = pixc['pixel_cloud']['dheight_dphase']
        pixc_dlat_dphi = pixc['pixel_cloud']['dlatitude_dphase']
        pixc_dlon_dphi = pixc['pixel_cloud']['dlongitude_dphase']
        pixc_phase_noise_std = pixc['pixel_cloud']['phase_noise_std']

        pixc_height_std = np.abs(pixc_phase_noise_std * pixc_dh_dphi)
        # set bad pix height std to high number to deweight
        # instead of giving infs/nans
        bad_num = 1.0e5
        pixc_height_std[pixc_height_std<=0] = bad_num
        pixc_height_std[np.isinf(pixc_height_std)] = bad_num
        pixc_height_std[np.isnan(pixc_height_std)] = bad_num

        looks_to_efflooks = pixc['pixel_cloud'].looks_to_efflooks

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

        pixc_ifgram = pixc['pixel_cloud']['interferogram']
        tvp_plus_y_antenna_xyz = (pixc['tvp']['plus_y_antenna_x'],
                                  pixc['tvp']['plus_y_antenna_y'],
                                  pixc['tvp']['plus_y_antenna_z'])
        tvp_minus_y_antenna_xyz = (pixc['tvp']['minus_y_antenna_x'],
                                   pixc['tvp']['minus_y_antenna_y'],
                                   pixc['tvp']['minus_y_antenna_z'])
        pixc_tvp_index = ag.get_sensor_index(pixc)
        pixc_wavelength = pixc.wavelength
        flat_ifgram = ag.flatten_interferogram(pixc_ifgram,
                                               tvp_plus_y_antenna_xyz,
                                               tvp_minus_y_antenna_xyz,
                                               target_xyz,
                                               pixc_tvp_index,
                                               pixc_wavelength)

        self.wse = np.ma.masked_all((self.size_y, self.size_x))
        self.wse_u = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    grid_height = ag.height_with_uncerts(
                        pixc_height[self.proj_mapping[i][j]],
                        mask,
                        pixc_num_rare_looks[self.proj_mapping[i][j]],
                        pixc_num_med_looks[self.proj_mapping[i][j]],
                        flat_ifgram[self.proj_mapping[i][j]],
                        pixc_power_minus_y[self.proj_mapping[i][j]],
                        pixc_power_plus_y[self.proj_mapping[i][j]],
                        looks_to_efflooks,
                        pixc_dh_dphi[self.proj_mapping[i][j]],
                        pixc_dlat_dphi[self.proj_mapping[i][j]],
                        pixc_dlon_dphi[self.proj_mapping[i][j]],
                        pixc_height_std[self.proj_mapping[i][j]],
                        method=self.height_agg_method)

                    self.wse[i][j] = grid_height[0]
                    self.wse_u[i][j] = grid_height[2]

        self.apply_wse_corrections()

    def aggregate_water_area(self, pixc, rasterization_mask):
        """ Aggregate water area, water fraction and associated uncertainties """
        LOGGER.info("aggregating water area")

        pixc_pixel_area = pixc['pixel_cloud']['pixel_area']
        pixc_water_fraction = pixc['pixel_cloud']['water_frac']
        pixc_water_fraction_uncert = pixc['pixel_cloud']['water_frac_uncert']
        pixc_darea_dheight = pixc['pixel_cloud']['darea_dheight']
        pixc_pfd = pixc['pixel_cloud']['false_detection_rate']
        pixc_pmd = pixc['pixel_cloud']['missed_detection_rate']
        pixc_classif = pixc['pixel_cloud']['classification']

        self.water_area = np.ma.masked_all((self.size_y, self.size_x))
        self.water_area_u = np.ma.masked_all((self.size_y, self.size_x))
        self.water_frac = np.ma.masked_all((self.size_y, self.size_x))
        self.water_frac_u = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    grid_area = ag.area_with_uncert(
                        pixc_pixel_area[self.proj_mapping[i][j]],
                        pixc_water_fraction[self.proj_mapping[i][j]],
                        pixc_water_fraction_uncert[self.proj_mapping[i][j]],
                        pixc_darea_dheight[self.proj_mapping[i][j]],
                        pixc_classif[self.proj_mapping[i][j]],
                        pixc_pfd[self.proj_mapping[i][j]],
                        pixc_pmd[self.proj_mapping[i][j]],
                        mask,
                        method=self.area_agg_method,
                        interior_water_klasses=self.interior_water_classes,
                        water_edge_klasses=self.water_edge_classes,
                        land_edge_klasses=self.land_edge_classes,
                        dark_water_klasses=self.dark_water_classes)

                    self.water_area[i][j] = grid_area[0]
                    self.water_area_u[i][j] = grid_area[1]

                    if self.projection_type == 'utm':
                        pixel_area = self.resolution**2
                    elif self.projection_type == 'geo':
                        px_latitude = self.y_min + self.resolution*i
                        pixel_area = raster_crs.wgs84_px_area(
                            px_latitude, self.resolution)
                    else:
                        raise RasterUsageException(
                            'Unknown projection type: {}'.format(
                                self.projection_type))

                    self.water_frac[i][j] = grid_area[0]/pixel_area
                    self.water_frac_u[i][j] = grid_area[1]/pixel_area

    def aggregate_sig0(self, pixc, rasterization_mask):
        """ Aggregate sigma0 """
        LOGGER.info("aggregating sigma0")

        pixc_sig0 = pixc['pixel_cloud']['sig0']
        pixc_sig0_uncert = pixc['pixel_cloud']['sig0_uncert']

        self.sig0 = np.ma.masked_all((self.size_y, self.size_x))
        self.sig0_u = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    grid_sig0 = ag.sig0_with_uncerts(
                        pixc_sig0[self.proj_mapping[i][j]], mask,
                        pixc_sig0_uncert[self.proj_mapping[i][j]],
                        method='rare')
                    self.sig0[i][j] = grid_sig0[0]
                    self.sig0_u[i][j] = grid_sig0[2]

    def aggregate_cross_track(self, pixc, rasterization_mask):
        """ Aggregate cross track """
        LOGGER.info("aggregating cross track")

        pixc_cross_track = pixc['pixel_cloud']['cross_track']

        self.cross_track = np.ma.masked_all((self.size_y, self.size_x))
        self.n_other_pix = np.ma.zeros((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    self.cross_track[i][j] = ag.simple(
                        pixc_cross_track[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.n_other_pix[i][j] = ag.simple(mask, metric='sum')

    def aggregate_inc(self, pixc, rasterization_mask):
        """ Aggregate incidence angle """
        LOGGER.info("aggregating incidence angle")

        pixc_inc = pixc['pixel_cloud']['inc']

        self.inc = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    self.inc[i][j] = ag.simple(
                        pixc_inc[self.proj_mapping[i][j]][mask], metric='mean')

    def aggregate_dark_frac(self, pixc, rasterization_mask):
        """ Aggregate dark water fraction """
        LOGGER.info("aggregating dark fraction")

        pixc_classif = pixc['pixel_cloud']['classification']
        pixc_pixel_area = pixc['pixel_cloud']['pixel_area']
        pixc_water_fraction = pixc['pixel_cloud']['water_frac']
        pixc_dark_mask = np.isin(pixc_classif, self.dark_water_classes)

        self.dark_frac = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    dark_mask = np.logical_and(
                        pixc_dark_mask[self.proj_mapping[i][j]], mask)
                    dark_area = ag.simple(
                        pixc_pixel_area[self.proj_mapping[i][j]][dark_mask],
                        metric='sum')
                    total_area, _ = ag.area_only(
                        pixc_pixel_area[self.proj_mapping[i][j]],
                        pixc_water_fraction[self.proj_mapping[i][j]],
                        pixc_classif[self.proj_mapping[i][j]],
                        mask,
                        method=self.area_agg_method,
                        interior_water_klasses=self.interior_water_classes,
                        water_edge_klasses=self.water_edge_classes,
                        land_edge_klasses=self.land_edge_classes,
                        dark_water_klasses=self.dark_water_classes)

                    if not np.any(dark_mask) or total_area==0:
                        self.dark_frac[i][j] = 0
                    else:
                        self.dark_frac[i][j] = dark_area/total_area

    def aggregate_illumination_time(self, pixc, rasterization_mask):
        """ Aggregate illumination time """
        LOGGER.info("aggregating illumination time")

        pixc_illumination_time = pixc['pixel_cloud']['illumination_time']
        pixc_illumination_time_tai = pixc['pixel_cloud']['illumination_time_tai']

        self.illumination_time = np.ma.masked_all((self.size_y, self.size_x))
        self.illumination_time_tai = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    self.illumination_time[i][j] = ag.simple(
                        pixc_illumination_time[self.proj_mapping[i][j]][mask],
                        metric='mean')
                    self.illumination_time_tai[i][j] = ag.simple(
                        pixc_illumination_time_tai[self.proj_mapping[i][j]][mask],
                        metric='mean')

        # Set the time coverage start and end based on illumination time
        if np.all(self.illumination_time.mask):
            self.time_coverage_start = products.EMPTY_DATETIME
            self.time_coverage_end = products.EMPTY_DATETIME
        else:
            start_illumination_time = np.min(self.illumination_time)
            end_illumination_time = np.max(self.illumination_time)
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
            np.argmin(self.illumination_time), self.illumination_time.shape)
        self.tai_utc_difference = \
            self.illumination_time_tai[min_illumination_time_index] \
            - self.illumination_time[min_illumination_time_index]

        # Set leap second
        if pixc.leap_second == products.EMPTY_LEAPSEC:
            self.leap_second = products.EMPTY_LEAPSEC
        else:
            leap_second = datetime.strptime(
                pixc.leap_second, products.LEAPSEC_FORMAT_STR)
            if leap_second < start_time or leap_second > end_time:
                leap_second = products.EMPTY_LEAPSEC

            self.leap_second = leap_second.strftime(
                products.LEAPSEC_FORMAT_STR)

    def aggregate_ice_flags(self, pixc, rasterization_mask):
        """ Aggregate ice flags """
        LOGGER.info("aggregating ice flags")

        pixc_ice_clim_flag = pixc['pixel_cloud']['ice_clim_flag']
        pixc_ice_dyn_flag = pixc['pixel_cloud']['ice_dyn_flag']

        self.ice_clim_flag = np.ma.masked_all((self.size_y, self.size_x))
        self.ice_dyn_flag = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    valid_ice_clim_flag = \
                        pixc_ice_clim_flag[self.proj_mapping[i][j]][mask]
                    valid_ice_dyn_flag = \
                        pixc_ice_dyn_flag[self.proj_mapping[i][j]][mask]

                    # If all flags are the same, then we return that flag value
                    if np.all(valid_ice_clim_flag == valid_ice_clim_flag[0]):
                        self.ice_clim_flag[i][j] = valid_ice_clim_flag[0]
                    else: # otherwise, return a value of 1 (partially covered)
                        self.ice_clim_flag[i][j] = 1

                    # If all flags are the same, then we return that flag value
                    if np.all(valid_ice_dyn_flag == valid_ice_dyn_flag[0]):
                        self.ice_dyn_flag[i][j] = valid_ice_dyn_flag[0]
                    else: # otherwise, return a value of 1 (partially covered)
                        self.ice_dyn_flag[i][j] = 1

    def aggregate_layover_impact(self, pixc, rasterization_mask):
        """ Aggregate layover impact """
        LOGGER.info("aggregating layover impact")

        pixc_layover_impact = pixc['pixel_cloud']['layover_impact']

        pixc_dh_dphi = pixc['pixel_cloud']['dheight_dphase']
        pixc_phase_noise_std = pixc['pixel_cloud']['phase_noise_std']
        pixc_height_std = np.abs(pixc_phase_noise_std * pixc_dh_dphi)
        # set bad pix height std to high number to deweight
        # instead of giving infs/nans
        bad_num = 1.0e5
        pixc_height_std[pixc_height_std<=0] = bad_num
        pixc_height_std[np.isinf(pixc_height_std)] = bad_num
        pixc_height_std[np.isnan(pixc_height_std)] = bad_num

        self.layover_impact = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    self.layover_impact[i][j] = ag.height_only(
                        pixc_layover_impact[self.proj_mapping[i][j]],
                        mask,
                        pixc_height_std[self.proj_mapping[i][j]],
                        method=self.height_agg_method)[0]

    def aggregate_wse_qual(self, rasterization_mask,
                           geo_qual, class_qual, bright_land_flag):
        """ Aggregate wse qual """
        LOGGER.info("aggregating wse qual")

        self.n_wse_pix = np.ma.zeros((self.size_y, self.size_x))
        self.wse_qual = \
            products.QUAL_IND_BAD + np.ma.zeros((self.size_y, self.size_x))
        self.wse_qual_bitwise = \
            products.QUAL_IND_NO_PIXELS + products.QUAL_IND_FEW_PIXELS \
            + np.ma.zeros((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if not np.any(mask):
                    continue

                self.n_wse_pix[i][j] = ag.simple(mask, metric='sum')

                # Default qual flags are no_pixels, if we have pixels reset them
                self.wse_qual[i][j] = products.QUAL_IND_GOOD
                self.wse_qual_bitwise[i][j] = products.QUAL_IND_GOOD

                these_idxs = [idx for idx,valid
                                  in zip(self.proj_mapping[i][j], mask) if valid]
                this_geo_qual = geo_qual[these_idxs]
                this_class_qual = class_qual[these_idxs]
                this_bright_land_flag = bright_land_flag[these_idxs]

                if np.any(this_class_qual==products.QUAL_IND_SUSPECT):
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_CLASS_QUAL_SUSPECT

                if np.any(this_geo_qual==products.QUAL_IND_SUSPECT):
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_GEOLOCATION_QUAL_SUSPECT

                if self.wse_u[i][j] > self.wse_uncert_suspect_thresh:
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_LARGE_UNCERT_SUSPECT

                if np.any(this_bright_land_flag):
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_BRIGHT_LAND

                if self.n_wse_pix[i][j] < self.num_wse_pix_suspect_thresh:
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_FEW_PIXELS

                if abs(self.cross_track[i][j]) > self.far_range_suspect_thresh:
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_FAR_RANGE_SUSPECT

                if abs(self.cross_track[i][j]) < self.near_range_suspect_thresh:
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_NEAR_RANGE_SUSPECT

                if np.any(this_class_qual==products.QUAL_IND_DEGRADED):
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_DEGRADED)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_CLASS_QUAL_DEGRADED

                if np.any(this_geo_qual==products.QUAL_IND_DEGRADED):
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_DEGRADED)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_GEOLOCATION_QUAL_DEGRADED

                if self.wse[i][j] < self.wse_bad_thresh_min \
                   or self.wse[i][j] > self.wse_bad_thresh_max:
                    self.wse_qual[i][j] = max(
                        self.wse_qual[i][j], products.QUAL_IND_BAD)
                    self.wse_qual_bitwise[i][j] += \
                        products.QUAL_IND_VALUE_BAD

    def aggregate_water_area_qual(self, rasterization_mask,
                                  geo_qual, class_qual, bright_land_flag,
                                  pixc_water_frac):
        """ Aggregate water area qual """
        LOGGER.info("aggregating water area qual")

        self.n_water_area_pix = np.ma.zeros((self.size_y, self.size_x))
        self.water_area_qual = \
            products.QUAL_IND_BAD + np.ma.zeros((self.size_y, self.size_x))
        self.water_area_qual_bitwise = \
            products.QUAL_IND_NO_PIXELS + products.QUAL_IND_FEW_PIXELS \
            + np.ma.zeros((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if not np.any(mask):
                    continue

                self.n_water_area_pix[i][j] = ag.simple(mask, metric='sum')

                if self.n_water_area_pix[i][j] <= 0:
                    continue # If no pixels, don't check other flags

                # Default qual flags are no_pixels, if we have pixels reset them
                self.water_area_qual[i][j] = products.QUAL_IND_GOOD
                self.water_area_qual_bitwise[i][j] = products.QUAL_IND_GOOD

                these_idxs = [idx for idx,valid
                                  in zip(self.proj_mapping[i][j], mask) if valid]
                this_geo_qual = geo_qual[these_idxs]
                this_class_qual = class_qual[these_idxs]
                this_bright_land_flag = bright_land_flag[these_idxs]
                this_pixc_water_frac = pixc_water_frac[these_idxs]

                if np.any(this_class_qual==products.QUAL_IND_SUSPECT):
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_CLASS_QUAL_SUSPECT

                if np.any(this_geo_qual==products.QUAL_IND_SUSPECT):
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_GEOLOCATION_QUAL_SUSPECT

                if np.any(this_pixc_water_frac>self.pixc_water_frac_suspect_thresh):
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_WATER_FRACTION_SUSPECT

                if self.water_frac_u[i][j] > self.water_frac_uncert_suspect_thresh:
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_LARGE_UNCERT_SUSPECT

                if np.any(this_bright_land_flag):
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_BRIGHT_LAND

                if self.n_water_area_pix[i][j] < self.num_water_area_pix_suspect_thresh:
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_FEW_PIXELS

                if abs(self.cross_track[i][j]) > self.far_range_suspect_thresh:
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_FAR_RANGE_SUSPECT

                if abs(self.cross_track[i][j]) < self.near_range_suspect_thresh:
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_NEAR_RANGE_SUSPECT

                if np.any(this_class_qual==products.QUAL_IND_DEGRADED):
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_DEGRADED)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_CLASS_QUAL_DEGRADED

                if np.any(this_geo_qual==products.QUAL_IND_DEGRADED):
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_DEGRADED)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_GEOLOCATION_QUAL_DEGRADED

                if self.water_frac[i][j] < self.water_frac_bad_thresh_min \
                   or self.water_frac[i][j] > self.water_frac_bad_thresh_max:
                    self.water_area_qual[i][j] = max(
                        self.water_area_qual[i][j], products.QUAL_IND_BAD)
                    self.water_area_qual_bitwise[i][j] += \
                        products.QUAL_IND_VALUE_BAD

    def aggregate_sig0_qual(self, rasterization_mask,
                            geo_qual, class_qual, sig0_qual, bright_land_flag):
        """ Aggregate sig0 qual """
        LOGGER.info("aggregating sig0 qual")

        self.n_sig0_pix = np.ma.zeros((self.size_y, self.size_x))
        self.sig0_qual = \
            products.QUAL_IND_BAD + np.ma.zeros((self.size_y, self.size_x))
        self.sig0_qual_bitwise = \
            products.QUAL_IND_NO_PIXELS + products.QUAL_IND_FEW_PIXELS \
            + np.ma.zeros((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if not np.any(mask):
                    continue

                self.n_sig0_pix[i][j] = ag.simple(mask, metric='sum')

                if self.n_sig0_pix[i][j] <= 0:
                    continue # If no pixels, don't check other flags

                # Default qual flags are no_pixels, if we have pixels reset them
                self.sig0_qual[i][j] = products.QUAL_IND_GOOD
                self.sig0_qual_bitwise[i][j] = products.QUAL_IND_GOOD

                these_idxs = [idx for idx,valid
                                  in zip(self.proj_mapping[i][j], mask) if valid]
                this_geo_qual = geo_qual[these_idxs]
                this_class_qual = class_qual[these_idxs]
                this_sig0_qual = sig0_qual[these_idxs]
                this_bright_land_flag = bright_land_flag[these_idxs]

                if np.any(this_sig0_qual==products.QUAL_IND_SUSPECT):
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_SIG0_QUAL_SUSPECT

                if np.any(this_class_qual==products.QUAL_IND_SUSPECT):
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_CLASS_QUAL_SUSPECT

                if np.any(this_geo_qual==products.QUAL_IND_SUSPECT):
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_GEOLOCATION_QUAL_SUSPECT

                if self.sig0_u[i][j] > self.sig0_uncert_suspect_thresh:
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_LARGE_UNCERT_SUSPECT

                if np.any(this_bright_land_flag):
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_BRIGHT_LAND

                if self.n_sig0_pix[i][j] < self.num_sig0_pix_suspect_thresh:
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_FEW_PIXELS

                if abs(self.cross_track[i][j]) > self.far_range_suspect_thresh:
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_FAR_RANGE_SUSPECT

                if abs(self.cross_track[i][j]) < self.near_range_suspect_thresh:
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_SUSPECT)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_NEAR_RANGE_SUSPECT

                if np.any(this_sig0_qual==products.QUAL_IND_DEGRADED):
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_DEGRADED)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_SIG0_QUAL_DEGRADED

                if np.any(this_class_qual==products.QUAL_IND_DEGRADED):
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_DEGRADED)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_CLASS_QUAL_DEGRADED

                if np.any(this_geo_qual==products.QUAL_IND_DEGRADED):
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_DEGRADED)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_GEOLOCATION_QUAL_DEGRADED

                if self.sig0[i][j] < self.sig0_bad_thresh_min \
                   or self.sig0[i][j] > self.sig0_bad_thresh_max:
                    self.sig0_qual[i][j] = max(
                        self.sig0_qual[i][j], products.QUAL_IND_BAD)
                    self.sig0_qual_bitwise[i][j] += \
                        products.QUAL_IND_VALUE_BAD

    def flag_large_karin_gaps(self, pixc):
        """ Flag large karin gaps"""
        LOGGER.info("flagging large karin gaps")

        pixc_line_qual_meanings = \
            pixc['pixel_cloud'].VARIABLES['pixc_line_qual']['flag_meanings'].split()
        pixc_line_qual_masks = \
            pixc['pixel_cloud'].VARIABLES['pixc_line_qual']['flag_masks']
        pixc_line_qual_ind_large_karin_gap = pixc_line_qual_masks[
            pixc_line_qual_meanings.index('large_karin_gap')]

        pixc_line_qual = pixc['pixel_cloud']['pixc_line_qual']
        pixc_tvp_index = pixc['pixel_cloud']['pixc_line_to_tvp'].astype(int)

        tvp_velocity_heading = pixc['tvp']['velocity_heading']
        tvp_xyz = np.row_stack((
            pixc['tvp']['x'], pixc['tvp']['y'], pixc['tvp']['z']))
        tvp_minus_y_antenna_xyz = np.row_stack((
            pixc['tvp']['minus_y_antenna_x'], pixc['tvp']['minus_y_antenna_y'],
            pixc['tvp']['minus_y_antenna_z']))
        tvp_plus_y_antenna_xyz = np.row_stack((
            pixc['tvp']['plus_y_antenna_x'], pixc['tvp']['plus_y_antenna_y'],
            pixc['tvp']['plus_y_antenna_z']))

        tvp_qual_mask = np.ma.zeros(tvp_velocity_heading.shape, dtype=bool)
        # Flag tvp qual as a gap if it is outside the pixc tvp index range
        # TODO: Check flagging at edges
        tvp_qual_mask[:min(pixc_tvp_index)] = True
        tvp_qual_mask[max(pixc_tvp_index)+1:] = True
        tvp_qual_mask[pixc_tvp_index[
            (pixc_line_qual & pixc_line_qual_ind_large_karin_gap) > 0]] = True

        transf = osr.CoordinateTransformation(self.input_crs, self.output_crs)

        large_karin_gap_polygons = []

        # Always flag the areas before and after tvp as karin gaps
        idx = 0
        group_tvp_xyz = np.column_stack([tvp_xyz[:,idx], tvp_xyz[:,idx]])
        group_tvp_minus_y_antenna_xyz = np.column_stack(
            [tvp_minus_y_antenna_xyz[:,idx],
             tvp_minus_y_antenna_xyz[:,idx]])
        group_tvp_plus_y_antenna_xyz = np.column_stack(
            [tvp_plus_y_antenna_xyz[:,idx],
             tvp_plus_y_antenna_xyz[:,idx]])
        group_tvp_velocity_heading = np.array(
            [tvp_velocity_heading[idx],
             tvp_velocity_heading[idx]])
        large_karin_gap_polygons.append(self.get_polygon_from_tvp(
            group_tvp_xyz, group_tvp_minus_y_antenna_xyz,
            group_tvp_plus_y_antenna_xyz, group_tvp_velocity_heading,
            products.POLYGON_EXTENT_DIST,
            alongtrack_start_buffer_dist=products.POLYGON_EXTENT_DIST))

        idx = len(tvp_velocity_heading)-1
        group_tvp_xyz = np.column_stack([tvp_xyz[:,idx], tvp_xyz[:,idx]])
        group_tvp_minus_y_antenna_xyz = np.column_stack(
            [tvp_minus_y_antenna_xyz[:,idx],
             tvp_minus_y_antenna_xyz[:,idx]])
        group_tvp_plus_y_antenna_xyz = np.column_stack(
            [tvp_plus_y_antenna_xyz[:,idx],
             tvp_plus_y_antenna_xyz[:,idx]])
        group_tvp_velocity_heading = np.array(
            [tvp_velocity_heading[idx],
             tvp_velocity_heading[idx]])
        large_karin_gap_polygons.append(self.get_polygon_from_tvp(
            group_tvp_xyz, group_tvp_minus_y_antenna_xyz,
            group_tvp_plus_y_antenna_xyz, group_tvp_velocity_heading,
            products.POLYGON_EXTENT_DIST,
            alongtrack_end_buffer_dist=products.POLYGON_EXTENT_DIST))

        # Flag any areas flagged as large gaps in the pixc_line qual
        for k, g in groupby(pixc_tvp_index, lambda x: tvp_qual_mask[x]==True):
            if k:
                group_idxs = list(g)
                start_idx = group_idxs[0]
                end_idx = group_idxs[-1]
                group_tvp_xyz = np.column_stack(
                    [tvp_xyz[:,start_idx],
                     tvp_xyz[:,end_idx]])
                group_tvp_minus_y_antenna_xyz = np.column_stack(
                    [tvp_minus_y_antenna_xyz[:,start_idx],
                     tvp_minus_y_antenna_xyz[:,end_idx]])
                group_tvp_plus_y_antenna_xyz = np.column_stack(
                    [tvp_plus_y_antenna_xyz[:,start_idx],
                     tvp_plus_y_antenna_xyz[:,end_idx]])
                group_tvp_velocity_heading = np.array(
                    [tvp_velocity_heading[start_idx],
                     tvp_velocity_heading[end_idx]])
                large_karin_gap_polygons.append(self.get_polygon_from_tvp(
                    group_tvp_xyz, group_tvp_minus_y_antenna_xyz,
                    group_tvp_plus_y_antenna_xyz, group_tvp_velocity_heading,
                    products.POLYGON_EXTENT_DIST))

        x_vec = np.linspace(self.x_min, self.x_max, self.size_x)
        y_vec = np.linspace(self.y_min, self.y_max, self.size_y)
        mask = np.zeros((self.size_y, self.size_x))
        for this_polygon_points in large_karin_gap_polygons:
            poly = Polygon(this_polygon_points)
            for i in range(0, self.size_y):
                for j in range(0, self.size_x):
                    if self.projection_type == 'geo': # Geo y coord is first
                        if Point((y_vec[i], x_vec[j])).intersects(poly):
                            mask[i][j] = True
                    else:
                        if Point((x_vec[j], y_vec[i])).intersects(poly):
                            mask[i][j] = True

        # Mask the datasets and flag
        wse_mask = np.logical_and(self.wse.mask, mask)
        water_area_mask = np.logical_and(self.water_area.mask, mask)
        sig0_mask = np.logical_and(self.sig0.mask, mask)

        self.wse_qual_bitwise[wse_mask] += products.QUAL_IND_LARGE_KARIN_GAP
        self.water_area_qual_bitwise[wse_mask] += products.QUAL_IND_LARGE_KARIN_GAP
        self.sig0_qual_bitwise[wse_mask] += products.QUAL_IND_LARGE_KARIN_GAP

    def flag_inner_swath(self, pixc):
        """ Flag inner swath"""
        LOGGER.info("flagging inner swath")

        tvp_velocity_heading = pixc['tvp']['velocity_heading']
        tvp_xyz = np.row_stack((
            pixc['tvp']['x'], pixc['tvp']['y'], pixc['tvp']['z']))
        tvp_minus_y_antenna_xyz = np.row_stack((
            pixc['tvp']['minus_y_antenna_x'], pixc['tvp']['minus_y_antenna_y'],
            pixc['tvp']['minus_y_antenna_z']))
        tvp_plus_y_antenna_xyz = np.row_stack((
            pixc['tvp']['plus_y_antenna_x'], pixc['tvp']['plus_y_antenna_y'],
            pixc['tvp']['plus_y_antenna_z']))

        inner_swath_polygon = self.get_polygon_from_tvp(
            tvp_xyz, tvp_minus_y_antenna_xyz, tvp_plus_y_antenna_xyz,
            tvp_velocity_heading, self.near_range_suspect_thresh,
            products.POLYGON_EXTENT_DIST, products.POLYGON_EXTENT_DIST)

        x_vec = np.linspace(self.x_min, self.x_max, self.size_x)
        y_vec = np.linspace(self.y_min, self.y_max, self.size_y)
        mask = np.zeros((self.size_y, self.size_x))
        poly = Polygon(inner_swath_polygon)
        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                if self.projection_type == 'geo': # Geo y coord is first
                    if Point((y_vec[i], x_vec[j])).intersects(poly):
                        mask[i][j] = True
                else:
                    if Point((x_vec[j], y_vec[i])).intersects(poly):
                        mask[i][j] = True

        # Mask the datasets and flag
        wse_mask = np.logical_and(self.wse.mask, mask)
        water_area_mask = np.logical_and(self.water_area.mask, mask)
        sig0_mask = np.logical_and(self.sig0.mask, mask)

        self.wse_qual_bitwise[wse_mask] += products.QUAL_IND_INNER_SWATH
        self.water_area_qual_bitwise[wse_mask] += products.QUAL_IND_INNER_SWATH
        self.sig0_qual_bitwise[wse_mask] += products.QUAL_IND_INNER_SWATH

    def get_polygon_from_tvp(self, sc_xyz, minus_y_antenna_xyz,
                             plus_y_antenna_xyz, sc_velocity_heading,
                             crosstrack_dist,
                             alongtrack_start_buffer_dist=None,
                             alongtrack_end_buffer_dist=None):
        """ Get polygon points from tvp points """
        LOGGER.info("getting polyon from tvp")

        transf = osr.CoordinateTransformation(self.input_crs, self.output_crs)

        polygon_minus_y = []
        polygon_plus_y = []
        for idx in range(sc_xyz.shape[1]):
            sc_llh = raster_crs.xyz2llh(sc_xyz[:,idx])
            minus_y_antenna_llh = raster_crs.xyz2llh(
                minus_y_antenna_xyz[:,idx])
            plus_y_antenna_llh = raster_crs.xyz2llh(
                plus_y_antenna_xyz[:,idx])
            minus_y_antenna_bearing = raster_crs.llh2bearing(
                sc_llh, minus_y_antenna_llh)
            plus_y_antenna_bearing = raster_crs.llh2bearing(
                sc_llh, plus_y_antenna_llh)

            minus_y_points_rad = \
                [raster_crs.terminal_loc_spherical(
                    sc_llh[0], sc_llh[1], crosstrack_dist,
                    minus_y_antenna_bearing)]
            plus_y_points_rad = \
                [raster_crs.terminal_loc_spherical(
                    sc_llh[0], sc_llh[1], crosstrack_dist,
                    plus_y_antenna_bearing)]

            if idx == 0 and alongtrack_start_buffer_dist is not None:
                sc_llh_buffer = raster_crs.terminal_loc_spherical(
                    sc_llh[0], sc_llh[1], alongtrack_start_buffer_dist,
                    np.deg2rad(np.mod(sc_velocity_heading[idx]-180, 360)))
                minus_y_points_rad = \
                    [raster_crs.terminal_loc_spherical(
                        sc_llh_buffer[0], sc_llh_buffer[1], crosstrack_dist,
                        minus_y_antenna_bearing)] + minus_y_points_rad
                plus_y_points_rad = \
                    [raster_crs.terminal_loc_spherical(
                        sc_llh_buffer[0], sc_llh_buffer[1], crosstrack_dist,
                        plus_y_antenna_bearing)] + plus_y_points_rad

            if idx == sc_xyz.shape[1]-1 and alongtrack_end_buffer_dist is not None:
                sc_llh_buffer = raster_crs.terminal_loc_spherical(
                    sc_llh[0], sc_llh[1], alongtrack_end_buffer_dist,
                    np.deg2rad(np.mod(sc_velocity_heading[idx], 360)))
                minus_y_points_rad = \
                    minus_y_points_rad \
                    + [raster_crs.terminal_loc_spherical(
                        sc_llh_buffer[0], sc_llh_buffer[1], crosstrack_dist,
                        minus_y_antenna_bearing)]
                plus_y_points_rad = \
                    plus_y_points_rad \
                    + [raster_crs.terminal_loc_spherical(
                        sc_llh_buffer[0], sc_llh_buffer[1], crosstrack_dist,
                        plus_y_antenna_bearing)]

            minus_y_points_deg = [
                np.rad2deg(point) for point in minus_y_points_rad]
            plus_y_points_deg = [
                np.rad2deg(point) for point in plus_y_points_rad]

            transf_minus_y_points = [point[:2] for point in
                                     transf.TransformPoints(minus_y_points_deg)]
            transf_plus_y_points = [point[:2] for point in
                                    transf.TransformPoints(plus_y_points_deg)]

            polygon_minus_y.extend(transf_minus_y_points)
            polygon_plus_y.extend(transf_plus_y_points)

        polygon = polygon_minus_y + polygon_plus_y[::-1]
        return polygon

    def aggregate_lat_lon(self, rasterization_mask):
        """ Aggregate latitude and longitude """
        LOGGER.info("aggregating latitude and longitude")

        x_vec = np.linspace(self.x_min, self.x_max, self.size_x)
        y_vec = np.linspace(self.y_min, self.y_max, self.size_y)

        transf = osr.CoordinateTransformation(self.output_crs, self.input_crs)

        self.latitude = np.ma.masked_all((self.size_y, self.size_x))
        self.longitude = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    # get the lat and lon if there are any good pixels at all
                    lon, lat = transf.TransformPoint(x_vec[j], y_vec[i])[:2]
                    self.latitude[i][j] = lon
                    self.longitude[i][j] = lat

    def aggregate_classification(self, pixc, rasterization_mask):
        """ Aggregate binary classification """
        LOGGER.info("aggregating classification")

        pixc_classif = pixc['pixel_cloud']['classification']

        self.classification = np.ma.masked_all((self.size_y, self.size_x))

        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                mask = rasterization_mask[self.proj_mapping[i][j]]
                if np.any(mask):
                    self.classification[i][j] = ag.simple(
                        pixc_classif[self.proj_mapping[i][j]][mask],
                        metric='mode')

    def build_product(self, populate_values=True, polygon_points=None):
        """ Assemble the product """
        LOGGER.info("building product")

        if self.projection_type == 'utm':
            if self.debug_flag:
                product = products.RasterUTMDebug()
            else:
                product = products.RasterUTM()
        elif self.projection_type == 'geo':
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

        if self.projection_type == 'utm':
            product.utm_zone_num = self.utm_zone
            product.mgrs_latitude_band = self.mgrs_band
            product.x_min = self.x_min
            product.x_max = self.x_max
            product.y_min = self.y_min
            product.y_max = self.y_max
            product['x'] = np.linspace(self.x_min, self.x_max, self.size_x)
            product['y'] = np.linspace(self.y_min, self.y_max, self.size_y)
            coordinate_system = self.output_crs
            product.VARIABLES['crs']['projected_crs_name'] = \
                coordinate_system.GetName()
            product.VARIABLES['crs']['false_northing'] = \
                coordinate_system.GetProjParm('false_northing')
            product.VARIABLES['crs']['longitude_of_central_meridian'] = \
                coordinate_system.GetProjParm('central_meridian')
        elif self.projection_type == 'geo':
            product.longitude_min = self.x_min
            product.longitude_max = self.x_max
            product.latitude_min = self.y_min
            product.latitude_max = self.y_max
            product['longitude'] = np.linspace(self.x_min,
                                               self.x_max,
                                               self.size_x)
            product['latitude'] = np.linspace(self.y_min,
                                              self.y_max,
                                              self.size_y)
            coordinate_system = raster_crs.wgs84_crs()
        else:
            raise RasterUsageException(
                'Unknown projection type: {}'.format(self.projection_type))

        product.VARIABLES['crs']['crs_wkt'] = coordinate_system.ExportToWkt()
        product.VARIABLES['crs']['spatial_ref'] = \
            product.VARIABLES['crs']['crs_wkt']

        if populate_values:
            if self.projection_type == 'utm':
                product['longitude'] = self.longitude
                product['latitude'] = self.latitude

            product['illumination_time'] = self.illumination_time
            product['illumination_time_tai'] = self.illumination_time_tai
            product.VARIABLES['illumination_time']['tai_utc_difference'] = \
                self.tai_utc_difference
            product.VARIABLES['illumination_time']['leap_second'] = \
                self.leap_second
            product['wse'] = self.wse
            product['wse_qual_bitwise'] = self.wse_qual_bitwise
            product['wse_qual'] = self.wse_qual
            product['wse_uncert'] = self.wse_u
            product['water_area'] = self.water_area
            product['water_area_qual_bitwise'] = self.water_area_qual_bitwise
            product['water_area_qual'] = self.water_area_qual
            product['water_area_uncert'] = self.water_area_u
            product['water_frac'] = self.water_frac
            product['water_frac_uncert'] = self.water_frac_u
            product['sig0'] = self.sig0
            product['sig0_qual_bitwise'] = self.sig0_qual_bitwise
            product['sig0_qual'] = self.sig0_qual
            product['sig0_uncert'] = self.sig0_u
            product['inc'] = self.inc
            product['cross_track'] = self.cross_track
            product['n_wse_pix'] = self.n_wse_pix
            product['n_water_area_pix'] = self.n_water_area_pix
            product['n_sig0_pix'] = self.n_sig0_pix
            product['n_other_pix'] = self.n_other_pix
            product['dark_frac'] = self.dark_frac
            product['ice_clim_flag'] = self.ice_clim_flag
            product['ice_dyn_flag'] = self.ice_dyn_flag
            product['layover_impact'] = self.layover_impact
            product['sig0_cor_atmos_model'] = self.sig0_cor_atmos_model
            product['height_cor_xover'] = self.height_cor_xover
            product['geoid'] = self.geoid
            product['solid_earth_tide'] = self.solid_earth_tide
            product['load_tide_fes'] = self.load_tide_fes
            product['load_tide_got'] = self.load_tide_got
            product['pole_tide'] = self.pole_tide
            product['model_dry_tropo_cor'] = self.model_dry_tropo_cor
            product['model_wet_tropo_cor'] = self.model_wet_tropo_cor
            product['iono_cor_gim_ka'] = self.iono_cor_gim_ka

            if self.debug_flag:
                product['classification'] = self.classification

        # Crop the product to the desired bounds
        if polygon_points is not None:
            product.crop_to_bounds(polygon_points)

        return product
