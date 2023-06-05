'''
Copyright (c) 2023-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author (s): Alexander Corben (JPL)
'''

import numpy as np
import SWOTWater.aggregate as ag
import SWOTRaster.products as products
import SWOTRaster.raster_crs as raster_crs

from osgeo import osr
from itertools import islice
from SWOTWater.constants import AGG_CLASSES
from SWOTRaster.errors import RasterUsageException

HEIGHT_STD_DEWEIGHT_VAL = 1.0e5
ICE_FLAG_PARTIAL_COVER_FLAG_VALUE = 1

def fn_it(iterable, fn):
    """ Calls fn for each value in iterable """
    it = iter(iterable)
    for x in it:
        yield fn(x)

def fn_map(fn, args):
    """ Calls function with iterable of argument iterables """
    return tuple(fn_star(fn, arglist) for arglist in zip(*args))

def fn_star(fn, args):
    """ Calls function with iterable of arguments """
    return fn(*args)

def args_mask(*args):
    """ Get mask where all input masks are False (unmasked) """
    return np.logical_not(np.logical_or.reduce(
        ([np.ma.getmaskarray(arg) for arg in args])))

def simple_masked(data, mask, metric='mean', fill_value=np.nan):
    """ Get simple aggregation with masking """
    mask = np.logical_and(mask, args_mask(data))
    if np.any(mask):
        return ag.simple(data[mask], metric=metric)
    else:
        return fill_value

def calc_height_std(phase_noise_std, dh_dphi):
    """ calculate height std """
    height_std = np.abs(phase_noise_std * dh_dphi)
    # Set bad pix height std to high number to deweight
    # instead of giving infs/nans
    height_std[height_std<=0] = HEIGHT_STD_DEWEIGHT_VAL
    height_std[np.isinf(height_std)] = HEIGHT_STD_DEWEIGHT_VAL
    height_std[np.isnan(height_std)] = HEIGHT_STD_DEWEIGHT_VAL
    return height_std

def height_weighted_mean_masked(data, phase_noise_std, dh_dphi, mask,
                                height_agg_method='weight', fill_value=np.nan):
    """ Get height weighted mean with masking """
    mask = np.logical_and(mask, args_mask(data, phase_noise_std, dh_dphi))
    if np.any(mask):
        height_std = calc_height_std(phase_noise_std, dh_dphi)
        return ag.height_only(
            data, mask, height_std, method=height_agg_method)[0]
    else:
        return fill_value

def aggregate_cross_track_and_incidence_angle(
        pixc_cross_track, pixc_incidence_angle, mask):
    """ Aggregate cross track and incidence angle """
    cross_track = simple_masked(pixc_cross_track, mask, metric='mean')
    inc = simple_masked(pixc_incidence_angle, mask, metric='mean')
    n_other_pix = int(ag.simple(mask, metric='sum'))
    return cross_track, inc, n_other_pix

def aggregate_illumination_time(
        pixc_illumination_time, pixc_illumination_time_tai, mask):
    """ Aggregate illumination time """
    illumination_time = simple_masked(
        pixc_illumination_time, mask, metric='mean')
    illumination_time_tai = simple_masked(
        pixc_illumination_time_tai, mask, metric='mean')
    return illumination_time, illumination_time_tai

def aggregate_px_latlon(x, y, mask, crs_wkt):
    """ Aggregate pixel lat/lon coordinates """
    if np.any(mask):
        from_crs = osr.SpatialReference()
        from_crs.ImportFromWkt(crs_wkt)
        to_crs = raster_crs.wgs84_crs()
        transf = osr.CoordinateTransformation(from_crs, to_crs)
        px_lat, px_lon = transf.TransformPoint(x, y)[:2]
    else:
        px_lat = np.nan
        px_lon = np.nan

    return px_lat, px_lon

def aggregate_wse_corrections(
        pixc_height_cor_xover, pixc_geoid, pixc_solid_earth_tide,
        pixc_load_tide_fes, pixc_load_tide_got, pixc_pole_tide,
        pixc_model_dry_tropo_cor, pixc_model_wet_tropo_cor,
        pixc_iono_cor_gim_ka, pixc_dh_dphi, pixc_phase_noise_std,
        mask, height_agg_method='weight'):
    """ Aggregate wse geophysical corrections """
    height_cor_xover = height_weighted_mean_masked(
        pixc_height_cor_xover, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    geoid = height_weighted_mean_masked(
        pixc_geoid, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    solid_earth_tide = height_weighted_mean_masked(
        pixc_solid_earth_tide, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    load_tide_fes = height_weighted_mean_masked(
        pixc_load_tide_fes, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    load_tide_got = height_weighted_mean_masked(
        pixc_load_tide_got, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    pole_tide = height_weighted_mean_masked(
        pixc_pole_tide, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    model_dry_tropo_cor = height_weighted_mean_masked(
        pixc_model_dry_tropo_cor, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    model_wet_tropo_cor = height_weighted_mean_masked(
        pixc_model_wet_tropo_cor, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    iono_cor_gim_ka = height_weighted_mean_masked(
        pixc_iono_cor_gim_ka, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    return (height_cor_xover, geoid, solid_earth_tide, load_tide_fes,
            load_tide_got, pole_tide, model_dry_tropo_cor, model_wet_tropo_cor,
            iono_cor_gim_ka)

def apply_wse_corrections(
        height, geoid, solid_earth_tide, load_tide_fes, pole_tide):
    """ Apply geophysical corrections to height to get wse """
    return height - (geoid + solid_earth_tide + load_tide_fes + pole_tide)

def aggregate_height(
        pixc_height, pixc_num_rare_looks,
        pixc_num_med_looks, pixc_power_plus_y, pixc_power_minus_y,
        pixc_dh_dphi, pixc_dlat_dphi, pixc_dlon_dphi, pixc_phase_noise_std,
        flat_ifgram, mask, looks_to_efflooks, height_agg_method='weight'):
    """ Aggregate height and associated uncertainties """
    mask = np.logical_and(mask,
        args_mask(pixc_height, pixc_num_rare_looks, pixc_num_med_looks,
                  pixc_power_plus_y, pixc_power_minus_y, pixc_dh_dphi,
                  pixc_dlat_dphi, pixc_dlon_dphi, pixc_phase_noise_std,
                  flat_ifgram))
    if np.any(mask):
        pixc_height_std = calc_height_std(pixc_phase_noise_std, pixc_dh_dphi)
        grid_height = ag.height_with_uncerts(
            pixc_height, mask, pixc_num_rare_looks,
            pixc_num_med_looks, flat_ifgram, pixc_power_minus_y,
            pixc_power_plus_y, looks_to_efflooks, pixc_dh_dphi,
            pixc_dlat_dphi, pixc_dlon_dphi, pixc_height_std,
            method=height_agg_method)
        height = grid_height[0]
        height_u = grid_height[2]
    else:
        height = np.nan
        height_u = np.nan

    return height, height_u

def aggregate_water_area(
        pixc_pixel_area, pixc_water_frac, pixc_water_frac_uncert,
        pixc_darea_dheight, pixc_pfd, pixc_pmd, pixc_classif, px_lat,
        mask, projection_type, resolution,
        interior_water_klasses=AGG_CLASSES['interior_water_klasses'],
        water_edge_klasses=AGG_CLASSES['water_edge_klasses'],
        land_edge_klasses=AGG_CLASSES['land_edge_klasses'],
        dark_water_klasses=AGG_CLASSES['dark_water_klasses'],
        area_agg_method='composite'):
    """ Aggregate water area, water fraction and associated uncertainties """
    mask = np.logical_and(mask,
        args_mask(pixc_pixel_area, pixc_water_frac, pixc_water_frac_uncert,
                  pixc_darea_dheight, pixc_pfd, pixc_pmd, pixc_classif))
    if np.any(mask):
        grid_area = ag.area_with_uncert(
            pixc_pixel_area, pixc_water_frac, pixc_water_frac_uncert,
            pixc_darea_dheight, pixc_classif, pixc_pfd, pixc_pmd,
            mask, method=area_agg_method,
            interior_water_klasses=interior_water_klasses,
            water_edge_klasses=water_edge_klasses,
            land_edge_klasses=land_edge_klasses,
            dark_water_klasses=dark_water_klasses)
        water_area = grid_area[0]
        water_area_u = grid_area[1]

        if projection_type=='utm':
            pixel_area = resolution**2
        elif projection_type=='geo':
            pixel_area = raster_crs.wgs84_px_area(px_lat, resolution)
        else:
            raise RasterUsageException(
                'Unknown projection type: {}'.format(projection_type))

        water_frac = water_area/pixel_area
        water_frac_u = water_area_u/pixel_area
    else:
        water_area = np.nan
        water_area_u = np.nan
        water_frac = np.nan
        water_frac_u = np.nan

    return water_area, water_area_u, water_frac, water_frac_u

def aggregate_sig0_corrections(pixc_sig0_cor_atmos_model, mask):
    """ Aggregate sig0 geophysical corrections """
    sig0_cor_atmos_model = simple_masked(
        pixc_sig0_cor_atmos_model, mask, metric='mean')

    return sig0_cor_atmos_model

def aggregate_sig0(pixc_sig0, pixc_sig0_uncert, mask, sig0_agg_method='rare'):
    """ Aggregate sigma0 """
    mask = np.logical_and(mask, args_mask(pixc_sig0, pixc_sig0_uncert))
    if np.any(mask):
        grid_sig0 = ag.sig0_with_uncerts(
            pixc_sig0, mask, pixc_sig0_uncert, method=sig0_agg_method)
        sig0 = grid_sig0[0]
        sig0_u = grid_sig0[2]
    else:
        sig0 = np.nan
        sig0_u = np.nan

    return sig0, sig0_u

def aggregate_dark_frac(
        pixc_classif, pixc_pixel_area, pixc_water_frac, mask,
        interior_water_klasses=AGG_CLASSES['interior_water_klasses'],
        water_edge_klasses=AGG_CLASSES['water_edge_klasses'],
        land_edge_klasses=AGG_CLASSES['land_edge_klasses'],
        dark_water_klasses=AGG_CLASSES['dark_water_klasses'],
        area_agg_method='composite'):
    """ Aggregate dark water fraction """
    mask = np.logical_and(mask,
        args_mask(pixc_classif, pixc_pixel_area, pixc_water_frac))
    if np.any(mask):
        pixc_dark_mask = np.logical_and(mask,
            np.isin(pixc_classif, dark_water_klasses))
        if np.any(pixc_dark_mask):
            dark_area = ag.simple(pixc_pixel_area[pixc_dark_mask], metric='sum')
            total_area, _ = ag.area_only(
                pixc_pixel_area, pixc_water_frac, pixc_classif, mask,
                method=area_agg_method,
                interior_water_klasses=interior_water_klasses,
                water_edge_klasses=water_edge_klasses,
                land_edge_klasses=land_edge_klasses,
                dark_water_klasses=dark_water_klasses)
            if total_area==0:
                dark_frac = 0
            else:
                dark_frac = dark_area/total_area
        else:
            dark_frac = 0
    else:
        dark_frac = np.nan

    return dark_frac

def aggregate_ice_flag(pixc_ice_flag, mask):
    """ Aggregate ice flag """
    mask = np.logical_and(mask, args_mask(pixc_ice_flag))
    if np.any(mask):
        valid_ice_flag = pixc_ice_flag[mask]
        min_flag_val = np.min(valid_ice_flag)
        # If all flags are the same, then we return that value
        if np.all(valid_ice_flag==min_flag_val):
            ice_flag_out = min_flag_val
        else: # Otherwise, return partial cover value
            ice_flag_out = ICE_FLAG_PARTIAL_COVER_FLAG_VALUE
    else:
        ice_flag_out = np.nan

    return ice_flag_out

def aggregate_layover_impact(
        pixc_layover_impact, pixc_dh_dphi, pixc_phase_noise_std,
        mask, height_agg_method='weight'):
    """ Aggregate layover impact """
    layover_impact = height_weighted_mean_masked(
        pixc_layover_impact, pixc_phase_noise_std, pixc_dh_dphi, mask,
        height_agg_method=height_agg_method)
    return layover_impact

def aggregate_wse_qual(
        wse, wse_u, cross_track, pixc_class_qual, pixc_geo_qual,
        pixc_bright_land_flag, mask, wse_uncert_suspect_thresh,
        num_wse_pix_suspect_thresh, near_range_suspect_thresh,
        far_range_suspect_thresh, wse_bad_thresh_min, wse_bad_thresh_max):
    """ Aggregate wse qual """
    if np.any(mask):
        # Default to good
        wse_qual = products.QUAL_IND_GOOD
        wse_qual_bitwise = products.QUAL_IND_GOOD
        n_wse_pix = int(ag.simple(mask, metric='sum'))

        if np.any(pixc_class_qual[mask]==products.QUAL_IND_SUSPECT):
            wse_qual = max(wse_qual, products.QUAL_IND_SUSPECT)
            wse_qual_bitwise += products.QUAL_IND_CLASS_QUAL_SUSPECT

        if np.any(pixc_geo_qual[mask]==products.QUAL_IND_SUSPECT):
            wse_qual = max(wse_qual, products.QUAL_IND_SUSPECT)
            wse_qual_bitwise += products.QUAL_IND_GEOLOCATION_QUAL_SUSPECT

        if wse_u > wse_uncert_suspect_thresh:
            wse_qual = max(wse_qual, products.QUAL_IND_SUSPECT)
            wse_qual_bitwise += products.QUAL_IND_LARGE_UNCERT_SUSPECT

        if np.any(pixc_bright_land_flag[mask]):
            wse_qual = max(wse_qual, products.QUAL_IND_SUSPECT)
            wse_qual_bitwise += products.QUAL_IND_BRIGHT_LAND

        if n_wse_pix < num_wse_pix_suspect_thresh:
            wse_qual = max(wse_qual, products.QUAL_IND_SUSPECT)
            wse_qual_bitwise += products.QUAL_IND_FEW_PIXELS

        if abs(cross_track) > far_range_suspect_thresh:
            wse_qual = max(wse_qual, products.QUAL_IND_SUSPECT)
            wse_qual_bitwise += products.QUAL_IND_FAR_RANGE_SUSPECT

        if abs(cross_track) < near_range_suspect_thresh:
            wse_qual = max(wse_qual, products.QUAL_IND_SUSPECT)
            wse_qual_bitwise += products.QUAL_IND_NEAR_RANGE_SUSPECT

        if np.any(pixc_class_qual[mask]==products.QUAL_IND_DEGRADED):
            wse_qual = max(wse_qual, products.QUAL_IND_DEGRADED)
            wse_qual_bitwise += products.QUAL_IND_CLASS_QUAL_DEGRADED

        if np.any(pixc_geo_qual[mask]==products.QUAL_IND_DEGRADED):
            wse_qual = max(wse_qual, products.QUAL_IND_DEGRADED)
            wse_qual_bitwise += products.QUAL_IND_GEOLOCATION_QUAL_DEGRADED

        if wse < wse_bad_thresh_min \
           or wse > wse_bad_thresh_max:
            wse_qual = max(wse_qual, products.QUAL_IND_BAD)
            wse_qual_bitwise += products.QUAL_IND_VALUE_BAD
    else:
        wse_qual = products.QUAL_IND_BAD
        wse_qual_bitwise = products.QUAL_IND_NO_PIXELS \
                           + products.QUAL_IND_FEW_PIXELS
        n_wse_pix = 0

    return wse_qual, wse_qual_bitwise, n_wse_pix

def aggregate_water_area_qual(
        water_frac, water_frac_u, cross_track, pixc_class_qual, pixc_geo_qual,
        pixc_bright_land_flag, pixc_water_frac, mask,
        pixc_water_frac_suspect_thresh, water_frac_uncert_suspect_thresh,
        num_water_area_pix_suspect_thresh, near_range_suspect_thresh,
        far_range_suspect_thresh, water_frac_bad_thresh_min,
        water_frac_bad_thresh_max):
    """ Aggregate water area qual """
    if np.any(mask):
        # Default to good
        water_area_qual = products.QUAL_IND_GOOD
        water_area_qual_bitwise = products.QUAL_IND_GOOD
        n_water_area_pix = int(ag.simple(mask, metric='sum'))

        if np.any(pixc_class_qual[mask]==products.QUAL_IND_SUSPECT):
            water_area_qual = max(water_area_qual, products.QUAL_IND_SUSPECT)
            water_area_qual_bitwise += products.QUAL_IND_CLASS_QUAL_SUSPECT

        if np.any(pixc_geo_qual[mask]==products.QUAL_IND_SUSPECT):
            water_area_qual = max(water_area_qual, products.QUAL_IND_SUSPECT)
            water_area_qual_bitwise += products.QUAL_IND_GEOLOCATION_QUAL_SUSPECT

        if np.any(np.abs(pixc_water_frac[mask]) > pixc_water_frac_suspect_thresh):
            water_area_qual = max(water_area_qual, products.QUAL_IND_SUSPECT)
            water_area_qual_bitwise += products.QUAL_IND_WATER_FRACTION_SUSPECT

        if water_frac_u > water_frac_uncert_suspect_thresh:
            water_area_qual = max(water_area_qual, products.QUAL_IND_SUSPECT)
            water_area_qual_bitwise += products.QUAL_IND_LARGE_UNCERT_SUSPECT

        if np.any(pixc_bright_land_flag[mask]):
            water_area_qual = max(water_area_qual, products.QUAL_IND_SUSPECT)
            water_area_qual_bitwise += products.QUAL_IND_BRIGHT_LAND

        if n_water_area_pix < num_water_area_pix_suspect_thresh:
            water_area_qual = max(water_area_qual, products.QUAL_IND_SUSPECT)
            water_area_qual_bitwise += products.QUAL_IND_FEW_PIXELS

        if abs(cross_track) > far_range_suspect_thresh:
            water_area_qual = max(water_area_qual, products.QUAL_IND_SUSPECT)
            water_area_qual_bitwise += products.QUAL_IND_FAR_RANGE_SUSPECT

        if abs(cross_track) < near_range_suspect_thresh:
            water_area_qual = max(water_area_qual, products.QUAL_IND_SUSPECT)
            water_area_qual_bitwise += products.QUAL_IND_NEAR_RANGE_SUSPECT

        if np.any(pixc_class_qual[mask]==products.QUAL_IND_DEGRADED):
            water_area_qual = max(water_area_qual, products.QUAL_IND_DEGRADED)
            water_area_qual_bitwise += products.QUAL_IND_CLASS_QUAL_DEGRADED

        if np.any(pixc_geo_qual[mask]==products.QUAL_IND_DEGRADED):
            water_area_qual = max(water_area_qual, products.QUAL_IND_DEGRADED)
            water_area_qual_bitwise += products.QUAL_IND_GEOLOCATION_QUAL_DEGRADED

        if water_frac < water_frac_bad_thresh_min \
           or water_frac > water_frac_bad_thresh_max:
            water_area_qual = max(water_area_qual, products.QUAL_IND_BAD)
            water_area_qual_bitwise += products.QUAL_IND_VALUE_BAD
    else:
        n_water_area_pix = 0
        water_area_qual = products.QUAL_IND_BAD
        water_area_qual_bitwise = products.QUAL_IND_NO_PIXELS \
                                  + products.QUAL_IND_FEW_PIXELS

    return water_area_qual, water_area_qual_bitwise, n_water_area_pix

def aggregate_sig0_qual(
        sig0, sig0_u, cross_track, pixc_sig0_qual, pixc_class_qual,
        pixc_geo_qual, pixc_bright_land_flag, mask, sig0_uncert_suspect_thresh,
        num_sig0_pix_suspect_thresh, near_range_suspect_thresh,
        far_range_suspect_thresh, sig0_bad_thresh_min, sig0_bad_thresh_max):
    """ Aggregate sig0 qual """
    if np.any(mask):
        # Default to good
        sig0_qual = products.QUAL_IND_GOOD
        sig0_qual_bitwise = products.QUAL_IND_GOOD
        n_sig0_pix = int(ag.simple(mask, metric='sum'))

        if np.any(pixc_sig0_qual[mask]==products.QUAL_IND_SUSPECT):
            sig0_qual = max(sig0_qual, products.QUAL_IND_SUSPECT)
            sig0_qual_bitwise += products.QUAL_IND_SIG0_QUAL_SUSPECT

        if np.any(pixc_class_qual[mask]==products.QUAL_IND_SUSPECT):
            sig0_qual = max(sig0_qual, products.QUAL_IND_SUSPECT)
            sig0_qual_bitwise += products.QUAL_IND_CLASS_QUAL_SUSPECT

        if np.any(pixc_geo_qual[mask]==products.QUAL_IND_SUSPECT):
            sig0_qual = max(sig0_qual, products.QUAL_IND_SUSPECT)
            sig0_qual_bitwise += products.QUAL_IND_GEOLOCATION_QUAL_SUSPECT

        if sig0_u > sig0_uncert_suspect_thresh:
            sig0_qual = max(sig0_qual, products.QUAL_IND_SUSPECT)
            sig0_qual_bitwise += products.QUAL_IND_LARGE_UNCERT_SUSPECT

        if np.any(pixc_bright_land_flag[mask]):
            sig0_qual = max(sig0_qual, products.QUAL_IND_SUSPECT)
            sig0_qual_bitwise += products.QUAL_IND_BRIGHT_LAND

        if n_sig0_pix < num_sig0_pix_suspect_thresh:
            sig0_qual = max(sig0_qual, products.QUAL_IND_SUSPECT)
            sig0_qual_bitwise += products.QUAL_IND_FEW_PIXELS

        if abs(cross_track) > far_range_suspect_thresh:
            sig0_qual = max(sig0_qual, products.QUAL_IND_SUSPECT)
            sig0_qual_bitwise += products.QUAL_IND_FAR_RANGE_SUSPECT

        if abs(cross_track) < near_range_suspect_thresh:
            sig0_qual = max(sig0_qual, products.QUAL_IND_SUSPECT)
            sig0_qual_bitwise += products.QUAL_IND_NEAR_RANGE_SUSPECT

        if np.any(pixc_sig0_qual[mask]==products.QUAL_IND_DEGRADED):
            sig0_qual = max(sig0_qual, products.QUAL_IND_DEGRADED)
            sig0_qual_bitwise += products.QUAL_IND_SIG0_QUAL_DEGRADED

        if np.any(pixc_class_qual[mask]==products.QUAL_IND_DEGRADED):
            sig0_qual = max(sig0_qual, products.QUAL_IND_DEGRADED)
            sig0_qual_bitwise += products.QUAL_IND_CLASS_QUAL_DEGRADED

        if np.any(pixc_geo_qual[mask]==products.QUAL_IND_DEGRADED):
            sig0_qual = max(sig0_qual, products.QUAL_IND_DEGRADED)
            sig0_qual_bitwise += products.QUAL_IND_GEOLOCATION_QUAL_DEGRADED

        if sig0 < sig0_bad_thresh_min \
           or sig0 > sig0_bad_thresh_max:
            sig0_qual = max(sig0_qual, products.QUAL_IND_BAD)
            sig0_qual_bitwise += products.QUAL_IND_VALUE_BAD
    else:
        sig0_qual = products.QUAL_IND_BAD
        sig0_qual_bitwise = products.QUAL_IND_NO_PIXELS \
                            + products.QUAL_IND_FEW_PIXELS
        n_sig0_pix = 0

    return sig0_qual, sig0_qual_bitwise, n_sig0_pix

def aggregate_classification(pixc_classif, mask):
    """ Aggregate binary classification """
    classification = simple_masked(pixc_classif, mask, metric='mode')
    return classification
