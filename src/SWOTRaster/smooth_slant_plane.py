'''
Copyright (c) 2024-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

'''
import logging
import multiprocessing
from functools import partial
from datetime import datetime

import bottleneck
import numpy as np
from scipy.ndimage import generic_filter
from SWOTWater.constants import PIXC_CLASSES

from SWOTRaster import products
from SWOTRaster.raster_agg import fn_star
from SWOTRaster.errors import RasterUsageException

DEFAULT_MAX_CHUNK_SHAPE = (2000, 2000)
DEFAULT_SMOOTHING_FILTER_SHAPE = (5, 13)
DEFAULT_GOOD_CLASSES = [PIXC_CLASSES['water_near_land'],
                        PIXC_CLASSES['open_water']]
DEFAULT_SUS_CLASSES = [PIXC_CLASSES['land_near_water'],
                       PIXC_CLASSES['dark_water'],
                       PIXC_CLASSES['low_coh_water_near_land'],
                       PIXC_CLASSES['low_coh_water'],
                       PIXC_CLASSES['dark_water_near_land'],
                       PIXC_CLASSES['dark_water_legacy']]

AZ_LOOKS_TOL = 1e-4
RNG_SPACING_TOL = 1e-4

LOGGER = logging.getLogger(__name__)


def smooth_slant_plane(
        scene_pixc, var_name='height',
        smoothing_filter_shape=DEFAULT_SMOOTHING_FILTER_SHAPE,
        good_klasses=DEFAULT_GOOD_CLASSES,
        sus_klasses=DEFAULT_SUS_CLASSES,
        class_qual_suspect=0,
        class_qual_degraded=0,
        class_qual_bad=0,
        geo_qual_suspect=0,
        geo_qual_degraded=0,
        geo_qual_bad=0,
        use_bright_land=True,
        specular_not_intersecting_prior_thresh=0.2,
        method='composite_with_sus_classes',
        max_chunk_shape=DEFAULT_MAX_CHUNK_SHAPE,
        max_worker_processes=1):
    """ Smoothes in slant plane """
    LOGGER.info('Smoothing %s in slant plane', var_name)
    if method not in ['simple', 'composite', 'composite_with_sus_classes']:
        raise RasterUsageException(
            'Unknown slant plane smoothing method: {}'.format(method))

    # If all input pixels are masked, return fully masked array
    var = scene_pixc['pixel_cloud'][var_name]
    var_out = np.ma.masked_all_like(var)
    if len(var) == 0 or np.all(var.mask):
        return var_out

    smoothing_footprint = np.ones(
        (smoothing_filter_shape[0], smoothing_filter_shape[1]), dtype=bool)

    # Get input variables
    classif = scene_pixc['pixel_cloud']['classification']
    classif_qual = scene_pixc.get_summary_qual_flag(
        'classification_qual', class_qual_suspect,
        class_qual_degraded, class_qual_bad)
    geolocation_qual = scene_pixc.get_summary_qual_flag(
        'geolocation_qual', geo_qual_suspect,
        geo_qual_degraded, geo_qual_bad)
    bright_land_flag = scene_pixc['pixel_cloud']['bright_land_flag'].filled(0)
    specular_ringing_flag = scene_pixc.get_qual_flag_bit(
        'classification_qual', 'specular_ringing_degraded')
    no_prior_water = scene_pixc['pixel_cloud']['prior_water_prob'].filled(0) \
        < specular_not_intersecting_prior_thresh
    specular_intersecting_prior = np.logical_and(
        specular_ringing_flag, np.logical_not(no_prior_water))
    specular_not_intersecting_prior = np.logical_and(
        specular_ringing_flag, no_prior_water)
    specular_ringing_qual = np.full(
        specular_ringing_flag.shape, products.QUAL_IND_GOOD)
    specular_ringing_qual[specular_intersecting_prior] = \
        products.QUAL_IND_SUSPECT
    specular_ringing_qual[specular_not_intersecting_prior] = \
        products.QUAL_IND_DEGRADED

    line_idx = scene_pixc['pixel_cloud']['pixc_line_index']
    tile_idx = scene_pixc['pixel_cloud']['pixc_line_to_tile'][line_idx]
    swath_side = scene_pixc['pixel_cloud']['tile_swath_side'][tile_idx]

    # Calculate per-tile azimuth and range offsets
    range_offsets = np.zeros(
        scene_pixc['pixel_cloud']['tile_tile_name'].shape, dtype=int)
    azimuth_offsets = np.zeros(
        scene_pixc['pixel_cloud']['tile_tile_name'].shape, dtype=int)
    for this_side in ['L', 'R']:
        tile_side_mask = np.char.lower(
            scene_pixc['pixel_cloud']['tile_swath_side']) == this_side.lower()
        range_offsets[tile_side_mask] = get_range_offsets(
            scene_pixc, tile_mask=tile_side_mask)
        azimuth_offsets[tile_side_mask] = get_azimuth_offsets(
            scene_pixc, smoothing_filter_shape[0],
            tile_mask=tile_side_mask)

    # Recompute indices from offsets
    recomputed_az_idx = \
        azimuth_offsets[tile_idx] + scene_pixc['pixel_cloud']['azimuth_index']
    recomputed_rng_idx = \
        range_offsets[tile_idx] + scene_pixc['pixel_cloud']['range_index']

    # Split into chunks and smooth, with multiprocessing if commanded
    smooth_fn = partial(smooth_chunk_and_mask,
                        smoothing_footprint=smoothing_footprint,
                        good_klasses=good_klasses,
                        sus_klasses=sus_klasses,
                        use_bright_land=use_bright_land,
                        method=method)
    _smooth_fn = partial(fn_star, smooth_fn)
    if max_worker_processes > 1:
        sz_az = np.max(recomputed_az_idx)-np.min(recomputed_az_idx)+1
        sz_rng = np.max(recomputed_rng_idx)-np.min(recomputed_rng_idx)+1
        chunk_shape = (
            int(max(1, min(np.sqrt((sz_az*sz_rng)/(max_worker_processes*2)),
                           max_chunk_shape[0]))),
            int(max(1, min(np.sqrt((sz_az*sz_rng)/(max_worker_processes*2)),
                           max_chunk_shape[1]))))
        with multiprocessing.get_context('spawn').Pool(
                processes=max_worker_processes) as pool:
            results = list(pool.imap(_smooth_fn, chunk_slant_map(
                var, recomputed_az_idx, recomputed_rng_idx, classif,
                classif_qual, geolocation_qual, bright_land_flag,
                specular_ringing_qual, swath_side, chunk_shape,
                (2*smoothing_footprint.shape[0],
                 2*smoothing_footprint.shape[1]))))
    else:
        chunk_shape = max_chunk_shape
        results = [_smooth_fn(arglist) for arglist in chunk_slant_map(
            var, recomputed_az_idx, recomputed_rng_idx, classif,
            classif_qual, geolocation_qual, bright_land_flag,
            specular_ringing_qual, swath_side, chunk_shape,
            (2*smoothing_footprint.shape[0], 2*smoothing_footprint.shape[1]))]

    for data, sort_idx in results:
        var_out[sort_idx] = data

    return var_out


def get_range_offsets(scene_pixc, tile_mask=None):
    """ Get range offset for each tile in scene_pixc """
    LOGGER.info('Getting range offsets')
    # If no tiles, return empty array
    if not np.any(tile_mask):
        return np.array([])

    near_range = scene_pixc['pixel_cloud']['tile_near_range'][tile_mask]
    nominal_slant_range_spacing = \
        scene_pixc['pixel_cloud']['tile_nominal_slant_range_spacing'][
            tile_mask]
    range_offsets = np.round((near_range - np.min(near_range))
                             / nominal_slant_range_spacing).astype(int)
    return range_offsets


def get_azimuth_offsets(scene_pixc, max_offset, tile_mask=None):
    """ Get azimuth offset for each tile in scene_pixc """
    LOGGER.info('Getting azimuth offsets')
    # If tile_mask is None, use all tiles
    if tile_mask is None:
        tile_mask = np.ones(scene_pixc['pixel_cloud']['tile_tile_name'].shape,
                            dtype=bool)

    # If no tiles, return empty array
    if not np.any(tile_mask):
        return np.array([])

    # Get mask of pixc lines that are not in overlap region
    pixc_line_in_tile = np.logical_not(scene_pixc.get_qual_flag_bit(
        'pixc_line_qual', 'not_in_tile'))

    # Sort tiles by granule start time
    all_tiles_time_granule_start = np.array(
        [datetime.strptime(time_granule_start, products.DATETIME_FORMAT_STR)
         for time_granule_start
         in scene_pixc['pixel_cloud']['tile_time_granule_start']])
    sort_idx = np.argsort(all_tiles_time_granule_start[tile_mask])
    tiles_idx = np.arange(len(all_tiles_time_granule_start))[tile_mask]
    outputs_idx = np.arange(len(tiles_idx))

    # Get azimuth offset for each tile
    azimuth_offsets = np.zeros(tiles_idx.shape, dtype=int)
    prev_last_line = None
    prev_last_record_counter = None
    prev_num_azimuth_looks = None
    prev_nominal_slant_range_spacing = None
    for tile_idx, output_idx in zip(
            tiles_idx[sort_idx], outputs_idx[sort_idx]):
        tile_pixc_line_mask = \
            scene_pixc['pixel_cloud']['pixc_line_to_tile'] == tile_idx
        tile_pixc_lines_in_tile = np.logical_and(
            tile_pixc_line_mask, pixc_line_in_tile)
        num_lines = np.sum(tile_pixc_lines_in_tile)

        # This tile is empty - this offset shouldn't be used, but still
        # set it to something reasonable
        if num_lines == 0:
            if prev_last_line is None:
                azimuth_offsets[output_idx] = 0
            else:
                azimuth_offsets[output_idx] = prev_last_line + 1
            continue

        first_pixc_line_idx = np.where(
            tile_pixc_lines_in_tile[tile_pixc_line_mask])[0][0]
        first_tvp_idx = scene_pixc['pixel_cloud']['pixc_line_to_tvp'][
            tile_pixc_lines_in_tile][0].astype('i4')
        last_tvp_idx = scene_pixc['pixel_cloud']['pixc_line_to_tvp'][
            tile_pixc_lines_in_tile][-1].astype('i4')
        first_record_counter = \
            scene_pixc['tvp']['record_counter'][first_tvp_idx]
        last_record_counter = \
            scene_pixc['tvp']['record_counter'][last_tvp_idx]
        num_azimuth_looks = \
            scene_pixc['pixel_cloud']['tile_num_azimuth_looks'][tile_idx]
        nominal_slant_range_spacing = \
            scene_pixc['pixel_cloud']['tile_nominal_slant_range_spacing'][
                tile_idx]

        # This is the first non-empty tile
        if prev_last_line is None:
            azimuth_offsets[output_idx] = 0
            prev_last_line = first_pixc_line_idx + num_lines - 1
            prev_last_record_counter = last_record_counter
            prev_num_azimuth_looks = num_azimuth_looks
            prev_nominal_slant_range_spacing = nominal_slant_range_spacing
            continue

        # Get the index shift between consecutive tiles
        # Set to max_offset if the number of azimuth looks is less than 0,
        # the record counter was reset, or the number of azimuth looks or
        # slant range spacing differ enough from the previous tile
        record_counter_shift = first_record_counter - prev_last_record_counter
        if (num_azimuth_looks <= 0 or record_counter_shift < 0
           or abs(num_azimuth_looks - prev_num_azimuth_looks) > AZ_LOOKS_TOL
           or abs(nominal_slant_range_spacing
                     - prev_nominal_slant_range_spacing) > RNG_SPACING_TOL)
            idx_shift = max_offset
        else:
            idx_shift = np.round(
                record_counter_shift / num_azimuth_looks).astype(int)

        # Clamp between 1 and max_offset
        idx_shift = max(1, min(idx_shift, max_offset))

        azimuth_offsets[output_idx] = \
            prev_last_line - first_pixc_line_idx + idx_shift
        prev_last_line += idx_shift + num_lines - 1
        prev_last_record_counter = last_record_counter
        prev_num_azimuth_looks = num_azimuth_looks
        prev_nominal_slant_range_spacing = nominal_slant_range_spacing

    return azimuth_offsets


def chunk_slant_map(var, az_idx, rng_idx, classif, classif_qual,
                    geolocation_qual, bright_land_flag,
                    specular_ringing_qual, swath_side,
                    chunk_shape, chunk_buffer=(0, 0)):
    """ Takes a variable in the slant plane (both sides) and splits it into
        chunks, with a buffer """
    LOGGER.info('Splitting slant plane into chunks')
    sort_idx = np.arange(len(var))
    for this_side in ['L', 'R']:
        side_mask = np.char.lower(swath_side) == this_side.lower()

        # Skip if side_mask has no valid points
        if not np.any(side_mask):
            continue

        side_var = var[side_mask]
        side_az_idx = az_idx[side_mask]
        side_rng_idx = rng_idx[side_mask]
        side_classif = classif[side_mask]
        side_classif_qual = classif_qual[side_mask]
        side_geolocation_qual = geolocation_qual[side_mask]
        side_bright_land_flag = bright_land_flag[side_mask]
        side_specular_ringing_qual = specular_ringing_qual[side_mask]
        side_sort_idx = sort_idx[side_mask]

        # Group into az/rng squares of chunk_shape[0]*chunk_shape[1],
        # throw away any chunks without data
        for start_az_idx in range(
                np.min(side_az_idx), np.max(side_az_idx) + 1,
                chunk_shape[0]):
            end_az_idx = start_az_idx + chunk_shape[0]
            for start_rng_idx in range(
                    np.min(side_rng_idx), np.max(side_rng_idx) + 1,
                    chunk_shape[1]):
                end_rng_idx = start_rng_idx + chunk_shape[1]
                # Get the mask of pixels within this processing chunk
                # (no buffer)
                side_use_mask = np.logical_and.reduce((
                    side_az_idx >= start_az_idx,
                    side_az_idx < end_az_idx,
                    side_rng_idx >= start_rng_idx,
                    side_rng_idx < end_rng_idx))

                # Skip if use_mask has no valid points
                if not np.any(side_use_mask):
                    continue

                # Add buffer to chunk if commanded
                buff_start_az_idx = start_az_idx
                buff_end_az_idx = end_az_idx
                buff_start_rng_idx = start_rng_idx
                buff_end_rng_idx = end_rng_idx
                if chunk_buffer[0] != 0:
                    masked_az_idx = side_az_idx[side_use_mask]
                    buff_start_az_idx = \
                        np.min(masked_az_idx) - chunk_buffer[0]
                    buff_end_az_idx = \
                        np.max(masked_az_idx) + chunk_buffer[0] + 1
                if chunk_buffer[1] != 0:
                    masked_rng_idx = side_rng_idx[side_use_mask]
                    buff_start_rng_idx = \
                        np.min(masked_rng_idx) - chunk_buffer[1]
                    buff_end_rng_idx = \
                        np.max(masked_rng_idx) + chunk_buffer[1] + 1
                mask = np.logical_and.reduce((
                    side_az_idx >= buff_start_az_idx,
                    side_az_idx < buff_end_az_idx,
                    side_rng_idx >= buff_start_rng_idx,
                    side_rng_idx < buff_end_rng_idx))

                yield (side_var[mask],
                       side_az_idx[mask],
                       side_rng_idx[mask],
                       side_classif[mask],
                       side_classif_qual[mask],
                       side_geolocation_qual[mask],
                       side_bright_land_flag[mask],
                       side_specular_ringing_qual[mask],
                       side_sort_idx[mask],
                       side_use_mask[mask])


def smooth_chunk_and_mask(var, az_idx, rng_idx, classif, classif_qual,
                          geolocation_qual, bright_land_flag,
                          specular_ringing_qual,
                          sort_idx, use_mask,
                          smoothing_footprint,
                          good_klasses=DEFAULT_GOOD_CLASSES,
                          sus_klasses=DEFAULT_SUS_CLASSES,
                          use_bright_land=True,
                          method='composite_with_sus_classes'):
    """ Smoothes a chunk and returns only pixels in use_mask """
    LOGGER.debug('Smoothing az: %s to %s, rng: %s to %s',
                 np.min(az_idx), np.max(az_idx),
                 np.min(rng_idx), np.max(rng_idx))
    smoothed_chunk = smooth_chunk(
        var, az_idx, rng_idx, classif, classif_qual, geolocation_qual,
        bright_land_flag, specular_ringing_qual, smoothing_footprint,
        good_klasses, sus_klasses,
        use_bright_land, method)
    return smoothed_chunk[use_mask], sort_idx[use_mask]


def smooth_chunk(var, az_idx, rng_idx, classif, classif_qual, geolocation_qual,
                 bright_land_flag, specular_ringing_qual, smoothing_footprint,
                 good_klasses=DEFAULT_GOOD_CLASSES,
                 sus_klasses=DEFAULT_SUS_CLASSES,
                 use_bright_land=True,
                 method='composite_with_sus_classes'):
    """ Smoothes a chunk """
    # Median filter with nans doesn't behave the way we want,
    # so we have to use generic_filter and bottleneck.nanmedian
    def _smooth_stage(var, rel_az_idx, rel_rng_idx,
                      slant_plane_var, mask, smoothing_footprint):
        _slant_plane_var = slant_plane_var.copy()
        smoothed_mask = np.isfinite(slant_plane_var)
        _mask = np.logical_and.reduce((
            mask, np.logical_not(smoothed_mask[rel_az_idx, rel_rng_idx]),
            np.logical_not(np.ma.getmaskarray(var))))
        _slant_plane_var[rel_az_idx[_mask], rel_rng_idx[_mask]] = var[_mask]
        slant_plane_var_sm = generic_filter(
            _slant_plane_var, function=bottleneck.nanmedian,
            footprint=smoothing_footprint, mode='constant', cval=np.nan)
        slant_plane_var[rel_az_idx[_mask], rel_rng_idx[_mask]] = \
            slant_plane_var_sm[rel_az_idx[_mask], rel_rng_idx[_mask]]
        slant_plane_var[smoothed_mask] = _slant_plane_var[smoothed_mask]
        return slant_plane_var

    # Get mask of good/sus quality pixels
    good_sus_qual_mask = np.logical_and(
        classif_qual < products.QUAL_IND_DEGRADED,
        geolocation_qual < products.QUAL_IND_DEGRADED,
        specular_ringing_qual < products.QUAL_IND_DEGRADED)

    # Treat bright land as degraded/bad if use_bright_land is false
    if not use_bright_land:
        good_sus_qual_mask = np.logical_and(
            good_sus_qual_mask, np.logical_not(bright_land_flag))

    # Get the start/end and relative az/rng indices
    start_az_idx = np.min(az_idx)
    end_az_idx = np.max(az_idx)
    start_rng_idx = np.min(rng_idx)
    end_rng_idx = np.max(rng_idx)
    rel_az_idx = az_idx-start_az_idx
    rel_rng_idx = rng_idx-start_rng_idx

    slant_plane_var_sm = np.full(
        (end_az_idx - start_az_idx + 1,
         end_rng_idx - start_rng_idx + 1), np.nan)

    if method in ['composite', 'composite_with_sus_classes']:
        # Smooth good/sus quality and good klasses only
        good_classes_mask = np.logical_and(
            np.isin(classif, good_klasses),
            np.logical_not(np.ma.getmaskarray(classif)))
        mask = np.logical_and(good_sus_qual_mask, good_classes_mask)
        slant_plane_var_sm = _smooth_stage(
            var, rel_az_idx, rel_rng_idx, slant_plane_var_sm, mask,
            smoothing_footprint)
        if method == 'composite_with_sus_classes':
            # Smooth good/sus quality and sus klasses
            # along with smoothed good/sus quality and good klasses
            sus_classes_mask = np.logical_and(
                np.isin(classif, sus_klasses),
                np.logical_not(np.ma.getmaskarray(classif)))
            mask = np.logical_and(good_sus_qual_mask, sus_classes_mask)
            slant_plane_var_sm = _smooth_stage(
                var, rel_az_idx, rel_rng_idx, slant_plane_var_sm, mask,
                smoothing_footprint)

    # Smooth all pixels that are left
    mask = np.ones(classif.shape, dtype=bool)
    slant_plane_var_sm = _smooth_stage(
        var, rel_az_idx, rel_rng_idx, slant_plane_var_sm, mask,
        smoothing_footprint)
    return np.ma.fix_invalid(slant_plane_var_sm[rel_az_idx, rel_rng_idx])
