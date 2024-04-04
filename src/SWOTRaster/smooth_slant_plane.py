'''
Copyright (c) 2024-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben

'''
import logging
import bottleneck
import numpy as np
import multiprocessing

from functools import partial
from scipy.ndimage import generic_filter
from SWOTRaster.raster_agg import fn_star
from SWOTWater.constants import PIXC_CLASSES

DEFAULT_GOOD_CLASSES=[PIXC_CLASSES['water_near_land'],
                      PIXC_CLASSES['open_water']]
DEFAULT_SUS_CLASSES=[PIXC_CLASSES['land_near_water'],
                     PIXC_CLASSES['dark_water'],
                     PIXC_CLASSES['low_coh_water_near_land'],
                     PIXC_CLASSES['low_coh_water'],
                     PIXC_CLASSES['dark_water_near_land'],
                     PIXC_CLASSES['dark_water_legacy']]

LOGGER = logging.getLogger(__name__)

def smooth_slant_plane(
        scene_pixc, var_name='height',
        max_chunk_shape=(2000, 2000),
        smoothing_filter_shape=(5, 10),
        good_klasses=DEFAULT_GOOD_CLASSES,
        sus_klasses=DEFAULT_SUS_CLASSES,
        class_qual_suspect=0,
        class_qual_degraded=0,
        class_qual_bad=0,
        geo_qual_suspect=0,
        geo_qual_degraded=0,
        geo_qual_bad=0,
        method='composite_with_sus_classes',
        max_worker_processes=1):
    """ Smoothes in slant plane """
    LOGGER.info('Smoothing {} in slant plane'.format(var_name))
    # TODO: verify that this doesn't crash if no pixels are in the input

    smoothing_footprint = np.ones((smoothing_filter_shape[0],
                                   smoothing_filter_shape[1]))

    az_idx = scene_pixc['pixel_cloud']['line_index']
    rng = scene_pixc['pixel_cloud']['range']
    classif = scene_pixc['pixel_cloud']['classification']
    classif_qual = scene_pixc.get_summary_qual_flag(
        'classification_qual', class_qual_suspect,
        class_qual_degraded, class_qual_bad)
    geolocation_qual = scene_pixc.get_summary_qual_flag(
        'geolocation_qual', geo_qual_suspect,
        geo_qual_degraded, geo_qual_bad)
    pixc_line_to_tvp = scene_pixc['pixel_cloud']['pixc_line_to_tvp'][az_idx].astype(int)
    record_counter = scene_pixc['tvp']['record_counter'][pixc_line_to_tvp]
    recomputed_az_idx = ((record_counter-scene_pixc['pixel_cloud'].azimuth_offset) \
                         / scene_pixc['pixel_cloud'].num_azimuth_looks).astype(int)
    recomputed_rng_idx = (rng/scene_pixc.nominal_slant_range_spacing).astype(int)
    swath_side = scene_pixc['tvp']['swath_side'][pixc_line_to_tvp]
    var = scene_pixc['pixel_cloud'][var_name]

    smooth_fn = partial(smooth_chunk_and_mask,
                        smoothing_footprint=smoothing_footprint,
                        good_klasses=good_klasses,
                        sus_klasses=sus_klasses,
                        method=method)
    _smooth_fn = partial(fn_star, smooth_fn)
    if max_worker_processes > 1:
        sz_az = np.max(recomputed_az_idx)-np.min(recomputed_az_idx)+1
        sz_rng = np.max(recomputed_rng_idx)-np.min(recomputed_rng_idx)+1
        chunk_shape = (
            int(min(np.sqrt((sz_az*sz_rng)/(max_worker_processes*2)), max_chunk_shape[0])),
            int(min(np.sqrt((sz_az*sz_rng)/(max_worker_processes*2)), max_chunk_shape[1])))
        with multiprocessing.get_context('spawn').Pool(
                processes=max_worker_processes) as pool:
            results = list(pool.imap(_smooth_fn, chunk_slant_map(
                var, recomputed_az_idx, recomputed_rng_idx, classif,
                classif_qual, geolocation_qual, swath_side, chunk_shape,
                (2*smoothing_footprint.shape[0], 2*smoothing_footprint.shape[1]))))
    else:
        chunk_shape = max_chunk_shape
        results = [_smooth_fn(arglist) for arglist in chunk_slant_map(
            var, recomputed_az_idx, recomputed_rng_idx, classif,
            classif_qual, geolocation_qual, swath_side, chunk_shape,
            (2*smoothing_footprint.shape[0], 2*smoothing_footprint.shape[1]))]

    var_out = np.ma.masked_all_like(var)
    for data, indices in results:
        var_out[indices] = data
    return var_out

def chunk_slant_map(var, az_idx, rng_idx, classif,
                    classif_qual, geolocation_qual, swath_side,
                    chunk_shape, chunk_buffer=(0,0)):
    """ Takes a variable in the slant plane (both sides) and splits it into
        chunks, with a buffer """
    indices = np.arange(len(var))
    for this_side in ['L', 'R']:
        side_mask = swath_side==this_side
        side_az_idx = az_idx[side_mask]
        side_rng_idx = rng_idx[side_mask]
        side_indices = indices[side_mask]
        side_var = var[side_mask]
        side_classif = classif[side_mask]
        side_classif_qual = classif_qual[side_mask]
        side_geolocation_qual = geolocation_qual[side_mask]

        # Group into az/rng squares of chunk_shape[0]*chunk_shape[1],
        # throw away any chunks without data
        for start_az_idx in range(np.min(side_az_idx), np.max(side_az_idx) + 1,
                                  chunk_shape[1]):
            end_az_idx = start_az_idx + chunk_shape[1]
            for start_rng_idx in range(np.min(side_rng_idx), np.max(side_rng_idx) + 1,
                                       chunk_shape[0]):
                end_rng_idx = start_rng_idx + chunk_shape[0]
                # Get the mask of pixels within this processing chunk (no buffer)
                use_mask = np.logical_and.reduce((
                    side_az_idx >= start_az_idx,
                    side_az_idx < end_az_idx,
                    side_rng_idx >= start_rng_idx,
                    side_rng_idx < end_rng_idx))

                # Skip if use_mask has no valid points
                if not np.any(use_mask):
                    continue

                # Add buffer to chunk if commanded
                buff_start_az_idx = start_az_idx
                buff_end_az_idx = end_az_idx
                buff_start_rng_idx = start_rng_idx
                buff_end_rng_idx = end_rng_idx
                if chunk_buffer[0] != 0:
                    masked_az_idx = side_az_idx[use_mask]
                    buff_start_az_idx = np.min(masked_az_idx) - chunk_buffer[0]
                    buff_end_az_idx = np.max(masked_az_idx) + chunk_buffer[0] + 1
                if chunk_buffer[1] != 0:
                    masked_rng_idx = side_rng_idx[use_mask]
                    buff_start_rng_idx = np.min(masked_rng_idx) - chunk_buffer[1]
                    buff_end_rng_idx = np.max(masked_rng_idx) + chunk_buffer[1] + 1
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
                       side_indices[mask],
                       use_mask[mask])

def smooth_chunk_and_mask(var, az_idx, rng_idx, classif,
                          classif_qual,geolocation_qual,
                          side_sort_indices, use_mask,
                          smoothing_footprint,
                          good_klasses=DEFAULT_GOOD_CLASSES,
                          sus_klasses=DEFAULT_SUS_CLASSES,
                          method='composite_with_sus_classes'):
    """ Smoothes a chunk and returns only pixels in use_mask """
    LOGGER.debug('Smoothing az: {} to {}, rng: {} to {}'.format(
        np.min(az_idx), np.max(az_idx), np.min(rng_idx), np.max(rng_idx)))
    smoothed_chunk = smooth_chunk(
        var, az_idx, rng_idx, classif, classif_qual, geolocation_qual,
        smoothing_footprint, good_klasses, sus_klasses, method)
    return smoothed_chunk[use_mask], side_sort_indices[use_mask]

def smooth_chunk(var, az_idx, rng_idx, classif, classif_qual, geolocation_qual,
                 smoothing_footprint,
                 good_klasses=DEFAULT_GOOD_CLASSES,
                 sus_klasses=DEFAULT_SUS_CLASSES,
                 method='composite_with_sus_classes'):
    """ Smoothes a chunk """
    # Median filter with nans doesn't behave the way we want,
    # so we have to use generic_filter and bottleneck.nanmedian
    def _smooth_stage(var, rel_az_idx, rel_rng_idx,
                      slant_plane_var, mask, smoothing_footprint):
        _slant_plane_var = slant_plane_var.copy()
        smoothed_mask = np.isfinite(slant_plane_var)
        _mask = np.logical_and(
            mask, np.logical_not(smoothed_mask[rel_az_idx, rel_rng_idx]))
        _slant_plane_var[rel_az_idx[_mask], rel_rng_idx[_mask]] = var[_mask]
        slant_plane_var_sm = generic_filter(
            _slant_plane_var, function=bottleneck.nanmedian,
            footprint=smoothing_footprint)
        slant_plane_var[rel_az_idx[_mask], rel_rng_idx[_mask]] = \
            slant_plane_var_sm[rel_az_idx[_mask], rel_rng_idx[_mask]]
        slant_plane_var[smoothed_mask] = _slant_plane_var[smoothed_mask]
        return slant_plane_var

    # Get mask of good/sus quality pixels
    good_sus_qual_mask = np.logical_and(classif_qual < 2, geolocation_qual < 2)

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

    if method in ['composite', 'composite_with_land']:
        # Smooth good/sus quality and good klasses only
        mask = np.logical_and(good_sus_qual_mask, np.isin(classif, good_klasses))
        slant_plane_var_sm = _smooth_stage(
            var, rel_az_idx, rel_rng_idx, slant_plane_var_sm, mask,
            smoothing_footprint)
        if method == 'composite_with_land':
            # Smooth good/sus quality and sus klasses
            # along with smoothed good/sus quality and good klasses
            mask = np.logical_and(good_sus_qual_mask, np.isin(classif, sus_klasses))
            slant_plane_var_sm = _smooth_stage(
                var, rel_az_idx, rel_rng_idx, slant_plane_var_sm, mask,
                smoothing_footprint)

    # Smooth all pixels that are left
    mask = np.ones(classif.shape, dtype=bool)
    slant_plane_var_sm = _smooth_stage(
        var, rel_az_idx, rel_rng_idx, slant_plane_var_sm, mask,
        smoothing_footprint)
    return slant_plane_var_sm[rel_az_idx, rel_rng_idx]
