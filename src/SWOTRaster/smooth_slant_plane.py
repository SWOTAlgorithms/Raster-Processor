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
import SWOTRaster.products as products

from functools import partial
from scipy.ndimage import generic_filter
from SWOTRaster.raster_agg import fn_star
from SWOTWater.constants import PIXC_CLASSES
from SWOTRaster.errors import RasterUsageException

DEFAULT_MAX_CHUNK_SHAPE=(2000, 2000)
DEFAULT_SMOOTHING_FILTER_SHAPE=(5, 10)
DEFAULT_GOOD_CLASSES=[PIXC_CLASSES['water_near_land'],
                      PIXC_CLASSES['open_water']]
DEFAULT_SUS_CLASSES=[PIXC_CLASSES['land_near_water'],
                     PIXC_CLASSES['dark_water'],
                     PIXC_CLASSES['low_coh_water_near_land'],
                     PIXC_CLASSES['low_coh_water'],
                     PIXC_CLASSES['dark_water_near_land'],
                     PIXC_CLASSES['dark_water_legacy']]

LOGGER = logging.getLogger(__name__)

def unwrap_idx(idx, unwrap_vec, ref_to_zero=False,
               max_idx_val=None, wrap_buffer=0):
    """ Unwraps an integer index based on a vector
        ref_to_zero makes the resulting index start at 0
        max_idx_val controls the wrap point of the index, default is dtype max
        wrap_buffer is added if there is a gap at the wrap point greater than
        wrap_buffer """
    if max_idx_val is None:
        max_idx_val = np.iinfo(idx.dtype).max

    sort_idx = np.argsort(unwrap_vec)
    sorted_idx = idx[sort_idx]

    # Get the shift needed to reference output to 0
    ref_shift = 0
    if ref_to_zero:
        ref_shift = sorted_idx[0]

    # Get the indices where the index wraps, return with ref shift if no wraps
    wrap_indices = np.where(sorted_idx[:-1] > sorted_idx[1:])[0]
    if wrap_indices.size==0:
        return idx-ref_shift

    # Use sorted_idx before ref shift to figure out if we need to add wrap_buffer
    # But use sorted_idx with ref shift to figure out what the offset should be
    gap_size = max_idx_val - sorted_idx[wrap_indices] + sorted_idx[wrap_indices+1]
    buffs = np.minimum(gap_size, wrap_buffer)
    sorted_idx[:wrap_indices[0]+1] -= ref_shift
    offsets = sorted_idx[wrap_indices] - sorted_idx[wrap_indices+1] + 1

    # Unwrap
    for idx, offset, buff in zip(wrap_indices, offsets, buffs):
        sorted_idx[idx+1:] += offset + buff

    return sorted_idx[np.argsort(sort_idx)]

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
    LOGGER.info('Smoothing {} in slant plane'.format(var_name))
    if method not in ['simple', 'composite', 'composite_with_sus_classes']:
        raise RasterUsageException(
            'Unknown slant plane smoothing method: {}'.format(method))

    # If all input pixels are masked, return fully masked array
    var = scene_pixc['pixel_cloud'][var_name]
    var_out = np.ma.masked_all_like(var)
    if len(var) == 0 or np.all(var.mask):
        return var_out

    smoothing_footprint = np.ones((smoothing_filter_shape[0],
                                   smoothing_filter_shape[1]))

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
    specular_ringing_qual = products.QUAL_IND_GOOD*np.ones(
        np.shape(specular_ringing_flag))
    specular_ringing_qual[specular_intersecting_prior] = \
        products.QUAL_IND_SUSPECT
    specular_ringing_qual[specular_not_intersecting_prior] = \
        products.QUAL_IND_DEGRADED

    line_idx = scene_pixc['pixel_cloud']['line_index']
    pixc_line_to_tvp = scene_pixc['pixel_cloud']['pixc_line_to_tvp'][line_idx].astype('i4')
    record_counter = scene_pixc['tvp']['record_counter'][pixc_line_to_tvp]
    tvp_time = scene_pixc['tvp']['time'][pixc_line_to_tvp]
    swath_side = scene_pixc['tvp']['swath_side'][pixc_line_to_tvp]

    # Recompute an azimuth index using the record counter
    recomputed_az_idx = ((record_counter-scene_pixc['pixel_cloud'].azimuth_offset) \
                         / scene_pixc['pixel_cloud'].num_azimuth_looks).astype('i4')
    recomputed_az_idx = unwrap_idx(
        recomputed_az_idx, tvp_time, ref_to_zero=True,
        wrap_buffer=smoothing_filter_shape[0])

    # Recompute a range index using the pixelwise range
    rng = scene_pixc['pixel_cloud']['range']
    recomputed_rng_idx = (rng / scene_pixc.nominal_slant_range_spacing).astype('i4')

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
                (2*smoothing_footprint.shape[0], 2*smoothing_footprint.shape[1]))))
    else:
        chunk_shape = max_chunk_shape
        results = [_smooth_fn(arglist) for arglist in chunk_slant_map(
            var, recomputed_az_idx, recomputed_rng_idx, classif,
            classif_qual, geolocation_qual, bright_land_flag,
            specular_ringing_qual, swath_side, chunk_shape,
            (2*smoothing_footprint.shape[0], 2*smoothing_footprint.shape[1]))]

    for data, indices in results:
        var_out[indices] = data
    return var_out

def chunk_slant_map(var, az_idx, rng_idx, classif, classif_qual,
                    geolocation_qual, bright_land_flag,
                    specular_ringing_qual, swath_side,
                    chunk_shape, chunk_buffer=(0,0)):
    """ Takes a variable in the slant plane (both sides) and splits it into
        chunks, with a buffer """
    indices = np.arange(len(var))
    for this_side in ['L', 'R']:
        side_mask = swath_side==this_side

        # Skip if side_mask has no valid points
        if not np.any(side_mask):
            continue

        side_az_idx = az_idx[side_mask]
        side_rng_idx = rng_idx[side_mask]
        side_indices = indices[side_mask]
        side_var = var[side_mask]
        side_classif = classif[side_mask]
        side_classif_qual = classif_qual[side_mask]
        side_geolocation_qual = geolocation_qual[side_mask]
        side_bright_land_flag = bright_land_flag[side_mask]
        side_specular_ringing_qual = specular_ringing_qual[side_mask]

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
                       side_bright_land_flag[mask],
                       side_specular_ringing_qual[mask],
                       side_indices[mask],
                       use_mask[mask])

def smooth_chunk_and_mask(var, az_idx, rng_idx, classif, classif_qual,
                          geolocation_qual, bright_land_flag,
                          specular_ringing_qual,
                          side_sort_indices, use_mask,
                          smoothing_footprint,
                          good_klasses=DEFAULT_GOOD_CLASSES,
                          sus_klasses=DEFAULT_SUS_CLASSES,
                          use_bright_land=True,
                          method='composite_with_sus_classes'):
    """ Smoothes a chunk and returns only pixels in use_mask """
    LOGGER.debug('Smoothing az: {} to {}, rng: {} to {}'.format(
        np.min(az_idx), np.max(az_idx), np.min(rng_idx), np.max(rng_idx)))
    smoothed_chunk = smooth_chunk(
        var, az_idx, rng_idx, classif, classif_qual, geolocation_qual,
        bright_land_flag, specular_ringing_qual, smoothing_footprint,
        good_klasses, sus_klasses,
        use_bright_land, method)
    return smoothed_chunk[use_mask], side_sort_indices[use_mask]

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
            footprint=smoothing_footprint)
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
        mask = np.logical_and(good_sus_qual_mask, np.isin(classif, good_klasses))
        slant_plane_var_sm = _smooth_stage(
            var, rel_az_idx, rel_rng_idx, slant_plane_var_sm, mask,
            smoothing_footprint)
        if method == 'composite_with_sus_classes':
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
    return np.ma.fix_invalid(slant_plane_var_sm[rel_az_idx, rel_rng_idx])
