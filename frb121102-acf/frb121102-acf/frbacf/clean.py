"""Channel normalisation, masking and rebinning."""
import warnings

import numpy as np
from scipy.ndimage import median_filter


def _bad_channels(std, freq, params):
    """Flag channels with unusual noise, widen the mask by +-10 channels, drop band edges."""
    med = np.nanmedian(std)
    mad = np.nanmedian(np.abs(std - med))
    bad = ~np.isfinite(std) | (std == 0) | (np.abs(std - med) > 6 * 1.4826 * mad)
    bad = np.convolve(bad, np.ones(21), mode="same") > 0
    bad |= (freq < params.f_lo) | (freq > params.f_hi)
    return bad


def normalise(intensity, freq, time_ms, off_win, params):
    """Subtract each channel's off-pulse mean and divide by its off-pulse std.

    Returns (z, bad). z is float32 with NaN in masked channels.
    """
    off = (time_ms >= off_win[0]) & (time_ms <= off_win[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(intensity[:, off], axis=1)
        std = np.nanstd(intensity[:, off], axis=1)
    bad = _bad_channels(std, freq, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ((intensity - mean[:, None]) / std[:, None]).astype(np.float32)
    z[bad, :] = np.nan
    return z, bad


def normalise_smoothed(intensity, freq, time_ms, off_win, params):
    """Alternative normalisation: divide by a smoothed noise level instead of
    each channel's own std. Used as a check that the result does not depend on
    the per-channel normalisation.
    """
    off = (time_ms >= off_win[0]) & (time_ms <= off_win[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(intensity[:, off], axis=1)
        std = np.nanstd(intensity[:, off], axis=1)
    bad = _bad_channels(std, freq, params)

    chan_bw = float(np.median(np.diff(freq)))
    size = int(round(params.smooth_noise_mhz / chan_bw)) | 1      # odd window
    filled = np.where(np.isfinite(std) & ~bad, std, np.nanmedian(std[~bad]))
    smooth = median_filter(filled, size=size, mode="nearest")
    z = ((intensity - mean[:, None]) / smooth[:, None]).astype(np.float32)
    z[bad, :] = np.nan
    return z, bad


def normalise_whole_file(intensity, freq, params):
    """Quick normalisation using the whole file (median and MAD per channel).
    Only meant for screening: the burst itself leaks into the baseline.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmedian(intensity, axis=1)
        std = 1.4826 * np.nanmedian(np.abs(intensity - mean[:, None]), axis=1)
    bad = _bad_channels(std, freq, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ((intensity - mean[:, None]) / std[:, None]).astype(np.float32)
    z[bad, :] = np.nan
    return z, bad


def rebin(a, ft, tt):
    """Average a[freq, time] in blocks of ft channels by tt time bins (NaN-aware)."""
    nf, nt = a.shape
    a = a[: nf // ft * ft, : nt // tt * tt]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a.reshape(nf // ft, ft, nt // tt, tt), axis=(1, 3))
