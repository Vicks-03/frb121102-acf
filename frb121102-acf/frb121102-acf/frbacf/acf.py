"""Burst spectrum, envelope subtraction, frequency ACF and Gaussian fit."""
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit


@dataclass
class Fit:
    width: float          # fitted HWHM in MHz (NaN if the fit failed)
    amplitude: float      # fitted amplitude of the Gaussian at zero lag
    width_err: float      # formal fit error on the width (MHz)
    lags: np.ndarray      # MHz
    acf: np.ndarray       # normalised so that lag 0 = 1


def window_spectrum(z, freq, time_ms, win, nav):
    """Average z over a time window, then average nav neighbouring channels.

    Returns (freq_avg, spectrum, good). Blocks with more than 10% masked
    channels are marked not good and set to NaN.
    """
    on = (time_ms >= win[0]) & (time_ms <= win[1])
    if not on.any():
        raise ValueError(f"no time bins inside window {win}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spec = np.nanmean(z[:, on], axis=1)
        n = len(spec) // nav * nav
        blocks = spec[:n].reshape(-1, nav)
        freq_avg = freq[:n].reshape(-1, nav).mean(axis=1)
        good = np.isfinite(blocks).mean(axis=1) > 0.9
        avg = np.where(good, np.nanmean(blocks, axis=1), np.nan)
    return freq_avg, avg, good


def envelope(spec, good, chan_bw, scale_mhz):
    """Moving average over scale_mhz that ignores masked channels."""
    w = max(int(scale_mhz / chan_bw), 3)
    num = uniform_filter1d(np.nan_to_num(spec), w)
    den = uniform_filter1d(good.astype(float), w)
    return num / np.maximum(den, 1e-6)


def masked_acf(x):
    """Autocorrelation of x ignoring NaNs, normalised so lag 0 = 1."""
    m = np.isfinite(x).astype(float)
    x0 = np.where(m > 0, x - np.nanmean(x), 0.0)
    num = np.correlate(x0, x0, "full")[len(x0) - 1:]
    cnt = np.correlate(m, m, "full")[len(x0) - 1:]
    a = num / np.maximum(cnt, 1)
    return a / a[0]


def gaussian(lag, amp, hwhm):
    return amp * np.exp(-np.log(2) * (lag / hwhm) ** 2)


def fit_acf(spec, good, chan_bw, env_scale, params):
    """Subtract the envelope, take the ACF and fit a Gaussian to lags in (0, fit_max]."""
    x = spec - envelope(spec, good, chan_bw, env_scale)
    a = masked_acf(x)
    lags = np.arange(len(a)) * chan_bw
    sel = (lags > 0) & (lags <= params.fit_max)
    try:
        popt, pcov = curve_fit(gaussian, lags[sel], a[sel], p0=[0.3, 8.0],
                               bounds=([0, 0.3], [2, 200]))
    except Exception:
        return Fit(np.nan, 0.0, np.nan, lags, a)
    return Fit(float(popt[1]), float(popt[0]), float(np.sqrt(pcov[1, 1])), lags, a)


def passes_rule(fit, params):
    """Detection rule: amplitude >= a_min and HWHM >= min_width.
    The caller applies it at every envelope scale.
    """
    return bool(np.isfinite(fit.width)
                and fit.amplitude >= params.a_min
                and fit.width >= params.min_width)
