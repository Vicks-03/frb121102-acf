"""Injection-recovery, empty-window and white-noise tests."""
from dataclasses import dataclass

import numpy as np

from .acf import fit_acf, passes_rule, window_spectrum


def scintle_pattern(n, chan_bw, width, rng):
    """Intensity pattern with unit mean and a Gaussian correlation of HWHM `width` (MHz).

    A complex Gaussian field is smoothed and squared, which gives the
    exponential intensity statistics of fully modulated scintillation.
    """
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    sigma = width / np.sqrt(2 * np.log(2)) / chan_bw
    k = np.arange(-int(5 * sigma) - 1, int(5 * sigma) + 2)
    field = np.convolve(z, np.exp(-0.5 * (k / sigma) ** 2), mode="same")
    power = np.abs(field) ** 2
    return power / power.mean()


@dataclass
class Recovery:
    widths: np.ndarray        # injected widths (MHz)
    percentiles: dict         # env scale -> (len(widths), 3) array of 16/50/84th percentiles
    detected: np.ndarray      # fraction of injections passing the rule at all scales
    n_trials: int


def recover_widths(island, good, chan_bw, backgrounds, widths, params, n_trials, rng):
    """Inject scintle patterns of known width on top of real off-pulse spectra and
    run the same fitting as on the burst.

    island       broad spectral shape of the burst (nonnegative)
    backgrounds  list of (spectrum, good) taken from burst-free windows
    """
    envs = params.env_scales
    pct = {e: np.full((len(widths), 3), np.nan) for e in envs}
    detected = np.zeros(len(widths))
    n_total = n_trials * len(backgrounds)

    for i, w in enumerate(widths):
        found = {e: [] for e in envs}
        n_pass_all = 0
        for bg, bg_good in backgrounds:
            g = good & bg_good
            for _ in range(n_trials):
                spec = island * scintle_pattern(len(island), chan_bw, w, rng) + np.nan_to_num(bg)
                spec = np.where(g, spec, np.nan)
                fits = {e: fit_acf(spec, g, chan_bw, e, params) for e in envs}
                ok = {e: passes_rule(f, params) for e, f in fits.items()}
                n_pass_all += all(ok.values())
                for e in envs:
                    found[e].append(fits[e].width if ok[e] else np.nan)
        detected[i] = n_pass_all / n_total
        for e in envs:
            v = np.array(found[e])
            if np.isfinite(v).sum() >= 5:
                pct[e][i] = np.nanpercentile(v, [16, 50, 84])
    return Recovery(np.asarray(widths, float), pct, detected, n_total)


def invert_width(measured, curve, widths):
    """Map a measured width to the injected width that would produce it.

    curve is the recovered width (median, or a percentile) at each injected width.
    Returns (value, flag). flag is empty when the measurement lies inside the
    calibrated range, otherwise it says which side it fell off.
    """
    ok = np.isfinite(curve)
    if ok.sum() < 2 or not np.isfinite(measured):
        return np.nan, "no calibration"
    c = np.maximum.accumulate(curve[ok])      # force monotonic so interpolation is defined
    w = np.asarray(widths)[ok]
    if measured > c[-1]:
        return float(w[-1]), "above calibrated range (lower limit)"
    if measured < c[0]:
        return float(w[0]), "below calibrated range (upper limit)"
    return float(np.interp(measured, c, w)), ""


def empty_window_starts(off_win, length):
    """Start times of non-overlapping windows of `length` ms inside off_win."""
    return np.arange(off_win[0], off_win[1] - length + 1e-9, length)


def empty_window_test(z, freq, time_ms, off_win, length, params):
    """Run the fit on burst-free windows. Returns a list of dicts, one per window."""
    out = []
    for s in empty_window_starts(off_win, length):
        f_avg, spec, good = window_spectrum(z, freq, time_ms, (s, s + length), params.nav)
        cw = float(np.median(np.diff(f_avg)))
        fits = {e: fit_acf(spec, good, cw, e, params) for e in params.env_scales}
        out.append({"start": float(s), "fits": fits,
                    "passes": all(passes_rule(f, params) for f in fits.values())})
    return out


def white_noise_test(good, chan_bw, noise_sd, params, n, rng):
    """Fraction of pure-noise spectra that pass the rule at every scale."""
    n_pass = 0
    for _ in range(n):
        spec = np.where(good, rng.normal(scale=noise_sd, size=len(good)), np.nan)
        n_pass += all(passes_rule(fit_acf(spec, good, chan_bw, e, params), params)
                      for e in params.env_scales)
    return n_pass / n
