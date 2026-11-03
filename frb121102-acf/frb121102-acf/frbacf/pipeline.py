"""Full analysis of one burst window: raw fit, empty windows, injections, calibration."""
from dataclasses import dataclass

import numpy as np

from .acf import envelope, fit_acf, passes_rule, window_spectrum
from .clean import normalise, normalise_smoothed
from .inject import (empty_window_starts, empty_window_test, invert_width,
                     recover_widths, white_noise_test)
from .params import DEFAULT_WIDTHS, Params


@dataclass
class WindowResult:
    label: str
    window: tuple
    off_window: tuple
    n_masked: int
    centre_freq: float        # MHz, weighted by the smoothed spectrum
    snr: float
    fits: dict                # env scale -> Fit (raw, on the real burst)
    passes: bool              # rule satisfied at every scale
    alt_fits: dict            # same, with the smoothed-noise normalisation
    empty: list               # output of empty_window_test
    white_noise_fraction: float
    recovery: object          # Recovery
    calibrated: dict          # env scale -> {"best", "lo", "hi", "flag"}


def burst_snr(z, time_ms, window, off_win):
    """Sum of the frequency-averaged profile over the window, divided by the
    off-pulse scatter times sqrt(number of bins)."""
    prof = np.nanmean(z, axis=0)
    on = (time_ms >= window[0]) & (time_ms <= window[1])
    off = (time_ms >= off_win[0]) & (time_ms <= off_win[1])
    return float(np.nansum(prof[on]) / (np.nanstd(prof[off]) * np.sqrt(on.sum())))


def raw_fits(z, freq, time_ms, window, params):
    """Fits on the real burst spectrum at each envelope scale (no calibration)."""
    f_avg, spec, good = window_spectrum(z, freq, time_ms, window, params.nav)
    cw = float(np.median(np.diff(f_avg)))
    return {e: fit_acf(spec, good, cw, e, params) for e in params.env_scales}


def analyse_window(burst, window, off_win, params=Params(), widths=DEFAULT_WIDTHS,
                   n_trials=8, n_white=200, seed=1, label=""):
    rng = np.random.default_rng(seed)
    t = burst.time_ms
    z, bad = normalise(burst.intensity, burst.freq, t, off_win, params)

    f_avg, spec, good = window_spectrum(z, burst.freq, t, window, params.nav)
    cw = float(np.median(np.diff(f_avg)))

    island = np.clip(envelope(spec, good, cw, params.island_scale), 0, None)
    centre = float(np.nansum(f_avg * island) / np.nansum(island))

    fits = {e: fit_acf(spec, good, cw, e, params) for e in params.env_scales}
    detected = all(passes_rule(f, params) for f in fits.values())

    z_alt, _ = normalise_smoothed(burst.intensity, burst.freq, t, off_win, params)
    _, spec_alt, good_alt = window_spectrum(z_alt, burst.freq, t, window, params.nav)
    alt_fits = {e: fit_acf(spec_alt, good_alt, cw, e, params) for e in params.env_scales}

    length = window[1] - window[0]
    starts = empty_window_starts(off_win, length)
    if len(starts) == 0:
        raise ValueError("off window is shorter than the burst window; widen it or shorten the burst window")
    empty = empty_window_test(z, burst.freq, t, off_win, length, params)
    backgrounds = [window_spectrum(z, burst.freq, t, (s, s + length), params.nav)[1:] for s in starts]

    step = np.diff(spec)
    noise_sd = float(np.std(step[np.isfinite(step)]) / np.sqrt(2))
    white = white_noise_test(good, cw, noise_sd, params, n_white, rng)

    recovery = recover_widths(island, good, cw, backgrounds, widths, params, n_trials, rng)

    calibrated = {}
    for e in params.env_scales:
        q = recovery.percentiles[e]
        best, flag = invert_width(fits[e].width, q[:, 1], recovery.widths)
        lo, _ = invert_width(fits[e].width, q[:, 2], recovery.widths)   # higher curve -> smaller true width
        hi, _ = invert_width(fits[e].width, q[:, 0], recovery.widths)
        calibrated[e] = {"best": best, "lo": lo, "hi": hi, "flag": flag}

    return WindowResult(label=label, window=tuple(window), off_window=tuple(off_win),
                        n_masked=int(bad.sum()), centre_freq=centre,
                        snr=burst_snr(z, t, window, off_win), fits=fits, passes=detected,
                        alt_fits=alt_fits, empty=empty, white_noise_fraction=white,
                        recovery=recovery, calibrated=calibrated)


def summarise(res):
    """Plain-dict version of a WindowResult, safe to dump as JSON."""
    def num(x):
        return None if x is None or not np.isfinite(x) else round(float(x), 3)

    best = np.array([c["best"] for c in res.calibrated.values()], float)
    return {
        "label": res.label,
        "window_ms": list(res.window),
        "off_window_ms": list(res.off_window),
        "masked_channels": res.n_masked,
        "centre_freq_mhz": num(res.centre_freq),
        "snr": num(res.snr),
        "raw": {str(e): {"width": num(f.width), "amplitude": num(f.amplitude)} for e, f in res.fits.items()},
        "alt_normalisation": {str(e): {"width": num(f.width), "amplitude": num(f.amplitude)}
                              for e, f in res.alt_fits.items()},
        "passes_rule": res.passes,
        "empty_windows_passing": f"{sum(w['passes'] for w in res.empty)}/{len(res.empty)}",
        "white_noise_pass_fraction": num(res.white_noise_fraction),
        "injected_widths": [num(w) for w in res.recovery.widths],
        "injection_detected_fraction": [num(d) for d in res.recovery.detected],
        "calibrated": {str(e): {k: (v if k == "flag" else num(v)) for k, v in c.items()}
                       for e, c in res.calibrated.items()},
        "calibrated_mean": num(np.nanmean(best)),
        "calibrated_range": [num(np.nanmin(best)), num(np.nanmax(best))],
    }
