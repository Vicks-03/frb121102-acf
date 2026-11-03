"""Quick look at a burst file: where is it, and how strong?

Uses a whole-file baseline, so the numbers are only for choosing windows.
The measurement itself normalises with the off-pulse window instead.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

from .clean import normalise_whole_file


def suggest_windows(burst, params, pad_ms=0.3, gap_ms=2.0, smooth_ms=0.5):
    z, _ = normalise_whole_file(burst.intensity, burst.freq, params)
    t = burst.time_ms
    prof = np.nanmean(z, axis=0)

    k = max(int(smooth_ms * 1e-3 / burst.dt), 1)
    sm = uniform_filter1d(np.nan_to_num(prof), k)
    base = np.median(sm)
    noise = 1.4826 * np.median(np.abs(sm - base))
    prof, sm = prof - base, sm - base

    peak = int(np.argmax(sm))
    thresh = max(0.25 * sm[peak], 3 * noise)
    i0 = i1 = peak
    while i0 > 0 and sm[i0 - 1] > thresh:
        i0 -= 1
    while i1 < len(sm) - 1 and sm[i1 + 1] > thresh:
        i1 += 1
    start, end = max(t[i0] - pad_ms, 0.0), min(t[i1] + pad_ms, t[-1])

    after = t[-1] - (end + gap_ms) - 0.1
    before = (start - gap_ms) - 0.1
    off = (end + gap_ms, t[-1] - 0.1) if after >= before else (0.1, start - gap_ms)

    on = (t >= start) & (t <= end)
    outside = (t < start - 2) | (t > end + 2)
    snr = float(np.nansum(prof[on]) / (np.nanstd(prof[outside]) * np.sqrt(on.sum())))
    return {"peak_ms": float(t[peak]), "peak_sigma": float(sm[peak] / noise),
            "burst_window": (float(start), float(end)), "off_window": (float(off[0]), float(off[1])),
            "snr": snr, "z": z, "profile": prof, "smoothed": sm}
