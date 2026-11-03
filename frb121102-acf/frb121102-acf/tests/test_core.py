"""Tests on synthetic data. Run with:  pytest -q"""
import numpy as np

from frbacf.acf import envelope, fit_acf, gaussian, masked_acf, passes_rule, window_spectrum
from frbacf.clean import normalise, rebin
from frbacf.inject import invert_width, scintle_pattern, white_noise_test
from frbacf.io import Burst
from frbacf.params import Params
from frbacf.pipeline import analyse_window

P = Params()


def make_burst(width=10.0, amp=0.12, seed=0, nchan=4096, nbin=512, chan_bw=0.75, dt=40e-6):
    """Noise plus a burst between 2.7 and 5.0 ms with a broad island and scintles of known width."""
    rng = np.random.default_rng(seed)
    freq = 4700.0 + chan_bw * np.arange(nchan)
    t = np.arange(nbin) * dt * 1e3

    island = np.exp(-0.5 * ((freq - 6000.0) / 400.0) ** 2)
    pattern = scintle_pattern(nchan, chan_bw, width, rng)
    on = ((t >= 2.7) & (t <= 5.0)).astype(float)

    data = 100.0 + rng.normal(size=(nchan, nbin))
    data += amp * (island * pattern)[:, None] * on[None, :]
    data[500:503, :] += 50 * rng.normal(size=(3, nbin))      # noisy channels (interference)
    data[1000, :] = np.nan                                   # flagged channel
    return Burst(path="synthetic", intensity=data.astype(np.float32), freq=freq, dt=dt)


def test_scintle_pattern_has_unit_mean_and_requested_width():
    rng = np.random.default_rng(1)
    chan_bw, width = 0.5, 8.0
    pat = scintle_pattern(20000, chan_bw, width, rng)
    assert abs(pat.mean() - 1.0) < 1e-9
    a = masked_acf(pat)
    lags = np.arange(len(a)) * chan_bw
    half = lags[np.argmax(a < 0.5 * a[1])] if a[1] > 0.5 else None
    assert half is not None and abs(half - width) < 2.0      # ACF drops to ~half near one HWHM


def test_gaussian_peak_and_half_height():
    assert gaussian(0.0, 0.4, 7.0) == 0.4
    assert abs(gaussian(7.0, 0.4, 7.0) - 0.2) < 1e-12


def test_masked_acf_ignores_nan_and_starts_at_one():
    x = np.random.default_rng(2).normal(size=500)
    x[100:110] = np.nan
    a = masked_acf(x)
    assert abs(a[0] - 1.0) < 1e-12 and np.all(np.isfinite(a))


def test_envelope_ignores_masked_channels():
    spec = np.ones(200)
    good = np.ones(200, dtype=bool)
    spec[50:60] = np.nan
    good[50:60] = False
    env = envelope(spec, good, 1.0, 20.0)
    assert np.allclose(env[good], 1.0, atol=1e-6)


def test_normalise_masks_noisy_and_flagged_channels():
    b = make_burst()
    z, bad = normalise(b.intensity, b.freq, b.time_ms, (8.0, 20.0), P)
    assert bad[500:503].all() and bad[1000]
    assert np.isnan(z[500:503]).all()
    assert abs(np.nanmean(z[:, b.time_ms > 8])) < 0.05


def test_rebin_shape_and_value():
    a = np.arange(24, dtype=float).reshape(4, 6)
    out = rebin(a, 2, 3)
    assert out.shape == (2, 2)
    assert out[0, 0] == np.mean([0, 1, 2, 6, 7, 8])


def test_white_noise_rarely_passes_rule():
    rng = np.random.default_rng(3)
    good = np.ones(2000, dtype=bool)
    frac = white_noise_test(good, 1.46, 1.0, P, 60, rng)
    assert frac <= 0.05


def test_fit_recovers_known_width_roughly():
    # No island, no noise: the envelope subtraction bias is the only thing at work
    rng = np.random.default_rng(4)
    widths = []
    for _ in range(30):
        spec = scintle_pattern(2000, 1.46, 10.0, rng)
        f = fit_acf(spec, np.ones(2000, dtype=bool), 1.46, 300, P)
        widths.append(f.width)
    med = np.nanmedian(widths)
    assert 7.0 < med < 12.0


def test_invert_width_and_flags():
    widths = np.array([4.0, 8.0, 12.0, 16.0])
    curve = np.array([3.5, 6.0, 8.0, 9.0])               # recovered widths, biased low
    val, flag = invert_width(7.0, curve, widths)
    assert flag == "" and 8.0 < val < 12.0
    assert "above" in invert_width(20.0, curve, widths)[1]
    assert "below" in invert_width(1.0, curve, widths)[1]
    assert np.isnan(invert_width(np.nan, curve, widths)[0])


def test_window_spectrum_marks_masked_blocks():
    z = np.random.default_rng(5).normal(size=(64, 20)).astype(np.float32)
    z[0:8, :] = np.nan                                    # one whole averaging block masked
    freq = np.arange(64, dtype=float)
    time_ms = np.arange(20) * 0.1
    f_avg, spec, good = window_spectrum(z, freq, time_ms, (0.5, 1.5), 8)
    assert len(spec) == 8 and not good[0] and np.isnan(spec[0]) and good[1:].all()


def test_pipeline_on_synthetic_burst():
    burst = make_burst(width=10.0, amp=0.12)
    res = analyse_window(burst, (2.7, 5.0), (8.0, 20.0), P, widths=(6, 10, 16, 24),
                         n_trials=2, n_white=20, seed=1, label="synthetic")
    assert res.passes, {e: (f.width, f.amplitude) for e, f in res.fits.items()}
    assert not any(w["passes"] for w in res.empty)
    best = np.array([c["best"] for c in res.calibrated.values()], float)
    assert 6.0 < np.nanmean(best) < 15.0, best
    assert 5800 < res.centre_freq < 6200


def test_pipeline_on_noise_only_does_not_pass():
    burst = make_burst(amp=0.0, seed=7)
    res = analyse_window(burst, (2.7, 5.0), (8.0, 20.0), P, widths=(6, 10, 16),
                         n_trials=1, n_white=10, seed=1, label="noise")
    assert not res.passes
