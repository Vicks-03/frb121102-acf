"""Plain-text report for a WindowResult."""
import numpy as np


def format_report(res, params):
    envs = params.env_scales
    lines = []
    add = lines.append

    add("=" * 70)
    add(f"{res.label}  window {res.window[0]}-{res.window[1]} ms   off-pulse {res.off_window[0]}-{res.off_window[1]} ms")
    add("=" * 70)
    add(f"masked channels        : {res.n_masked}")
    add(f"S/N over the window    : {res.snr:.1f}")
    add(f"centre frequency       : {res.centre_freq:.0f} MHz")
    add("")
    add(f"detection rule: amplitude >= {params.a_min} and HWHM >= {params.min_width} MHz at every scale")
    add(f"{'env (MHz)':>10} {'raw HWHM':>9} {'amplitude':>10} {'alt. norm HWHM':>15}")
    for e in envs:
        f, g = res.fits[e], res.alt_fits[e]
        add(f"{e:>10} {f.width:9.1f} {f.amplitude:10.2f} {g.width:15.1f}")
    add(f"passes the rule at all scales: {res.passes}")
    widths = [res.fits[e].width for e in envs]
    add(f"raw width changes by a factor of {max(widths) / max(min(widths), 0.1):.1f} across the scales")
    add("")
    add(f"empty windows passing the rule: {sum(w['passes'] for w in res.empty)} of {len(res.empty)}")
    for w in res.empty:
        cells = "  ".join(f"{e}: w={w['fits'][e].width:5.1f} A={w['fits'][e].amplitude:4.2f}" for e in envs)
        add(f"  {w['start']:5.1f} ms | {cells}")
    add(f"white-noise spectra passing the rule: {res.white_noise_fraction:.3f}")
    add("")
    add("injection test (real off-pulse backgrounds)")
    add("  injected width (MHz)        : " + " ".join(f"{w:5.0f}" for w in res.recovery.widths))
    add("  fraction passing the rule   : " + " ".join(f"{d:5.2f}" for d in res.recovery.detected))
    for e in envs:
        add(f"  recovered median, env {e:>3}  : " + " ".join(f"{m:5.1f}" for m in res.recovery.percentiles[e][:, 1]))
    add("")
    add("calibrated width (injected width that reproduces the measurement)")
    for e in envs:
        c = res.calibrated[e]
        add(f"  env {e:>3}: {c['best']:5.1f} MHz   68% range {c['lo']:.1f} - {c['hi']:.1f}   {c['flag']}")
    best = np.array([c["best"] for c in res.calibrated.values()], float)
    add(f"  mean {np.nanmean(best):.1f} MHz, spread across scales {np.nanmin(best):.1f} - {np.nanmax(best):.1f}")
    add("  (the three scales use the same data, so this is a robustness range, not a confidence interval)")
    return "\n".join(lines)
