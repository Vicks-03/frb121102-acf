"""Figures for the report. Each function returns a matplotlib Figure."""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from .acf import gaussian
from .clean import rebin

C_LATE, C_EARLY, C_NULL = "#D55E00", "#0072B2", "0.6"   # colour-blind-safe


def set_style():
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "axes.linewidth": 0.8,
    })


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")


def dynamic_spectrum(z, freq, time_ms, late, early, params):
    """Cleaned dynamic spectrum and frequency-averaged profile."""
    z_small = rebin(z, 32, 16)
    f_small = freq[: len(freq) // 32 * 32].reshape(-1, 32).mean(axis=1)
    prof = np.nanmean(z, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]}, constrained_layout=True)
    im = ax1.imshow(z_small, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, time_ms[-1], f_small[0], f_small[-1]],
                    vmin=np.nanpercentile(z_small, 1), vmax=np.nanpercentile(z_small, 99.7))
    for x in (early[0], early[1], late[1]):
        ax1.axvline(x, color="w", ls="--", lw=0.7)
    ax1.set(ylim=(params.f_lo, params.f_hi), ylabel="Frequency (MHz)")
    ax1.set_title("(a) Cleaned dynamic spectrum", loc="left")
    fig.colorbar(im, ax=[ax1, ax2], pad=0.01, label="Intensity (noise units, rebinned)")

    ax2.plot(time_ms, prof, color="0.75", lw=0.4)
    ax2.plot(time_ms, uniform_filter1d(np.nan_to_num(prof), 10), color="k", lw=1)
    ax2.axvspan(*early, color=C_EARLY, alpha=0.2, label="early window")
    ax2.axvspan(*late, color=C_LATE, alpha=0.2, label="late window")
    ax2.set(xlim=(0, 12), xlabel="Time since start of snapshot (ms)", ylabel="Mean intensity")
    ax2.set_title("(b) Frequency-averaged profile (black: 0.1 ms smoothing)", loc="left")
    ax2.legend(loc="upper right")
    return fig


def acf_fits(res, params):
    envs = params.env_scales
    fig, axs = plt.subplots(1, len(envs), figsize=(3.4 * len(envs), 3.2), sharey=True, constrained_layout=True)
    for ax, e, tag in zip(np.atleast_1d(axs), envs, "abcdefgh"):
        f = res.fits[e]
        k = int(150 / (f.lags[1] - f.lags[0]))
        sel = (f.lags > 0) & (f.lags <= params.fit_max)
        ax.plot(f.lags[1:k], f.acf[1:k], "k.", ms=3, label="ACF (zero lag omitted)")
        ax.plot(f.lags[sel], gaussian(f.lags[sel], f.amplitude, f.width), color=C_LATE, lw=1.6, label="Gaussian fit")
        ax.axvline(f.width, color=C_LATE, ls=":", lw=1)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.text(0.97, 0.95, f"HWHM = {f.width:.1f} MHz\nA = {f.amplitude:.2f}",
                transform=ax.transAxes, ha="right", va="top")
        ax.set_title(f"({tag}) envelope scale {e} MHz", loc="left")
        ax.set_xlabel("Frequency lag (MHz)")
    np.atleast_1d(axs)[0].set_ylabel("Normalised ACF")
    np.atleast_1d(axs)[0].legend(loc="center right")
    return fig


def injection(res, params):
    envs = params.env_scales
    rec = res.recovery
    lim = float(rec.widths.max()) + 2
    fig, axs = plt.subplots(1, len(envs), figsize=(3.4 * len(envs), 3.4), sharey=True, constrained_layout=True)
    for ax, e, tag in zip(np.atleast_1d(axs), envs, "abcdefgh"):
        q = rec.percentiles[e]
        cal = res.calibrated[e]
        ax.fill_between(rec.widths, q[:, 0], q[:, 2], color="tab:blue", alpha=0.3, label="recovered, 16-84%")
        ax.plot(rec.widths, q[:, 1], color="tab:blue", label="recovered, median")
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="perfect recovery")
        ax.axhline(res.fits[e].width, color=C_LATE, label=f"measured: {res.fits[e].width:.1f} MHz")
        if np.isfinite(cal["lo"]) and np.isfinite(cal["hi"]):
            ax.axvspan(cal["lo"], cal["hi"], color=C_LATE, alpha=0.15)
        if np.isfinite(cal["best"]):
            ax.axvline(cal["best"], color=C_LATE, ls=":", label=f"corrected: {cal['best']:.1f} MHz")
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Injected correlation width (MHz)")
        ax.set_title(f"({tag}) envelope scale {e} MHz", loc="left")
        ax.legend(loc="upper left")
    np.atleast_1d(axs)[0].set_ylabel("Recovered width (MHz)")
    return fig


def acf_comparison(res, early_fits, params):
    """Late burst, early burst and each empty window, at the middle envelope scale."""
    env = params.env_scales[len(params.env_scales) // 2]
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)

    null_acfs = []
    for i, w in enumerate(res.empty):
        f = w["fits"][env]
        k = (f.lags > 0) & (f.lags <= 150)
        null_acfs.append(f.acf)
        ax.plot(f.lags[k], f.acf[k], color=C_NULL, alpha=0.6, lw=0.8,
                label="individual empty windows" if i == 0 else None)
    ax.plot(f.lags[k], np.nanmean(null_acfs, axis=0)[k], "k--", lw=1.5, label="mean of empty windows")

    for fit, color, name in [(early_fits[env], C_EARLY, "early burst"), (res.fits[env], C_LATE, "late burst")]:
        k = (fit.lags > 0) & (fit.lags <= 150)
        ax.plot(fit.lags[k], fit.acf[k], color=color, lw=2, label=name)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set(xlabel="Frequency lag (MHz)", ylabel="Normalised ACF")
    ax.set_title(f"Envelope scale {env} MHz", loc="left")
    ax.legend(loc="upper right")
    return fig


def envelope_dependence(res, early_fits, params):
    envs = params.env_scales
    x = np.arange(len(envs))
    best = np.array([res.calibrated[e]["best"] for e in envs])
    lo = np.array([res.calibrated[e]["lo"] for e in envs])
    hi = np.array([res.calibrated[e]["hi"] for e in envs])

    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    ax.plot(x - 0.06, [early_fits[e].width for e in envs], "o-", color=C_EARLY, label="early, raw")
    ax.plot(x - 0.06, [res.fits[e].width for e in envs], "s-", color=C_LATE, label="late, raw")
    ax.errorbar(x + 0.06, best, yerr=[best - lo, hi - best], fmt="*", ms=11, color=C_LATE,
                mfc="w", capsize=3, label="late, injection-corrected (68%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in envs])
    ax.set(xlabel="Envelope smoothing scale (MHz)", ylabel="ACF HWHM (MHz)")
    ax.legend(loc="upper left")
    return fig
