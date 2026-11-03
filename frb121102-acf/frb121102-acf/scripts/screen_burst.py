"""Look at one or more burst files and suggest windows for the full analysis.

    python scripts/screen_burst.py data/11B_263sec.calib.4p data/11C_284sec.calib.4p

Plots are written to results/. The S/N printed here uses a whole-file baseline
and an automatic window, so treat it as approximate.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frbacf.clean import rebin
from frbacf.io import load_burst
from frbacf.params import Params
from frbacf.screen import suggest_windows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    params = Params()
    os.makedirs(args.out, exist_ok=True)

    for path in args.files:
        print("=" * 70)
        print(path)
        burst = load_burst(path)
        s = suggest_windows(burst, params)
        print(f"peak at {s['peak_ms']:.2f} ms ({s['peak_sigma']:.1f} sigma in the smoothed profile)")
        print(f"suggested burst window: {s['burst_window'][0]:.1f} - {s['burst_window'][1]:.1f} ms")
        print(f"suggested off window  : {s['off_window'][0]:.1f} - {s['off_window'][1]:.1f} ms")
        print(f"approximate S/N       : {s['snr']:.1f}")
        if s["off_window"][1] - s["off_window"][0] < 6:
            print("warning: the off region is short, so there will be few empty windows")

        t = burst.time_ms
        z_small = rebin(s["z"], 16, 8)
        f_small = burst.freq[: len(burst.freq) // 16 * 16].reshape(-1, 16).mean(axis=1)

        fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
        ax[0].imshow(z_small, aspect="auto", origin="lower",
                     extent=[0, t[-1], f_small[0], f_small[-1]],
                     vmin=-1, vmax=np.nanpercentile(z_small, 99.8))
        ax[0].set(ylim=(params.f_lo, params.f_hi), xlabel="Time (ms)", ylabel="Frequency (MHz)",
                  title=os.path.basename(path))
        ax[1].plot(t, s["profile"], lw=0.5)
        ax[1].plot(t, s["smoothed"], "r", lw=1.5)
        ax[1].axvspan(*s["burst_window"], color="orange", alpha=0.25)
        ax[1].axvspan(*s["off_window"], color="gray", alpha=0.15)
        ax[1].set(xlabel="Time (ms)", ylabel="Baseline-subtracted profile")
        fig.tight_layout()
        name = os.path.basename(path).split(".")[0]
        fig.savefig(os.path.join(args.out, f"screen_{name}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
