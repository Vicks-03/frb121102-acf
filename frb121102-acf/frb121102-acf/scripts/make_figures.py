"""Make the report figures for one burst.

    python scripts/make_figures.py configs/11A.json

Needs "early_window_ms" in the config for the early/late comparison.
Runs the whole analysis again (a few minutes), then writes PNG and PDF files to figures/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frbacf import figures
from frbacf.clean import normalise
from frbacf.config import load_config
from frbacf.io import load_burst
from frbacf.params import Params
from frbacf.pipeline import analyse_window, raw_fits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if "early_window_ms" not in cfg:
        sys.exit("this config has no early_window_ms; the early/late figures need it")
    params = Params()
    figures.set_style()

    burst = load_burst(cfg["file"])
    t = burst.time_ms
    z, _ = normalise(burst.intensity, burst.freq, t, cfg["off_window_ms"], params)

    res = analyse_window(burst, cfg["burst_window_ms"], cfg["off_window_ms"], params,
                         n_trials=args.n_trials, seed=args.seed, label=cfg["label"])
    early = raw_fits(z, burst.freq, t, cfg["early_window_ms"], params)

    prefix = cfg["label"]
    made = {
        "dynamic_spectrum": figures.dynamic_spectrum(z, burst.freq, t, cfg["burst_window_ms"],
                                                      cfg["early_window_ms"], params),
        "acf_fits": figures.acf_fits(res, params),
        "injection_recovery": figures.injection(res, params),
        "acf_comparison": figures.acf_comparison(res, early, params),
        "envelope_dependence": figures.envelope_dependence(res, early, params),
    }
    for name, fig in made.items():
        figures.save(fig, args.out, f"{prefix}_{name}")
        plt.close(fig)
        print(f"wrote {args.out}/{prefix}_{name}.png and .pdf")

    print("\nnumbers behind the figures")
    for e in params.env_scales:
        c = res.calibrated[e]
        print(f"  env {e:>3}: late raw {res.fits[e].width:5.1f}  corrected {c['best']:5.1f} "
              f"({c['lo']:.1f}-{c['hi']:.1f})  early raw {early[e].width:5.1f}")


if __name__ == "__main__":
    main()
