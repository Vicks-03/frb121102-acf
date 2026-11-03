"""Run the full analysis on one burst window.

    python scripts/measure.py configs/11A.json
    python scripts/measure.py configs/11A.json --n-trials 4      # faster, noisier calibration

Prints a report and writes results/<label>_summary.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frbacf.config import load_config
from frbacf.io import load_burst, scale_order_check
from frbacf.params import Params
from frbacf.pipeline import analyse_window, summarise
from frbacf.report import format_report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--n-trials", type=int, default=8, help="injections per width and background window")
    ap.add_argument("--n-white", type=int, default=200, help="white-noise spectra for the false-alarm check")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-audit", action="store_true", help="skip the calibration-array ordering check")
    args = ap.parse_args()

    cfg = load_config(args.config)
    params = Params()

    print(f"loading {cfg['file']}")
    burst = load_burst(cfg["file"])
    print(f"{burst.intensity.shape[0]} channels x {burst.intensity.shape[1]} bins, "
          f"dt = {burst.dt * 1e6:.2f} us, channel = {burst.chan_bw:.4f} MHz")

    if not args.skip_audit:
        used, wrong = scale_order_check(cfg["file"], cfg["off_window_ms"])
        print(f"bandpass roughness: adopted ordering {used:.4f}, wrong ordering {wrong:.4f} "
              f"(the adopted one should be much smaller)")

    res = analyse_window(burst, cfg["burst_window_ms"], cfg["off_window_ms"], params,
                         n_trials=args.n_trials, n_white=args.n_white, seed=args.seed,
                         label=cfg["label"])
    print(format_report(res, params))

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"{cfg['label']}_summary.json")
    with open(out, "w") as fh:
        json.dump(summarise(res), fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
