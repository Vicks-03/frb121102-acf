# frb121102-acf

Frequency autocorrelation analysis of burst spectra from the repeating fast radio
burst FRB 121102, using the public Breakthrough Listen observations made with the
Green Bank Telescope (4-8 GHz).

I wrote this while learning how scintillation bandwidths are measured, starting from
Spitler et al. (2018), who measured one at 4.85 GHz with Effelsberg. Their data are
not public, so the aim here is not to reproduce their number. It is to implement the
same kind of measurement, find out how reliable it is on a different data set, and
say clearly what the result does and does not show.

**Short version of the result.** In the later part of burst 11A the spectrum has a
frequency correlation width of about 10 MHz near 6 GHz (calibration range roughly
8-13 MHz). That is the right order of magnitude for Galactic scintillation, but these
data do not show that it is scintillation. Of the five bursts I looked at, only 11A
was bright enough to give a stable width. The write-up has the details.

## What the code does

For one burst and one time window:

1. Read the PSRFITS snapshot (Stokes I) and normalise every frequency channel by its
   own off-pulse noise. Channels with unusual noise, and the band edges, are masked.
2. Average the burst window in time and groups of 8 channels in frequency.
3. Subtract a smoothed copy of the spectrum (the "envelope"), so that only the fine
   structure is left. This is done at three smoothing scales: 80, 150 and 300 MHz.
4. Compute the frequency ACF and fit a Gaussian to lags up to 60 MHz, leaving out
   zero lag. The half-width at half-maximum (HWHM) is the reported width.
5. Apply a detection rule: fitted amplitude at least 0.1 and HWHM at least 3 MHz, at
   all three scales.

The measured width depends on the envelope scale, so the code also calibrates it:

6. Inject scintillation-like patterns of known width into real burst-free spectra from
   the same file, run exactly the same steps, and use the injected-versus-recovered
   curve to convert the measured width into a corrected one.
7. Run the same fit on empty windows and on pure noise, and repeat the measurement
   with a different normalisation, as checks.

## Setup

    git clone https://github.com/<your-username>/frb121102-acf.git
    cd frb121102-acf
    pip install -r requirements.txt

Download the data as described in [data/README.md](data/README.md). Run everything
from the repository root.

## Usage

Look at a burst and get suggested windows (S/N here is approximate):

    python scripts/screen_burst.py data/11E_344sec.calib.4p

Run the full analysis from a config file:

    python scripts/measure.py configs/11A.json

This prints a report and writes `results/11A_summary.json`. A full run takes a few
minutes; `--n-trials 4` makes it faster with a noisier calibration, and
`--n-white 500` uses more noise-only spectra for the false-alarm check.

Make the report figures (needs `early_window_ms` in the config):

    python scripts/make_figures.py configs/11A.json

This writes PNG and PDF files to `figures/`.

A config looks like this:

    {
      "label": "11A",
      "file": "data/11A_16sec.calib.4p",
      "burst_window_ms": [2.7, 5.0],
      "early_window_ms": [1.0, 2.7],
      "off_window_ms": [8.0, 21.0]
    }

The off-pulse window sets each channel's noise level and supplies the empty windows,
so it must be free of burst emission. Windows are in milliseconds from the start of
the 20 ms snapshot.

## Results

Numbers from my notebook runs on burst 11A (late window, 2.7-5.0 ms, centre frequency
about 6.0 GHz):

| Envelope scale | Raw HWHM | Corrected HWHM |
|---|---|---|
| 80 MHz | 5.8 MHz | 8.6 MHz |
| 150 MHz | 7.7 MHz | 9.8 MHz |
| 300 MHz | 9.4 MHz | 10.7 MHz |

The three scales use the same data, so the spread (about 8-13 MHz including the
calibration ranges) is a robustness range and not a confidence interval.

The other bursts:

| Burst | S/N (approx.) | Result |
|---|---|---|
| 11E | 17 | Fails the detection rule (amplitude 0.07 at one scale); no stable width |
| 11K | 11 | Fails the detection rule; no usable width |
| 11C, 11B | about 7 and 5 | Screened out at low S/N, not analysed further |

The early part of 11A (1.0-2.7 ms) also gave no stable width: it came out at 3.8, 13.6
and 31.2 MHz for the three scales.

The code in this repository is a cleaned-up restructuring of the notebook code, with
the empty windows now starting at the beginning of the off-pulse window. The
calibration is random, so rerunning gives numbers that can differ from the table by a
few tenths of a MHz.

## Limitations

- One burst gives a usable width, from one night with one telescope. Nothing here
  says anything about change with time or frequency.
- The detection thresholds (amplitude 0.1, width 3 MHz) were chosen after looking at
  11A. Passing the rule is a consistency check for picking candidates, not a
  calibrated significance, and the noise tests are not a false-alarm probability.
- The injected patterns are idealised (Gaussian correlation, full modulation), and
  the calibration only holds as far as they resemble the real burst.
- Intrinsic spectral structure of the burst could produce a similar ACF, and these
  data cannot separate it from propagation effects.
- Both parts of 11A show an extra ACF bump at lags of about 85-110 MHz that I cannot
  explain. It is not present in the empty windows.
- The scaling to other frequencies in the report assumes width proportional to
  frequency to the power 4-4.4, which I did not test.

## Tests

    pip install -r requirements-dev.txt
    pytest -q

The tests use synthetic data: a burst with a known scintillation width, noise-only
spectra, and the array handling in the file reader (checked against a stand-in for
astropy, so no data file is needed). They do not replace running the pipeline on a
real file.

## Layout

    frbacf/
        params.py     analysis settings
        io.py         PSRFITS reader and the calibration-array ordering check
        clean.py      normalisation, masking, rebinning
        acf.py        window spectrum, envelope, ACF, Gaussian fit, detection rule
        inject.py     injection-recovery, empty-window and white-noise tests
        pipeline.py   full analysis of one window
        report.py     text report
        figures.py    report figures
        screen.py     window suggestions for new bursts
        config.py     config file loading
    scripts/          command-line entry points
    configs/          one JSON file per burst
    tests/
    data/             put the downloaded burst files here

## Reference

Spitler, L. G., et al. 2018, Detection of Bursts from FRB 121102 with the Effelsberg
100-m Radio Telescope at 5 GHz and the Role of Scintillation, ApJ, 863, 150
(arXiv:1807.03722).

Data: Breakthrough Listen, https://seti.berkeley.edu/frb121102/technical.html

## License

MIT, see [LICENSE](LICENSE).
