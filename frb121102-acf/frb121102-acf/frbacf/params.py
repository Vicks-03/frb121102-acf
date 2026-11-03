"""Analysis settings in one place.

A few of these (a_min, min_width) were chosen after looking at burst 11A.
See the README for what that means for how the results should be read.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    f_lo: float = 4750.0          # MHz, band edges below/above this are dropped
    f_hi: float = 8100.0
    nav: int = 8                  # channels averaged before the ACF (8 x 0.183 = ~1.47 MHz)
    env_scales: tuple = (80, 150, 300)   # MHz, smoothing scales for the envelope subtraction
    island_scale: float = 300.0   # MHz, smoothing used for the "island" shape in injections
    fit_max: float = 60.0         # MHz, ACF is fitted out to this lag
    a_min: float = 0.1            # minimum fitted ACF amplitude for a detection
    min_width: float = 3.0        # MHz, minimum fitted HWHM for a detection
    smooth_noise_mhz: float = 50.0  # scale for the alternative normalisation


DEFAULT_WIDTHS = (4, 6, 8, 10, 12, 16, 20, 30)   # MHz, injected correlation widths
