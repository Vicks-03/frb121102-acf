"""Reading Breakthrough Listen PSRFITS burst snapshots.

Each file is one 20 ms snapshot with the layout (nsub=1, npol, nchan, nbin).
astropy is imported inside the functions so the rest of the package can be
used (and tested) without it.
"""
import gc
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Burst:
    path: str
    intensity: np.ndarray    # (nchan, ntime) float32, NaN where the channel was flagged
    freq: np.ndarray         # MHz, ascending
    dt: float                # seconds per time bin

    @property
    def time_ms(self):
        return np.arange(self.intensity.shape[1]) * self.dt * 1e3

    @property
    def chan_bw(self):
        return float(np.median(np.diff(self.freq)))


def load_burst(path):
    """Return total intensity as a Burst. POL_TYPE IQUV -> pol 0 is Stokes I."""
    from astropy.io import fits

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with fits.open(path, memmap=False) as hdul:
        sub = hdul["SUBINT"]
        hdr = sub.header

        needed = hdr["NAXIS1"] * hdr["NAXIS2"]
        if os.path.getsize(path) < needed:
            raise IOError(f"{path} looks truncated ({os.path.getsize(path)} bytes, "
                          f"header expects at least {needed}). Download it again.")

        npol, nchan, nbin = hdr["NPOL"], hdr["NCHAN"], hdr["NBIN"]
        pol_type = str(hdr.get("POL_TYPE", "")).strip()

        raw = sub.data["DATA"]                                   # (nsub, npol, nchan, nbin)
        scale = np.asarray(sub.data["DAT_SCL"], dtype=np.float32).reshape(npol, nchan, 1)
        offset = np.asarray(sub.data["DAT_OFFS"], dtype=np.float32).reshape(npol, nchan, 1)
        weights = np.asarray(sub.data["DAT_WTS"], dtype=np.float32)[0]
        freq = np.asarray(sub.data["DAT_FREQ"], dtype=float)[0]
        dt = float(np.asarray(sub.data["TSUBINT"])[0]) / nbin

        def one_pol(p):
            return np.asarray(raw[0, p], dtype=np.float32) * scale[p] + offset[p]

        if pol_type.startswith("AABB"):
            stokes_i = one_pol(0) + one_pol(1)                   # I = AA + BB
        else:
            stokes_i = one_pol(0)
        del raw

    stokes_i[weights == 0, :] = np.nan
    if freq[0] > freq[-1]:
        freq, stokes_i = freq[::-1], stokes_i[::-1].copy()
    gc.collect()
    return Burst(path=path, intensity=stokes_i.astype(np.float32), freq=freq, dt=dt)


def scale_order_check(path, off_win):
    """Compare bandpass roughness for the scale/offset ordering used in load_burst
    and for a deliberately wrong (channel-major) ordering.

    Returns (rough_used, rough_wrong). The adopted ordering should give a much
    smaller number. This is a consistency check, not a proof of the file layout.
    """
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        sub = hdul["SUBINT"]
        hdr = sub.header
        npol, nchan, nbin = hdr["NPOL"], hdr["NCHAN"], hdr["NBIN"]
        raw = np.asarray(sub.data["DATA"][0, 0], dtype=np.float32)
        scl = np.asarray(sub.data["DAT_SCL"], dtype=np.float32).ravel()
        ofs = np.asarray(sub.data["DAT_OFFS"], dtype=np.float32).ravel()
        wts = np.asarray(sub.data["DAT_WTS"], dtype=np.float32)[0]
        dt = float(np.asarray(sub.data["TSUBINT"])[0]) / nbin

    t_ms = np.arange(nbin) * dt * 1e3
    off = (t_ms >= off_win[0]) & (t_ms <= off_win[1])
    good = wts > 0

    def roughness(s, o):
        spec = (raw * s[:, None] + o[:, None])[:, off].mean(axis=1)[good]
        return np.median(np.abs(np.diff(spec, 2))) / np.median(np.abs(spec))

    used = roughness(scl[:nchan], ofs[:nchan])
    wrong = roughness(scl.reshape(nchan, npol)[:, 0], ofs.reshape(nchan, npol)[:, 0])
    return float(used), float(wrong)
