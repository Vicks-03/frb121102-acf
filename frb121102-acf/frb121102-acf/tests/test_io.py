"""Tests the reshaping logic in load_burst against a stand-in for astropy.

The stand-in mimics the small part of astropy.io.fits that load_burst uses, so
these tests do not need astropy or a real data file.
"""
import sys
import types

import numpy as np


class FakeHDU:
    def __init__(self, header, data):
        self.header, self.data = header, data


class FakeHDUList:
    def __init__(self, hdus):
        self._hdus = hdus

    def __getitem__(self, name):
        return self._hdus[name]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def install_fake_astropy(monkeypatch, hdus):
    fits = types.ModuleType("astropy.io.fits")
    fits.open = lambda path, memmap=False: FakeHDUList(hdus)
    io = types.ModuleType("astropy.io")
    io.fits = fits
    astropy = types.ModuleType("astropy")
    astropy.io = io
    monkeypatch.setitem(sys.modules, "astropy", astropy)
    monkeypatch.setitem(sys.modules, "astropy.io", io)
    monkeypatch.setitem(sys.modules, "astropy.io.fits", fits)


def make_subint(npol=4, nchan=6, nbin=10, pol_type="IQUV", descending=False):
    rng = np.random.default_rng(0)
    raw = rng.integers(-100, 100, size=(1, npol, nchan, nbin)).astype(np.int16)
    # distinct scale/offset per (pol, channel), stored pol-major as in PSRFITS
    scl = (1.0 + np.arange(npol * nchan, dtype=np.float32)).reshape(1, -1)
    ofs = (10.0 * np.arange(npol * nchan, dtype=np.float32)).reshape(1, -1)
    wts = np.ones((1, nchan), dtype=np.float32)
    wts[0, 2] = 0
    freq = 4600.0 + 10.0 * np.arange(nchan)
    if descending:
        freq = freq[::-1]
    data = {"DATA": raw, "DAT_SCL": scl, "DAT_OFFS": ofs, "DAT_WTS": wts,
            "DAT_FREQ": freq[None, :], "TSUBINT": np.array([0.02])}
    header = {"NPOL": npol, "NCHAN": nchan, "NBIN": nbin, "POL_TYPE": pol_type,
              "NAXIS1": 1, "NAXIS2": 1}
    return FakeHDU(header, data), raw, scl, ofs


class DictHeader(dict):
    pass


def test_load_burst_applies_scale_and_offset_per_pol_and_channel(monkeypatch, tmp_path):
    from frbacf.io import load_burst
    sub, raw, scl, ofs = make_subint()
    sub.header = DictHeader(sub.header)
    install_fake_astropy(monkeypatch, {"SUBINT": sub})
    path = tmp_path / "fake.fits"
    path.write_bytes(b"x" * 10)

    b = load_burst(str(path))
    expected = raw[0, 0].astype(np.float32) * scl[0, :6, None] + ofs[0, :6, None]   # pol 0 uses the first nchan values
    got = b.intensity.copy()
    assert np.allclose(got[[0, 1, 3, 4, 5]], expected[[0, 1, 3, 4, 5]])
    assert np.isnan(got[2]).all()                          # zero-weight channel is flagged
    assert abs(b.dt - 0.02 / 10) < 1e-12
    assert b.intensity.shape == (6, 10)


def test_load_burst_flips_descending_frequencies(monkeypatch, tmp_path):
    from frbacf.io import load_burst
    sub, raw, scl, ofs = make_subint(descending=True)
    sub.header = DictHeader(sub.header)
    install_fake_astropy(monkeypatch, {"SUBINT": sub})
    path = tmp_path / "fake.fits"
    path.write_bytes(b"x" * 10)
    b = load_burst(str(path))
    assert np.all(np.diff(b.freq) > 0)
    # first output channel is the last stored channel
    exp_last = raw[0, 0, -1].astype(np.float32) * scl[0, 5] + ofs[0, 5]
    assert np.allclose(b.intensity[0], exp_last)


def test_load_burst_sums_aa_bb_when_not_stokes(monkeypatch, tmp_path):
    from frbacf.io import load_burst
    sub, raw, scl, ofs = make_subint(pol_type="AABBCRCI")
    sub.header = DictHeader(sub.header)
    install_fake_astropy(monkeypatch, {"SUBINT": sub})
    path = tmp_path / "fake.fits"
    path.write_bytes(b"x" * 10)
    b = load_burst(str(path))
    aa = raw[0, 0].astype(np.float32) * scl[0, :6, None] + ofs[0, :6, None]
    bb = raw[0, 1].astype(np.float32) * scl[0, 6:12, None] + ofs[0, 6:12, None]
    ok = [0, 1, 3, 4, 5]
    assert np.allclose(b.intensity[ok], (aa + bb)[ok])


def test_truncated_file_is_reported(monkeypatch, tmp_path):
    from frbacf.io import load_burst
    sub, *_ = make_subint()
    sub.header = DictHeader(sub.header)
    sub.header["NAXIS1"] = 10_000
    install_fake_astropy(monkeypatch, {"SUBINT": sub})
    path = tmp_path / "fake.fits"
    path.write_bytes(b"x" * 10)
    try:
        load_burst(str(path))
    except IOError as err:
        assert "truncated" in str(err)
    else:
        raise AssertionError("expected an IOError")
