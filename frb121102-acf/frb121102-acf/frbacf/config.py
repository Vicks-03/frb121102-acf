"""Small JSON config files, one per burst."""
import json

REQUIRED = ("label", "file", "burst_window_ms", "off_window_ms")


def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")
    return cfg
