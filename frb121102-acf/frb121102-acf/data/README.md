# Data

The burst files are not stored in this repository (about 320 MB each).
Download them from the Breakthrough Listen page for FRB 121102:

https://seti.berkeley.edu/frb121102/technical.html

and put them in this folder with their original names, for example:

    data/11A_16sec.calib.4p
    data/11E_344sec.calib.4p
    data/11K_769sec.calib.4p

Use the `.calib.4p` files. The much smaller `_tot.fits` files (11L, 11P, 11R) have
a different layout and were not used.

Each file should be 322,367,040 bytes. If yours is smaller the download was cut
short; `load_burst` raises an error in that case instead of failing later.

Files used in the report: 11A, 11B, 11C, 11E, 11K. Only 11A, 11E and 11K were
analysed beyond the screening step (see the main README).
