# Third-party components

The Docker image bundles these Python packages (exact versions in `requirements.txt`), installed
from PyPI wheels on `python:3.12.16-slim-bookworm` (Debian bookworm with CPython 3.12, PSF licence).
No data, models or other people's analysis code are bundled. The response windows, the dose-response
model selection, thresholds, the response surface and the neural-fulcrum crossing are this app's own
code (Apache-2.0), written from the published methods cited in the README.

| Component | Version | Authors / copyright holder | Licence | URL |
|---|---|---|---|---|
| NumPy | 2.5.4 | NumPy Developers | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | Enthought, Inc. and the SciPy Developers | BSD-3-Clause | https://scipy.org |
| pandas | 3.1.1 | AQR Capital Management, Lambda Foundry, the PyData Development Team and open source contributors | BSD-3-Clause | https://pandas.pydata.org |
| statsmodels | 0.15.0 | Jonathan E. Taylor, the SciPy Developers and the statsmodels Developers | BSD-3-Clause | https://www.statsmodels.org |
| patsy | 1.0.3 | Patsy Developers | BSD-2-Clause (`patsy.compat`, derived from the Python standard library: PSF-2.0) | https://github.com/pydata/patsy |
| h5py | 3.16.0 | Andrew Collette and the h5py contributors | BSD-3-Clause | https://github.com/h5py/h5py |
| Matplotlib | 3.11.2 | John D. Hunter, Michael Droettboom and the Matplotlib Development Team | Matplotlib License (PSF-based) | https://matplotlib.org |
| contourpy | 1.4.0 | ContourPy Developers | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | Matplotlib project | BSD-3-Clause | https://github.com/matplotlib/cycler |
| fonttools | 4.68.0 | Just van Rossum and the fontTools contributors | MIT | https://github.com/fonttools/fonttools |
| kiwisolver | 1.5.1 | Nucleic Development Team | BSD-3-Clause | https://github.com/nucleic/kiwi |
| packaging | 26.3 | Donald Stufft and contributors | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| Pillow | 13.0.0 | Secret Labs AB, Fredrik Lundh, Jeffrey A. Clark and contributors | MIT-CMU | https://github.com/python-pillow/Pillow |
| pyparsing | 3.3.3 | Paul McGuire | MIT | https://github.com/pyparsing/pyparsing |
| python-dateutil | 4.0.0 | Paul Ganssle and the dateutil contributors | BSD-3-Clause | https://github.com/dateutil/dateutil |
| six | 1.17.0 | Benjamin Peterson | MIT | https://github.com/benjaminp/six |

statsmodels 0.15 lists `formulaic` as a dependency but needs only one of patsy or formulaic; the
image ships patsy (MixedLM is used without formulas).

## Native libraries inside the wheels

The binary wheels above carry compiled libraries of their own (under `numpy.libs/`, `scipy.libs/`, `h5py.libs/`,
`pillow.libs/`, or linked into the extension modules). Their licence texts ship in each package's
`*.dist-info/licenses` (or `LICENSE`) folder in the image.

| Library | Inside | Copyright holder | Licence |
|---|---|---|---|
| OpenBLAS (with LAPACK) | NumPy and SciPy (`libscipy_openblas`) | The OpenBLAS Project; LAPACK: University of Tennessee and collaborators | BSD-3-Clause; LAPACK BSD-3-Clause-Open-MPI |
| GCC runtime (`libgfortran`) | NumPy and SciPy | Free Software Foundation | GPL-3.0-or-later WITH GCC-exception-3.1 (the exception allows distribution with non-GPL code) |
| libquadmath | NumPy and SciPy | Free Software Foundation | LGPL-2.1-or-later (a separate shared library, replaceable) |
| HDF5 (`libhdf5`, `libhdf5_hl`) | h5py | The HDF Group | BSD-3-Clause |
| libaec / libsz (SZIP-compatible) | h5py | Deutsches Klimarechenzentrum GmbH (DKRZ) | BSD-2-Clause |
| LZF filter | h5py | Marc Alexander Lehmann | BSD-2-Clause |
| FreeType 2.14.3 | Pillow (`libfreetype`) and Matplotlib (`ft2font`, statically linked) | The FreeType Project | FreeType License (FTL) (Matplotlib: FTL OR GPL-2.0-or-later) |
| HarfBuzz | Pillow (`libharfbuzz`) | HarfBuzz authors | MIT ("Old MIT") |
| libjpeg-turbo | Pillow (`libjpeg`) | Independent JPEG Group; libjpeg-turbo authors | IJG License, BSD-3-Clause, Zlib |
| libpng | Pillow (`libpng16`) | Cosmin Truta and the PNG Reference Library authors | libpng License / PNG Reference Library License v2 |
| zlib, zstd, brotli, xz (`liblzma`) | Pillow | their authors (Jean-loup Gailly and Mark Adler; Meta Platforms; Google; Lasse Collin) | Zlib; BSD-3-Clause; MIT; 0BSD / public domain |
| LittleCMS (`liblcms2`), libtiff, OpenJPEG, libwebp, libavif, libxcb, libXau | Pillow | Marti Maria; Sam Leffler and Silicon Graphics; the OpenJPEG authors; Google; Alliance for Open Media; the X.Org Foundation | MIT; libtiff (BSD-like); BSD-2-Clause; BSD-3-Clause; BSD-2-Clause; MIT; MIT |
| Qhull | Matplotlib (`_qhull`) | C.B. Barber and The Geometry Center | Qhull License |
| DejaVu, STIX, Computer Modern and Last Resort fonts | Matplotlib (`mpl-data/fonts`) | Bitstream / DejaVu authors; STI Pub Companies; American Mathematical Society (BaKoMa); Unicode, Inc. | Bitstream Vera / DejaVu licence; SIL OFL 1.1; Knuth-style free licence; SIL OFL 1.1 |

Credit lines these licences ask for:

- Portions of this software are copyright © 2026 The FreeType Project (www.freetype.org). All rights
  reserved.
- This software is based in part on the work of the Independent JPEG Group.

Test-only (not in the image): pytest (MIT).

Formats followed (not code): NWB 2.x (Rübel et al. 2022) for the optional raw traces, the BIDS-style
`events.tsv` of stim-events, and the `series.csv` contract of the physiology workflows.
