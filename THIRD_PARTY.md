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
| patsy | 1.0.3 | Patsy Developers | BSD-2-Clause | https://github.com/pydata/patsy |
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

Test-only (not in the image): pytest (MIT).

Formats followed (not code): NWB 2.x (Rübel et al. 2022) for the optional raw traces, the BIDS-style
`events.tsv` of stim-events, and the `series.csv` contract of the physiology workflows.
