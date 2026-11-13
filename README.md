# stim-dose-response

How heart rate, blood pressure, breathing (or any other beat- or breath-derived series) respond to
each nerve stimulation, and how the response grows with the stimulation dose. The dose-response
analysis is Pennsieve's own implementation of published methods, and the credit for those methods
is their authors': the **neural fulcrum** is the concept of **Jeffrey L. Ardell, Heath Nier, Matthew
Hammer, E. Marie Southerland, Christopher L. Ardell, Eric Beaumont, Bruce H. KenKnight and J. Andrew
Armour** ([Ardell et al. 2017, J Physiol](https://doi.org/10.1113/JP274678));
model selection uses the information criterion of **H. Akaike** ([Akaike 1974](https://doi.org/10.1109/TAC.1974.1100705))
with the small-sample correction of **C. M. Hurvich and C.-L. Tsai** ([Hurvich & Tsai 1989](https://doi.org/10.1093/biomet/76.2.297)),
and the fulcrum interval the bootstrap of **B. Efron** ([Efron 1979](https://doi.org/10.1214/aos/1176344552)).
The curve fits run on **SciPy** (`scipy.optimize.curve_fit`; Pauli Virtanen, Ralf Gommers, Travis E.
Oliphant and the SciPy contributors, [github.com/scipy/scipy](https://github.com/scipy/scipy),
[Virtanen et al. 2020](https://doi.org/10.1038/s41592-019-0686-2)) and the pooled mixed model on
**statsmodels** (Skipper Seabold, Josef Perktold and the statsmodels developers,
[github.com/statsmodels/statsmodels](https://github.com/statsmodels/statsmodels),
[Seabold & Perktold 2010](https://doi.org/10.25080/Majora-92bf1922-011)). Pennsieve packages these
methods for any lab's data; it did not develop them. The demo data are **Pennsieve Discover 349**
(pig vagus nerve stimulation over a current x frequency grid; Oliver Armitage, Elvijs Sarkans, Lorenz
Wernisch, Pascal Fortier-Poisson, Antonin Berthon and Olivier Tessier-Larivière,
[doi:10.26275/dm5l-tspb](https://doi.org/10.26275/dm5l-tspb)) and **Pennsieve Discover 655** (rat
closed-loop heart-rate control by vagus stimulation and block; Shane A Bender, David B Green, Varun S
Thakkar, Hope L Zimmerman, Mohamed Elazab, Kevin L Kilgore and Tina L Vrabec,
[doi:10.26275/5cwo-mkpz](https://doi.org/10.26275/5cwo-mkpz)).

**Research outputs.** Responses, thresholds, EC50 values and fulcrum amplitudes describe the
recordings analysed. They are not clinical findings and not dosing recommendations.

## What it does

**For a physiologist.** Give it the stimulation trains that `stim-events` found and the heart-rate
and blood-pressure series that `ecg-hrv` and `bp-resp` computed. For every train it measures each
signal in a baseline window just before the train, during the train and just after it, and reports
the change, the percent change, the largest change (and when it happened), and how long the signal
took to come back. Trains that are too close to another train, or whose signal is mostly artefact,
are kept in the table but left out of the curves. For every parameter that was varied with the
others held fixed (e.g. current at each frequency), it plots the response against the parameter,
fits a sigmoid (Hill or Boltzmann), a straight line, or no dependence, whichever the data support
best, and reports the **threshold** (the lowest level whose mean response clearly exceeds the
baseline noise; marked as an upper bound when the level below it was not analysed), the **EC50** (the level giving half of the maximum response) and the maximum, with
95 % confidence intervals. When both amplitude and frequency were varied it draws the **response
surface**; when heart rate goes up at low amplitudes and down at high ones it reports the amplitude
where the response changes sign (the **neural fulcrum**, see the caveat below). With three or more
animals of one species it also fits the pooled curve with the animal as a random effect.

**For a data scientist.** Input: `events.tsv` (one row per train with its parameters) and any number
of long-format `series.csv` (`recording_id, time_s, variable, value, valid`) on a regular grid.
Every variable of every series is analysed, so a new upstream step joins by writing the same shape.
Per (train, variable): windowed means of valid samples; delta/pct change; peak of a centred moving
mean; recovery to within a fraction of the peak. Per (subject, trial type, x, group of the other
parameters, variable): four models fitted by bounded least squares to the per-train
deltas, chosen by AICc; threshold from level means against a data-driven criterion; t-based 95 %
intervals from the fit's covariance. Pooled (>= 3 subjects of one species): level estimates and their
standard errors from `MixedLM(delta ~ 0 + C(level), groups=subject)`, then the same model choice on
those estimates, weighted by their standard errors, with normal-quantile intervals (n = number of
levels). Fulcrum: the first sign change of the level means between levels that exceed the criterion
in opposite directions, linearly interpolated; its interval is a bootstrap of trains within levels.

## Ports

| Port | Direction | Content |
|---|---|---|
| `events` | input, required | stim-events' `events` output: `stim-events/events.tsv`. physio-import's own `events.tsv` (raw markers, next to `recordings.csv`) is never read. |
| `hrv` | input, optional | ecg-hrv's `hrv` output (`series.csv`: `hr_bpm`, `rr_ms`, `hr_device_bpm`), or any `series.csv` |
| `bp` | input, optional | bp-resp's `bp` output (`series.csv`: `sbp_mmhg`, `dbp_mmhg`, `map_mmhg`, `pp_mmhg`, `resp_rate_bpm`, `resp_amp`), or any `series.csv` |
| `physio` | input, optional | physio-import's `physio` output: `recordings.csv` (subject, species) and NWB files (one raw-trace figure per recording) |
| `responses` | output | `stim-dose-response/` (below) |

At least one `series.csv` is needed. Files are found by name and content, recursively (also when all
ports are merged side by side). A recording with trains but no series (e.g. no ECG), or with series
but no trains (a resting recording), is listed as skipped in `provenance.json`; the step still exits 0.

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `SPECIES` | `auto` | Species preset for the windows; `auto` = from `recordings.csv` or the upstream steps' `provenance.json`, else `human` |
| `BASELINE_S` | `auto` | Baseline window before each train (s); `auto` = species preset |
| `AFTER_S` | `auto` | Window after each train (s); `auto` = species preset |
| `RESPONSE_WINDOW` | `train` | `train` (mean over the train), `peak` (mean over `PEAK_WINDOW_S` around the largest change), or `fixed:<a>,<b>` (seconds after onset) |
| `PEAK_WINDOW_S` | `2` | Moving-mean width for the peak, time to peak and recovery |
| `RECOVERY_FRACTION` | `0.1` | Recovered = back within this fraction of the peak change (0.1 = 90 % recovered) |
| `MIN_VALID` | `0.8` | Trains whose baseline window or response window (each checked on its own) has a smaller share of valid samples are left out of the fits |
| `OVERLAP` | `exclude` | `exclude`: a train whose baseline or response window overlaps another train is left out of the fits; `clip`: the baseline starts where the previous train ended, and the train is left out only when less than half of `BASELINE_S` remains |
| `THRESHOLD_CRITERION` | `auto` | `auto` = the larger of 5 % of baseline (a convention of this app, no published source) and 3 SD of the train-to-train baseline variability (the SD of the included trains' baseline means; with fewer than 3 included trains, the pooled within-window SD of the baseline samples; it pools all of a subject's trains across frequencies and other parameters, and includes slow drift and carry-over from the previous train); or `pct:<n>`, `sd:<k>`, `abs:<value>` |
| `MIN_LEVELS` | `4` | Fewest distinct levels for a curve fit (fewer: level means and threshold only) |
| `MODELS` | `hill,boltzmann,linear,none` | Models compared by AICc (`none` always included) |
| `N_BOOT` | `500` | Bootstrap resamples for the fulcrum interval (trains resampled within levels; only when levels were repeated); `0` = none. The other intervals are t-based |
| `SEED` | `42` | Bootstrap seed; each fit derives its own seed from it and its identity, so linking another variable does not change other intervals |
| `VARIABLES` | `all` | Variables to analyse (comma list) |
| `FIGURE_VARIABLES` | `auto` | Variables with figures: `auto` = `hr_bpm` (else `hr_device_bpm`), `map_mmhg`, `resp_rate_bpm` when present; `all`; or a list |
| `FIGURES` | `true` | Write figures |
| `THREADS` | `0` | BLAS threads; 0 = from the container's CPU quota |

### Species presets

Only the windows depend on the species. No publication prescribes them; they follow a rule of this
app: the window must hold at least two breaths and about ten beats at resting rates, so the
respiratory oscillation of heart rate and pressure largely averages out. Each value can be
overridden.

| Species | `BASELINE_S` | `AFTER_S` | Source |
|---|---|---|---|
| human, dog | 10 | 10 | design rule above |
| pig | 10 | 10 | design rule; equals the 10 s recorded before and after each train in Discover 349 (Armitage et al.) |
| rabbit, rat, mouse | 5 | 5 | design rule above (fast breathing and heart rates) |

## Outputs (`stim-dose-response/`)

| File | Columns |
|---|---|
| `responses.csv` | `recording_id, event_id, variable, baseline, during, after, delta, pct_change, peak_delta, time_to_peak_s, recovery_s, valid_fraction` (the smaller of the two windows' valid shares), then `subject, unit, series_source, onset, duration, trial_type, included, exclusion_reason, window_start_s, window_end_s, baseline_sd, valid_fraction_baseline, valid_fraction_response, after_s_used` (the after window stops where the next train starts), then the train's parameters (`amplitude, amplitude_unit, pulse_width_us, frequency_hz, train_duration_s, ..., condition` and any extra parameter column of events.tsv) |
| `fits.csv` | conventions.md order first: `subject, recording_id` (`pooled` for pooled rows; `|`-joined when a subject has several recordings), `variable, x, group` (the other parameters held fixed, only those that vary), `model` (`hill`/`boltzmann`/`linear`/`none`), `threshold, threshold_criterion, ec50, top, bottom, slope, r2, n` (trains; for pooled rows the number of level estimates), `aic` (AICc); then `x_unit, threshold_bound` (`bracketed`, or `upper`: the next lower tested level was not analysed or not tested, so the true threshold may be lower), `unit, trial_type, n_levels, n_subjects, criterion_value, direction, ec50_ci_low, ec50_ci_high, top_ci_low, top_ci_high, slope_ci_low, slope_ci_high, aic_hill, aic_boltzmann, aic_linear, aic_none, levels, level_means, species, note`. A group in which fewer than two levels have included trains still gets a row (`none`, with the reason). When the response changes sign along x (fulcrum.csv), the monotone fit's `ec50, top, bottom, slope` and intervals are left empty |
| `surface.csv` | `subject, variable, amplitude, frequency_hz, mean_delta, sd, n, group, trial_type, unit, species` (when amplitude and frequency both vary; pooled rows average the subjects' cell means) |
| `fulcrum.csv` | `subject, variable, frequency_hz, amplitude_at_fulcrum, method, ci_low, ci_high, group, recording_id, levels, level_means, species, note` |
| `features.csv` | `recording_id, channel, feature, value`: `dose_n_trains`, `dose_n_trains_included`, `resp_max_delta_<variable>`, `dose_threshold_<variable>_<x>[_<group>]`, `dose_ec50_...`, `fulcrum_amplitude_<variable>[_<group>]` |
| `figures/` | `<subject>_<variable>_timecourse.png`, `_dose_response.png`, `_surface.png`; `pooled-<species>_<variable>_...`; `<recording_id>_raw.png` (with the `physio` port) |
| `summary.json` | headline numbers per recording and per subject, captioned figures and tables, warnings, and a `notice` (research outputs; the fulcrum caveat when a fulcrum is reported) |
| `provenance.json` | the shared keys of the physiology apps, plus `windows` (per recording: species, windows and where they came from) and `species_presets` |

Model details: Hill `y = top·xⁿ/(ec50ⁿ + xⁿ)` and Boltzmann `y = top/(1 + e^((ec50 − x)/slope))` with
the lower asymptote (bottom) fixed at 0, the baseline. The Hill curve passes through 0 at x = 0; the
Boltzmann curve only approaches 0 and is `top/(1 + e^(ec50/slope))` at x = 0. Linear `y = a + b·x`;
none `y = c`. Models are compared by AICc = n·ln(RSS/n) + 2K + 2K(K + 1)/(n − K − 1), K = fitted
parameters + 1 (Hurvich & Tsai 1989), and a model takes part only when n ≥ K + 2: at least 4 points
for `none`, 5 for `linear` and 6 for a sigmoid. **With one train per level and the default
MIN_LEVELS = 4, a sigmoid needs six levels; with four or five levels the choice is only between
linear and none.** Repeating trains per level lets the sigmoids take part. Intervals: t quantiles on
the residual degrees of freedom from the fit's covariance (Hill EC50 on the log scale), none below 2
residual degrees of freedom; pooled, normal quantiles from the mixed model's standard errors. `top` is the fitted maximum change (for `linear`,
the fitted value at the highest level; `bottom` at the lowest). An EC50 at the fit bounds is left
empty; one outside the tested range is kept and flagged as extrapolated in `note`.

**Neural fulcrum caveat.** Ardell et al. (2017) defined the neural fulcrum in dogs with chronically
implanted cervical vagus electrodes, studied awake, as the frequency-amplitude-pulse-width operating
point where a null heart-rate response is *reproducibly* evoked during the on-phase of stimulation,
between afferent-driven tachycardia at low intensities and efferent-driven bradycardia at higher
ones. Their abstract states that VNS heart control differs qualitatively between awake and
anaesthetised states but that the physiological expression of the neural fulcrum is maintained. Here
it is only the amplitude where the heart-rate response to a recording's trains changes sign at a
fixed frequency, interpolated between two tested levels. With one train per level the null response
cannot be shown to be reproducible (fulcrum.csv then has no interval): treat it as a descriptive
crossing point, not the operating point of chronic therapy. `summary.json` carries this caveat as its `notice` whenever a
fulcrum is reported.

## Runtime

`cpu: 512`, `memory: 1024`, `timeoutSeconds: 900`, `computeTypes: [standard]` (Lambda-eligible).
Measured in Docker, **amd64 emulation on an M-series Mac**, run as Lambda runs it (`--read-only
--user 65534:65534 --tmpfs /tmp`, `--cpus 1 --memory 2g`):

| Input | Trains x variables | Peak memory (cgroup) | Wall time |
|---|---|---|---|
| Discover 349, sub-2 (ecg-hrv series: `hr_bpm`, `rr_ms`, `hr_device_bpm`) | 20 x 3 | 213 MB | 3.2 s |
| Discover 655, 3 recordings (`hr_bpm`, `rr_ms`) | 16 x 2 | 239 MB | 3.0 s |
| Synthetic fixture (6 recordings, 132 fits) | 79 x 1-4 | 232 MB | 8.2 s |

Time grows with the number of fits (and `N_BOOT` for fulcrums with repeated levels); memory with the length of the series (all series of a
run are held in memory: about 1 MB per hour of recording per variable at 4 Hz). Image: 616 MB
unpacked (147 MB compressed).

## Validation

**Synthetic fixture** (`tests/make_fixture.py`, asserted by `tests/test_processor.py`): pig
amplitude x frequency grids with a planted Hill bradycardia (every level twice), a pig whose response
changes sign, a rat sweep (every level twice, plus one train that starts 2 s before the series ends),
and a resting human recording.

| Planted | Recovered |
|---|---|
| EC50 600 / 500 / 700 µA at 20 Hz (pigs p1, p3, p4) | 612.3 [602.0-622.7] / 500.4 [494.8-506.1] / 700.5 [683.3-718.3] µA, Hill, r² > 0.999 |
| EC50 at 5 Hz (same pigs) | 596.3 / 482.8 / 736.7 µA (intervals contain the planted values) |
| Maximum change at 20 Hz: -30 bpm x 0.801 (mean over a 5 s train with a 1 s rise) = -24.0 bpm | -23.5 / -23.3 / -23.4 bpm |
| Thresholds (5 % of 80 bpm): 400 µA at 20 Hz, 800 µA at 5 Hz | exactly these levels, all three pigs, `bracketed` |
| Rat: EC50 300 µA, maximum -100 x 0.801 bpm, threshold 200 µA (5 % of 380 bpm) | 300.7 [297.8-303.6] µA, -79.1 bpm, 200 µA; 5 s windows from the rat preset |
| Rat edge train (series ends 2 s into it) | response window 40 % valid: left out |
| Pooled (3 pigs): mean EC50 600 µA | MixedLM level estimates, weighted Hill fit: EC50 598.6 µA [533.6-671.4], threshold 400 µA, n = 6 levels |
| Mean arterial pressure falls linearly with amplitude | linear chosen in all 6 subject x frequency fits and both pooled fits |
| Fulcrum (pig p2, 10 Hz) at 1063.8 µA | 1042.7 µA [1038.6-1046.8] (-2.0 %: linear interpolation between 800 and 1600 µA is biased, and the interval covers only the train-to-train noise); the group's linear fit is blanked |
| Resting recording, physio-import's markers file | skipped, not read |

**Discover 349** (pig sub-2, ECG-derived `hr_bpm` from ecg-hrv; 20 trains, 50-2500 µA x 2-30 Hz, 5 s):
with `OVERLAP=exclude`, 14 of 20 trains are used (6 start less than 10 s after the previous train, as
stim-events' QC also lists). Baseline heart rate about 70 bpm; the threshold criterion is 5 % of
baseline (3.5 bpm; the train-to-train baseline SD is 0.6 bpm). Responses are small and mostly
tachycardic at 500-1500 µA (+4.1 to +5.3 bpm at 10-30 Hz), turning to bradycardia at 2500 µA and 20-30 Hz
(-1.9 and -5.3 bpm). Thresholds: 500 µA at 30 Hz (bracketed); 1000 µA at 10 Hz and 1500 µA at 20 Hz,
both **upper bounds** because the trains at the levels below were left out (overlap); none at 2 Hz.
With one train per level and at most five levels, AICc chooses `none` at every frequency: these data
do not support a dose-response curve. `OVERLAP=clip` keeps 19 of 20 trains: thresholds become
500 µA (bracketed) at 10, 20 and 30 Hz, and the models are still `none`. At 30 Hz the response
changes sign between 1500 and 2500 µA: crossing at **1751 µA** from the ECG-derived heart rate and
**1714 µA** from the device heart-rate channel (`hr_device_bpm`), which agree; no interval (one train
per level, so the null response is not shown to be reproducible). This is a crossing point in an
anaesthetised acute preparation, not the fulcrum of Ardell et al.

**Discover 655** (rat, closed-loop heart-rate control; `hr_bpm` from ecg-hrv): the controller's trains
on two electrodes overlap almost everywhere, so only 4 of 16 trains are free of overlap. The
open-loop perturbation trains (3.78 a.u. on the proximal electrode) lowered heart rate by 54.1 bpm
(44 s train, gain 5 session, -17 %; largest change -67.6 bpm, 38.5 s after onset) and 40.4 bpm (61 s,
gain 10, -13 %; -51.8 bpm at 56.4 s) from about 310 bpm, and recovered within 7-8 s after the train;
0.1 s bursts changed nothing (+1.8 and +1.7 bpm). Kilohertz block epochs (kept in `responses.csv`,
all overlapping) raised heart rate by 10-19 bpm. The amplitude barely
varies, so there is no dose-response curve here: the dataset tests the overlap and block handling,
long trains and the rat preset.

## Limitations

- The response is a mean change over a window: it cannot separate a direct effect of the train from
  a slower drift, and with short gaps between trains the previous train's after-effect sits in the
  next baseline. The table flags overlap, not after-effects; use `MIN_GAP_S` in stim-events and the
  time-course figures to judge it.
- With one train per level (common in grid protocols) a curve rests on 4-6 points: AICc then allows
  only linear vs none (sigmoids from six levels), the t-based intervals are wide and conditional on
  the chosen model, and thresholds are only as fine as the tested levels. Repeat trains per level
  where possible.
- The threshold is the lowest level whose mean change exceeds the criterion; it is not a statistical
  test and does not correct for testing several levels. When trains of the level below were left out
  (overlap, artefact), it is only an upper bound (`threshold_bound=upper`).
- The 5 % part of the criterion is a convention without a published source; the 3 SD part pools all
  of a subject's trains (every frequency) and includes slow drift and carry-over between trains.
- Anaesthesia changes autonomic tone and responses to vagus stimulation; values from anaesthetised
  preparations (Discover 349) do not transfer to conscious animals or patients.
- Validated on synthetic data with known answers, on one pig (Discover 349) and one rat (Discover
  655). Not validated on human, dog, rabbit or mouse recordings; the windows for those species are
  presets, not measured choices.
- Uncalibrated pressures arrive from bp-resp without absolute values; their changes are relative.
- The neural fulcrum here is a crossing point, with the caveat above.

## Licences

Apache-2.0 (this app). Bundled packages and their licences: `THIRD_PARTY.md` (all BSD, MIT, PSF-style
or Apache). No licence notice applies to the outputs. The demo datasets are CC BY 4.0; cite them when
you use their results.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in CITATION.cff.

Methods:

- Ardell JL, Nier H, Hammer M, Southerland EM, Ardell CL, Beaumont E, KenKnight BH, Armour JA (2017).
  Defining the neural fulcrum for chronic vagus nerve stimulation: implications for integrated cardiac
  control. *The Journal of Physiology* 595(22):6887-6903. doi:10.1113/JP274678. PMID 28862330.
- Akaike H (1974). A new look at the statistical model identification. *IEEE Transactions on
  Automatic Control* 19(6):716-723. doi:10.1109/TAC.1974.1100705.
- Hurvich CM, Tsai CL (1989). Regression and time series model selection in small samples.
  *Biometrika* 76(2):297-307. doi:10.1093/biomet/76.2.297.
- Efron B (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*
  7(1):1-26. doi:10.1214/aos/1176344552.

Software:

- Virtanen P, Gommers R, Oliphant TE, et al. (2020). SciPy 1.0: fundamental algorithms for scientific
  computing in Python. *Nature Methods* 17(3):261-272. doi:10.1038/s41592-019-0686-2. PMID 32015543.
- Seabold S, Perktold J (2010). Statsmodels: Econometric and Statistical Modeling with Python.
  *Proceedings of the Python in Science Conference* 92-96. doi:10.25080/Majora-92bf1922-011.
- Harris CR, Millman KJ, van der Walt SJ, et al. (2020). Array programming with NumPy. *Nature*
  585(7825):357-362. doi:10.1038/s41586-020-2649-2. PMID 32939066.
- McKinney W (2010). Data Structures for Statistical Computing in Python. *Proceedings of the Python
  in Science Conference* 56-61. doi:10.25080/Majora-92bf1922-00a.
- Hunter JD (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*
  9(3):90-95. doi:10.1109/MCSE.2007.55.
- Rübel O, Tritt A, Ly R, et al. (2022). The Neurodata Without Borders ecosystem for
  neurophysiological data science. *eLife* 11:e78362. doi:10.7554/eLife.78362. PMID 36193886.
- h5py, Andrew Collette and contributors, https://github.com/h5py/h5py; patsy, Patsy Developers,
  https://github.com/pydata/patsy.

Background (vagus nerve stimulation dose-response with physiological markers):

- Chang YC, Cracchiolo M, Ahmed U, et al. (2020). Quantitative estimation of nerve fiber engagement by
  vagus nerve stimulation using physiological markers. *Brain Stimulation* 13(6):1617-1630.
  doi:10.1016/j.brs.2020.09.002. PMID 32956868.
- Huffman WJ, Musselman ED, Pelot NA, Grill WM (2023). Measuring and modeling the effects of vagus
  nerve stimulation on heart rate and laryngeal muscles. *Bioelectronic Medicine* 9:3.
  doi:10.1186/s42234-023-00107-4. PMID 36797733.

Data:

- Armitage O, Sarkans E, Wernisch L, Fortier-Poisson P, Berthon A, Tessier-Larivière O (2023).
  Influence of vagus nerve stimulation on cardiac activity in pigs. Pennsieve Discover / SPARC
  Portal. doi:10.26275/dm5l-tspb.
- Bender SA, Green DB, Thakkar VS, Zimmerman HL, Elazab M, Kilgore KL, Vrabec TL (2026).
  Single-electrode bidirectional control of heart rate via vagus nerve modulation in rat model.
  Pennsieve Discover / SPARC Portal. doi:10.26275/5cwo-mkpz.

## Acknowledgements

The SciPy, NumPy, pandas, statsmodels, Matplotlib and h5py projects and their contributors. The demo
datasets were shared on Pennsieve Discover by their authors (listed above) under CC BY 4.0.

## Development

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    make test PYTHON=.venv/bin/python        # pytest on the synthetic fixture (no Docker)
    make run                                  # Docker over data/fixture into data/output
    make lambda-check                         # read-only, any uid, HOME/TMPDIR under /tmp
    make run-demo DEMO=../workflow-physiology/data/work/d349
