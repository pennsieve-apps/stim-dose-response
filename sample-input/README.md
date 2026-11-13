# Sample input

stim-dose-response reads stim-events' `events` output (port `events`), at least one `series.csv`
(ecg-hrv's `hrv` port, bp-resp's `bp` port, or any step that writes the same shape) and, optionally,
physio-import's `physio` output (subject, species, raw traces).

## Synthetic (no download)

    python -m tests.make_fixture data/fixture   # data/fixture/{events,hrv,bp,physio}, truth.json

Five recordings with planted responses (pig amplitude x frequency grids, a pig with a sign-changing
heart-rate response, a rat sweep) and a resting human recording; `truth.json` has the planted
thresholds, EC50s and fulcrum.

## Real recordings used to test this app

Run physio-import, stim-events and ecg-hrv (and bp-resp when the recording has blood pressure) on
these first; their output folders are this app's inputs.

- **Pennsieve Discover 349** (doi:10.26275/dm5l-tspb, CC BY 4.0; Armitage et al.): pig vagus nerve
  stimulation over a current x frequency grid (50-2500 uA x 2-30 Hz, 5 s trains),
  `files/primary/sub-2/subject-2.hdf5` (496 MB). ECG and a device heart-rate channel; no blood pressure.
- **Pennsieve Discover 655** (doi:10.26275/5cwo-mkpz, CC BY 4.0; Bender et al.): rat closed-loop
  heart-rate control by vagus stimulation and kilohertz block, the CSV exports
  `files/primary/sub-Bimodal01/perf-*/*.smrx.csv`. ECG only.

In the workflow repo, `scripts/fetch-sample-data.sh` stages both under `data/`; the upstream outputs
are under `data/work/<demo>/<app>/`, and `make run-demo DEMO=../workflow-physiology/data/work/d349`
mounts them.
