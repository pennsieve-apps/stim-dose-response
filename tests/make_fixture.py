"""
Synthetic inputs for stim-dose-response, laid out as the compute node lays out INPUT_DIR:

  events/stim-events/events.tsv     stim-events' trains (+ provenance.json)
  hrv/ecg-hrv/series.csv            hr_bpm, rr_ms at the species' SERIES_HZ (+ provenance.json)
  bp/bp-resp/series.csv             map_mmhg, sbp_mmhg (only for the pig grid subjects)
  physio/recordings.csv, channels.csv, events.tsv (physio-import's raw markers: must be ignored)
  physio/sub-p1/sub-p1_rec-grid.nwb  ECG and BP waveforms of one recording (raw-trace figure)
  truth.json                         the planted thresholds, EC50s and the fulcrum

Recordings (all responses are planted, with known parameters):
  sub-p1, sub-p3, sub-p4 (pig, 4 Hz): amplitude 100-3200 uA x frequency 5 and 20 Hz, 5 s trains. Heart
      rate falls by a Hill function of amplitude (top -30 bpm at 20 Hz, -10 bpm at 5 Hz, EC50 600 / 500 /
      700 uA, n = 2); mean arterial pressure falls linearly with amplitude. Three subjects -> pooled fits.
  sub-p2 (pig, 4 Hz, no BP): 10 Hz, amplitudes 100-3200 uA twice each; heart rate rises at low amplitude
      and falls at high amplitude (a planted neural fulcrum near 1063 uA).
  sub-r1 (rat, 10 Hz, no BP): amplitude 50-1600 uA at 20 Hz; heart rate 380 bpm falls by up to 100 bpm
      (EC50 300 uA); tests the rat preset (5 s windows).
  sub-h1 (human, 4 Hz): resting, no stimulation -> skipped.

The response to each train rises with a 1 s time constant during the train and decays with a 3 s time
constant after it, so the mean over a 5 s train is 0.8013 x the plateau; thresholds in truth.json are
computed from those noise-free train means and the auto criterion (5 % of baseline).

Usage: python -m tests.make_fixture <out_dir>
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

TRAIN_S = 5.0
RISE_TAU, DECAY_TAU = 1.0, 3.0
TRAIN_MEAN_FACTOR = 1 - RISE_TAU * (1 - math.exp(-TRAIN_S / RISE_TAU)) / TRAIN_S

EVENT_COLS = ["recording_id", "onset", "duration", "trial_type", "event_id", "amplitude", "amplitude_unit",
              "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s", "train_off_s", "n_pulses",
              "polarity", "waveform", "site", "contact", "condition", "label", "source", "trial_group"]


def hill(a, top, ec50, n):
    a = np.asarray(a, float)
    return top * a ** n / (ec50 ** n + a ** n)


def fulcrum_curve(a):
    a = np.asarray(a, float)
    return 12 * a / (a + 150) - 40 * a ** 3 / (1500 ** 3 + a ** 3)


def fulcrum_true() -> float:
    lo, hi = 800.0, 1600.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if fulcrum_curve(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def response_shape(t: np.ndarray, onset: float) -> np.ndarray:
    """0..1 profile of one train's effect: exponential rise during the train, decay after it."""
    r = np.zeros_like(t)
    on = (t >= onset) & (t < onset + TRAIN_S)
    r[on] = 1 - np.exp(-(t[on] - onset) / RISE_TAU)
    end_level = 1 - math.exp(-TRAIN_S / RISE_TAU)
    after = t >= onset + TRAIN_S
    r[after] = end_level * np.exp(-(t[after] - onset - TRAIN_S) / DECAY_TAU)
    return r


def schedule(levels: list[tuple], start=20.0, spacing=35.0, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(levels))
    return [(start + i * spacing, levels[j]) for i, j in enumerate(order)]


def threshold_of(levels, means, crit):
    for lv, m in sorted(zip(levels, means)):
        if abs(m) >= crit:
            return lv
    return None


def main(out: Path) -> dict:
    out = Path(out)
    rng = np.random.default_rng(7)
    events, hrv_rows, bp_rows, recordings, truth = [], [], [], [], {"subjects": {}}

    def add_series(rows, rid, t, var, v, valid=None):
        valid = np.ones(len(t), bool) if valid is None else valid
        for ti, vi, ok in zip(t, v, valid):
            rows.append([rid, f"{ti:.3f}", var, f"{vi:.4f}", "true" if ok else "false"])

    def add_events(rid, sched, freq_of, amp_of, extra=None):
        for k, (onset, lv) in enumerate(sched, 1):
            f = freq_of(lv)
            events.append({"recording_id": rid, "onset": f"{onset:.3f}", "duration": f"{TRAIN_S:g}",
                           "trial_type": "stimulation", "event_id": f"stim-{k:04d}", "amplitude": f"{amp_of(lv):g}",
                           "amplitude_unit": "uA", "pulse_width_us": "500", "frequency_hz": f"{f:g}",
                           "train_duration_s": f"{TRAIN_S:g}", "train_on_s": "n/a", "train_off_s": "n/a",
                           "n_pulses": f"{int(f * TRAIN_S)}", "polarity": "cathodic", "waveform": "biphasic",
                           "site": "cervical vagus left", "contact": "n/a", "condition": "n/a", "label": "n/a",
                           "source": "trigger", "trial_group": f"trial-{rid}-{k}"})

    # ---- pig grid subjects
    amps = [100, 200, 400, 800, 1600, 3200]
    freqs = [5, 20]
    tops = {5: -10.0, 20: -30.0}
    for subj, ec50, seed in (("p1", 600.0, 1), ("p3", 500.0, 2), ("p4", 700.0, 3)):
        rid = f"sub-{subj}_rec-grid"
        sched = schedule([(a, f) for f in freqs for a in amps], seed=seed)
        dur = sched[-1][0] + 40
        t = np.arange(0, dur, 0.25)
        hr = 80 + 0.5 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.4, len(t))
        mp = 90 + rng.normal(0, 0.4, len(t))
        for onset, (a, f) in sched:
            prof = response_shape(t, onset)
            hr += float(hill(a, tops[f], ec50, 2.0)) * prof
            mp += (-0.008 if f == 20 else -0.003) * a * prof
        valid = np.ones(len(t), bool)
        valid[(t > 3) & (t < 5)] = False          # an artefact far from any train
        add_series(hrv_rows, rid, t, "hr_bpm", hr, valid)
        add_series(hrv_rows, rid, t, "rr_ms", 60000.0 / hr, valid)
        add_series(bp_rows, rid, t, "map_mmhg", mp)
        add_series(bp_rows, rid, t, "sbp_mmhg", mp + 25)
        add_events(rid, sched, lambda lv: lv[1], lambda lv: lv[0])
        recordings.append({"recording_id": rid, "subject": subj, "species": "pig", "duration_s": dur,
                           "path": f"sub-{subj}/{rid}.nwb" if subj == "p1" else ""})
        thr = {}
        for f in freqs:
            means = [float(hill(a, tops[f], ec50, 2.0)) * TRAIN_MEAN_FACTOR for a in amps]
            thr[str(f)] = threshold_of(amps, means, 0.05 * 80)
        truth["subjects"][subj] = {"species": "pig", "ec50_hr_uA": ec50, "threshold_hr_uA": thr,
                                   "map_model": "linear"}
        if subj == "p1":
            write_nwb(out / "physio" / f"sub-{subj}" / f"{rid}.nwb", rid, t, hr, mp)

    # ---- pig with a fulcrum (no BP)
    rid = "sub-p2_rec-fulcrum"
    sched = schedule([(a, 10) for a in amps for _ in range(2)], seed=4)
    dur = sched[-1][0] + 40
    t = np.arange(0, dur, 0.25)
    hr = 80 + 0.5 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.4, len(t))
    for onset, (a, _f) in sched:
        hr += float(fulcrum_curve(a)) * response_shape(t, onset)
    add_series(hrv_rows, rid, t, "hr_bpm", hr)
    add_series(hrv_rows, rid, t, "rr_ms", 60000.0 / hr)
    add_events(rid, sched, lambda lv: lv[1], lambda lv: lv[0])
    recordings.append({"recording_id": rid, "subject": "p2", "species": "pig", "duration_s": dur, "path": ""})
    truth["subjects"]["p2"] = {"species": "pig", "fulcrum_hr_uA": fulcrum_true(), "fulcrum_bracket": [800, 1600]}

    # ---- rat (10 Hz series, 5 s windows, no BP)
    rid = "sub-r1_rec-sweep"
    ramps = [50, 100, 200, 400, 800, 1600]
    sched = schedule([(a, 20) for a in ramps], start=15, spacing=25, seed=5)
    dur = sched[-1][0] + 30
    t = np.arange(0, dur, 0.1)
    hr = 380 + 1.0 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 1.0, len(t))
    for onset, (a, _f) in sched:
        hr += float(hill(a, -100.0, 300.0, 2.0)) * response_shape(t, onset)
    add_series(hrv_rows, rid, t, "hr_bpm", hr)
    add_series(hrv_rows, rid, t, "rr_ms", 60000.0 / hr)
    add_events(rid, sched, lambda lv: lv[1], lambda lv: lv[0])
    recordings.append({"recording_id": rid, "subject": "r1", "species": "rat", "duration_s": dur, "path": ""})
    means = [float(hill(a, -100.0, 300.0, 2.0)) * TRAIN_MEAN_FACTOR for a in ramps]
    truth["subjects"]["r1"] = {"species": "rat", "ec50_hr_uA": 300.0,
                               "threshold_hr_uA": {"20": threshold_of(ramps, means, 0.05 * 380)}}

    # ---- human resting recording: series, no stimulation
    rid = "sub-h1_rec-rest"
    t = np.arange(0, 300, 0.25)
    add_series(hrv_rows, rid, t, "hr_bpm", 65 + rng.normal(0, 1, len(t)))
    recordings.append({"recording_id": rid, "subject": "h1", "species": "human", "duration_s": 300, "path": ""})
    truth["skipped"] = [rid]
    truth["train_mean_factor"] = TRAIN_MEAN_FACTOR

    # ---- write
    ev_dir = out / "events" / "stim-events"
    ev_dir.mkdir(parents=True, exist_ok=True)
    with open(ev_dir / "events.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, EVENT_COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for e in sorted(events, key=lambda e: (e["recording_id"], float(e["onset"]))):
            w.writerow(e)
    (ev_dir / "provenance.json").write_text(json.dumps({"app": "stim-events", "version": "fixture"}))
    hdr = ["recording_id", "time_s", "variable", "value", "valid"]
    species = {r["recording_id"]: r["species"] for r in recordings}
    for port, app, rows in (("hrv", "ecg-hrv", hrv_rows), ("bp", "bp-resp", bp_rows)):
        d = out / port / app
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "series.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(hdr)
            w.writerows(rows)
        rids = sorted({r[0] for r in rows})
        (d / "provenance.json").write_text(json.dumps(
            {"app": app, "version": "fixture", "recordings": {r: {"status": "ok", "species": species[r]} for r in rids}}))
    pdir = out / "physio"
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / "recordings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "subject", "session", "label", "path", "species", "duration_s"])
        for r in recordings:
            w.writerow([r["recording_id"], r["subject"], "", r["recording_id"].rsplit("-", 1)[-1], r["path"],
                        r["species"], r["duration_s"]])
    with open(pdir / "channels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "channel", "type", "original_name", "unit", "sampling_rate_hz"])
        w.writerow(["sub-p1_rec-grid", "ecg", "ecg", "ECG", "V", 250])
        w.writerow(["sub-p1_rec-grid", "bp", "bp", "ABP", "mmHg", 250])
    # physio-import's markers file: same name, raw markers; stim-dose-response must not read it
    with open(pdir / "events.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["recording_id", "onset", "duration", "trial_type", "event_id", "label", "source"])
        w.writerow(["sub-p1_rec-grid", "1.0", "n/a", "stimulation", "mk-0001", "decoy stim marker", "marker"])
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def write_nwb(path: Path, rid: str, t4: np.ndarray, hr: np.ndarray, mp: np.ndarray, rate: float = 250.0) -> None:
    """A minimal NWB 2.x layout (h5py): acquisition/ecg and acquisition/bp with starting_time + rate."""
    import h5py  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(0, t4[-1], 1 / rate)
    hr_t = np.interp(t, t4, hr)
    phase = np.cumsum(hr_t / 60.0 / rate)
    frac = phase % 1.0
    ecg = 1e-3 * np.exp(-((frac - 0.3) / 0.012) ** 2) - 1e-4 * np.exp(-((frac - 0.6) / 0.06) ** 2)
    bp = np.interp(t, t4, mp) + 20 * np.exp(-((frac - 0.4) / 0.08) ** 2) - 7
    with h5py.File(path, "w") as f:
        f.attrs["namespace"] = "core"
        f.attrs["neurodata_type"] = "NWBFile"
        f.create_dataset("identifier", data=rid)
        acq = f.create_group("acquisition")
        for name, y, unit in (("ecg", ecg, "V"), ("bp", bp, "mmHg")):
            g = acq.create_group(name)
            g.attrs["neurodata_type"] = "TimeSeries"
            d = g.create_dataset("data", data=y.astype("float32"), compression="gzip")
            d.attrs["unit"] = unit
            d.attrs["conversion"] = 1.0
            d.attrs["offset"] = 0.0
            st = g.create_dataset("starting_time", data=0.0)
            st.attrs["rate"] = rate
            st.attrs["unit"] = "seconds"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(Path(sys.argv[1]))
    print(f"fixture written to {sys.argv[1]}")
