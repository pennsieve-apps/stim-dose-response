"""
stim-dose-response: responses to each stimulation and dose-response fits, for every physiological series.

  INPUT_DIR   events/  stim-events' output (stim-events/events.tsv, one row per train)
              hrv/, bp/  series.csv of ecg-hrv and bp-resp (or of any step that writes that shape)
              physio/  optional: physio-import's recordings.csv (subject, species) and NWB files
              (all found by name and content, recursively, also when merged side by side)
  OUTPUT_DIR/stim-dose-response/
      responses.csv  fits.csv  surface.csv  fulcrum.csv  features.csv  figures/*.png
      summary.json  provenance.json

One recording at a time for the responses; fits per subject (and pooled over >= 3 subjects).
A recording without stimulation events or without any series is listed as skipped (not a failure).
Exit codes: 0 ok, 1 no usable input / every recording failed, 2 bad parameters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_environment() -> None:
    """Lambda-safe start: HOME and TMPDIR may be fresh, empty folders and everything else read-only."""
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    for var, default in (("TMPDIR", tmp), ("HOME", tmp / "home")):
        p = Path(os.environ.get(var) or default)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        os.environ[var] = str(p)
    mpl = Path(os.environ["TMPDIR"]) / "mpl"
    baked = Path(os.environ.get("STIM_DOSE_RESPONSE_MPL_CACHE", "/opt/stim-dose-response/matplotlib"))
    if not os.environ.get("MPLCONFIGDIR") or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
        mpl.mkdir(parents=True, exist_ok=True)
        if baked.is_dir():
            for f in baked.glob("fontlist-*.json"):
                target = mpl / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
        os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("MPLBACKEND", "Agg")


_prepare_environment()

from processor import resources as R  # noqa: E402

_requested = os.environ.get("THREADS", "0").strip() or "0"
try:
    _THREADS, _THREADS_WHY = R.resolve_workers(int(_requested), per_worker_mb=300, reserve_mb=400)
except ValueError:
    _THREADS, _THREADS_WHY = 1, f"THREADS={_requested!r} is not an integer"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import numpy as np  # noqa: E402

from processor import doseresp as DR  # noqa: E402
from processor import inputs as I  # noqa: E402
from processor import presets as PS  # noqa: E402
from processor import responses as RS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("stim-dose-response")
log.setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

APP_NAME = "stim-dose-response"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE, FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
MIN_SUBJECTS_POOLED = 3

DOI_SCIPY = "10.1038/s41592-019-0686-2"
DOI_STATSMODELS = "10.25080/Majora-92bf1922-011"
DOI_FULCRUM = "10.1113/JP274678"
DOI_AIC = "10.1109/TAC.1974.1100705"
DOI_BOOTSTRAP = "10.1214/aos/1176344552"
DOI_AICC = "10.1093/biomet/76.2.297"

FULCRUM_CAVEAT = (
    "The neural fulcrum was defined by Ardell et al. (2017) in dogs with chronically implanted cervical vagus "
    "stimulation electrodes, awake, as the frequency-amplitude-pulse-width operating point where a null heart-rate "
    "response is reproducibly evoked during the on-phase of stimulation, between afferent-driven tachycardia at low "
    "intensities and efferent-driven bradycardia at higher ones; they report that VNS heart control differs "
    "qualitatively between awake and anaesthetised states but that the neural fulcrum is maintained. Here it is "
    "only the amplitude where the heart-rate response to the trains of a recording changes sign at a fixed "
    "frequency, interpolated between two tested levels. With one train per level the null response cannot be "
    "shown to be reproducible (fulcrum.csv then has no interval), so treat it as a descriptive crossing point, "
    "not as the operating point of chronic therapy.")
RESEARCH_NOTICE = ("Research outputs: responses, thresholds and EC50 values describe these recordings and are "
                   "not clinical findings or dosing recommendations.")

DEFAULTS = {
    "SPECIES": "auto", "BASELINE_S": "auto", "AFTER_S": "auto", "RESPONSE_WINDOW": "train", "PEAK_WINDOW_S": "2",
    "RECOVERY_FRACTION": "0.1", "MIN_VALID": "0.8", "OVERLAP": "exclude", "THRESHOLD_CRITERION": "auto",
    "MIN_LEVELS": "4", "MODELS": "hill,boltzmann,linear,none", "N_BOOT": "500", "SEED": "42",
    "VARIABLES": "all", "FIGURE_VARIABLES": "auto", "FIGURES": "true", "THREADS": "0",
}

EVENT_PARAM_ORDER = ["amplitude", "amplitude_unit", "pulse_width_us", "frequency_hz", "train_duration_s",
                     "train_on_s", "train_off_s", "n_pulses", "polarity", "waveform", "site", "contact", "condition"]
RESP_HEADER = ["recording_id", "event_id", "variable", "baseline", "during", "after", "delta", "pct_change",
               "peak_delta", "time_to_peak_s", "recovery_s", "valid_fraction", "subject", "unit", "series_source",
               "onset", "duration", "trial_type", "included", "exclusion_reason", "window_start_s", "window_end_s",
               "baseline_sd", "valid_fraction_baseline", "valid_fraction_response", "after_s_used"]
FIT_HEADER = ["subject", "recording_id", "variable", "x", "group", "model", "threshold", "threshold_criterion",
              "ec50", "top", "bottom", "slope", "r2", "n", "aic",
              # beyond the conventions.md columns
              "x_unit", "threshold_bound", "unit", "trial_type", "n_levels", "n_subjects", "criterion_value",
              "direction", "ec50_ci_low", "ec50_ci_high", "top_ci_low", "top_ci_high", "slope_ci_low",
              "slope_ci_high", "aic_hill", "aic_boltzmann", "aic_linear", "aic_none", "levels", "level_means",
              "species", "note"]
SURF_HEADER = ["subject", "variable", "amplitude", "frequency_hz", "mean_delta", "sd", "n", "group", "trial_type",
               "unit", "species"]
FULC_HEADER = ["subject", "variable", "frequency_hz", "amplitude_at_fulcrum", "method", "ci_low", "ci_high", "group",
               "recording_id", "levels", "level_means", "species", "note"]
FEATURE_HEADER = ["recording_id", "channel", "feature", "value"]


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "pandas", "statsmodels", "matplotlib", "h5py"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


# --------------------------------------------------------------------------- parameters

def _bool(name, v):
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {v!r}")


def _num(name, v, lo=None, hi=None, lo_open=False):
    try:
        f = float(v)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {v!r}") from None
    if not math.isfinite(f) or (lo is not None and (f < lo or (lo_open and f == lo))) or (hi is not None and f > hi):
        raise ValueError(f"{name} out of range: {v!r}")
    return f


def _int(name, v, lo=None):
    try:
        i = int(str(v).strip())
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {v!r}") from None
    if lo is not None and i < lo:
        raise ValueError(f"{name} must be >= {lo}, got {v!r}")
    return i


def _auto_pos(name, v):
    s = str(v).strip().lower()
    return None if s in ("auto", "") else _num(name, s, 0, lo_open=True)


def _list(v) -> list[str]:
    return [s.strip() for s in str(v).split(",") if s.strip()]


def read_params(env=os.environ) -> tuple[dict, dict]:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    sp = raw["SPECIES"].strip().lower()
    if sp != "auto" and PS.normalise_species(sp) == "":
        raise ValueError(f"SPECIES must be auto or one of {', '.join(PS.SPECIES)}, got {raw['SPECIES']!r}")
    rw = raw["RESPONSE_WINDOW"].strip().lower().replace(" ", "")
    fixed = None
    if rw.startswith("fixed:"):
        try:
            a, b = (float(s) for s in rw.split(":", 1)[1].split(","))
        except ValueError:
            raise ValueError("RESPONSE_WINDOW fixed:<a>,<b> needs two numbers (seconds from onset)") from None
        if b <= a:
            raise ValueError("RESPONSE_WINDOW fixed:<a>,<b> needs b > a")
        fixed, rw = (a, b), "fixed"
    elif rw not in ("train", "peak"):
        raise ValueError(f"RESPONSE_WINDOW must be train, peak or fixed:<a>,<b>, got {raw['RESPONSE_WINDOW']!r}")
    crit = raw["THRESHOLD_CRITERION"].strip().lower().replace(" ", "")
    if crit != "auto":
        kind, _, arg = crit.partition(":")
        if kind not in ("pct", "sd", "abs"):
            raise ValueError("THRESHOLD_CRITERION must be auto, pct:<n>, sd:<k> or abs:<value>")
        _num("THRESHOLD_CRITERION", arg, 0, lo_open=True)
    models = [m.lower() for m in _list(raw["MODELS"])]
    if not models or any(m not in DR.ALL_MODELS for m in models):
        raise ValueError(f"MODELS must be a comma list of {', '.join(DR.ALL_MODELS)}")
    if "none" not in models:
        models.append("none")
    overlap = raw["OVERLAP"].strip().lower()
    if overlap not in ("exclude", "clip"):
        raise ValueError("OVERLAP must be exclude or clip")
    variables = raw["VARIABLES"].strip()
    figvars = raw["FIGURE_VARIABLES"].strip()
    p = {
        "species": sp,
        "baseline_s": _auto_pos("BASELINE_S", raw["BASELINE_S"]),
        "after_s": _auto_pos("AFTER_S", raw["AFTER_S"]),
        "response_window": rw, "fixed": fixed,
        "peak_window_s": _num("PEAK_WINDOW_S", raw["PEAK_WINDOW_S"], 0, lo_open=True),
        "recovery_fraction": _num("RECOVERY_FRACTION", raw["RECOVERY_FRACTION"], 0, 1, lo_open=True),
        "min_valid": _num("MIN_VALID", raw["MIN_VALID"], 0, 1),
        "overlap": overlap,
        "criterion": crit,
        "min_levels": _int("MIN_LEVELS", raw["MIN_LEVELS"], 2),
        "models": models,
        "n_boot": _int("N_BOOT", raw["N_BOOT"], 0),
        "seed": _int("SEED", raw["SEED"]),
        "variables": None if variables.lower() in ("all", "") else _list(variables),
        "figure_variables": figvars.lower() if figvars.lower() in ("auto", "all") else _list(figvars),
        "figures": _bool("FIGURES", raw["FIGURES"]),
    }
    _int("THREADS", raw["THREADS"], 0)
    return raw, p


def num(x, nd: int = 6):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(xf):
        return None
    return float(f"{xf:.{nd}g}") if abs(xf) >= 1 else round(xf, nd)


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.7g}" if math.isfinite(float(v)) else ""
    return str(v)


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([cell(r.get(h)) for h in header])


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def slug(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-." else "-" for c in str(s).lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


# --------------------------------------------------------------------------- per recording

def recording_species(rid: str, prm: dict, recs: dict, prov_species: dict) -> tuple[str, str]:
    if prm["species"] != "auto":
        return PS.normalise_species(prm["species"]), "SPECIES parameter"
    for src, val in (("recordings.csv", (recs.get(rid) or {}).get("species")), ("upstream provenance.json",
                                                                                prov_species.get(rid))):
        s = PS.normalise_species(val)
        if s:
            return s, src
    return "human", "default (no species in the inputs)"


def settings_for(species: str, prm: dict) -> tuple[RS.Settings, dict]:
    pre = PS.PRESETS[species]
    b = prm["baseline_s"] if prm["baseline_s"] is not None else pre["baseline_s"]
    a = prm["after_s"] if prm["after_s"] is not None else pre["after_s"]
    used = {"species": species, "baseline_s": b, "after_s": a,
            "baseline_source": "BASELINE_S" if prm["baseline_s"] is not None else pre["source"],
            "after_source": "AFTER_S" if prm["after_s"] is not None else pre["source"]}
    return RS.Settings(baseline_s=b, after_s=a, response_window=prm["response_window"], fixed=prm["fixed"],
                       peak_window_s=prm["peak_window_s"], recovery_fraction=prm["recovery_fraction"],
                       min_valid=prm["min_valid"], overlap=prm["overlap"]), used


def process_recording(rid: str, events: list[RS.Event], series, settings: RS.Settings, subject: str,
                      variables: list[str]) -> tuple[list[dict], dict]:
    """Response rows of one recording, and the aligned time courses for the figures."""
    rows, courses = [], {}
    wins = {e.event_id: RS.event_windows(e, [o for o in events if o is not e], settings) for e in events}
    for var in variables:
        sub = series[series["variable"] == var]
        t = sub["time_s"].to_numpy(float)
        v = sub["value"].to_numpy(float)
        ok = sub["valid"].to_numpy(bool) & np.isfinite(v)
        src = str(sub["source"].iloc[0]) if len(sub) else ""
        rate = RS.series_rate(t)
        smooth = RS.moving_mean(v, ok, round(settings.peak_window_s * rate))
        unit = PS.unit_of(var)
        max_dur = max((e.end - e.onset for e in events), default=0.0)
        step = 1.0 / rate if rate > 0 else 0.25
        grid = np.arange(-settings.baseline_s, max_dur + settings.after_s + step / 2, step)
        for e in events:
            w = wins[e.event_id]
            r = RS.respond(e, w, t, v, ok, smooth, rate, settings)
            reasons = list(w["reasons"])
            for wname, key in (("baseline", "valid_fraction_baseline"), ("response window", "valid_fraction_response")):
                if r[key] < settings.min_valid:
                    reasons.append(f"{wname} valid share {r[key]:.2f} < MIN_VALID {settings.min_valid:g}")
            if r["baseline"] is None:
                reasons.append("no valid baseline samples")
            if r["delta"] is None:
                reasons.append("no valid samples in the response window")
            included = not w["excluded"] and r["delta"] is not None and r["valid_fraction"] >= settings.min_valid
            row = {"recording_id": rid, "event_id": e.event_id, "variable": var, "subject": subject, "unit": unit,
                   "series_source": src, "onset": e.onset, "duration": (e.end - e.onset) if e.duration_known else None,
                   "trial_type": e.trial_type, "included": included,
                   "exclusion_reason": ("; ".join(x for x in reasons if x.startswith("duration")) if included
                                        else "; ".join(reasons) or "excluded"),
                   **{k: r[k] for k in ("baseline", "during", "after", "delta", "pct_change", "peak_delta",
                                        "time_to_peak_s", "recovery_s", "valid_fraction", "valid_fraction_baseline",
                                        "valid_fraction_response", "after_s_used", "window_start_s",
                                        "window_end_s", "baseline_sd")},
                   "_next_onset": w["next_onset"],
                   "_baseline_var": (r["baseline_sd"] ** 2) if r["baseline_sd"] is not None else None,
                   "_params": e.params}
            rows.append(row)
            if len(t) and r["baseline"] is not None:
                courses[(var, e.event_id)] = RS.aligned(e, t, v, ok, r["baseline"], grid)
        courses[("_grid", var)] = grid
    return rows, courses


# --------------------------------------------------------------------------- fits

def x_unit(x: str, params: list[dict]) -> str:
    if x == "amplitude":
        units = {str(p.get("amplitude_unit")) for p in params if p.get("amplitude_unit") not in (None, "", "n/a")}
        return units.pop() if len(units) == 1 else ""
    return {"frequency_hz": "Hz", "pulse_width_us": "us", "train_duration_s": "s", "train_on_s": "s",
            "train_off_s": "s"}.get(x, "")


def subject_criteria(rows: list[dict], spec: str) -> dict:
    """(subject, trial_type, variable) -> (criterion value, description)."""
    acc = defaultdict(lambda: {"base": [], "var": []})
    for r in rows:
        if not r["included"]:
            continue
        k = (r["subject"], r["trial_type"], r["variable"])
        acc[k]["base"].append(r["baseline"])
        if r["_baseline_var"] is not None:
            acc[k]["var"].append(r["_baseline_var"])
    out = {}
    for k, a in acc.items():
        bm = float(np.mean(a["base"])) if a["base"] else None
        # baseline variability: the train-to-train SD of the baseline means (what a change would look like
        # without stimulation); with fewer than 3 trains, the pooled within-window SD of the baseline samples
        if len(a["base"]) >= 3:
            bsd = float(np.std(a["base"], ddof=1))
        else:
            bsd = float(np.sqrt(np.mean(a["var"]))) if a["var"] else None
        out[k] = DR.criterion(spec, bm, bsd, PS.unit_of(k[2]))
    return out


def fit_seed(seed: int, *key) -> np.random.Generator:
    """A generator per fit, from SEED and the fit's identity, so adding a variable or a recording does not
    change any other fit's bootstrap."""
    h = hashlib.sha256("|".join(map(str, (seed, *key))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _fit_row(base: dict, x: np.ndarray, y: np.ndarray, prm: dict, crit, tested, pooled=None) -> dict:
    """One fits.csv row. Per subject: x, y are the included trains. Pooled: x, y are the mixed-model level
    estimates and pooled["se"] their standard errors (weighted fit; n = number of levels)."""
    sigma = pooled["se"] if pooled is not None else None
    fit, aics, note = DR.choose(x, y, prm["models"], prm["min_levels"], sigma=sigma)
    d = DR.describe(fit, x)
    ux, means, _ = DR.level_means(x, y)
    thr, direction, bound = DR.threshold(ux, means, crit[0], tested)
    notes = [n for n in (note, "" if direction in ("increase", "decrease") else direction) if n]
    if bound == "upper":
        notes.append("threshold is an upper bound: the next lower tested level was not analysed or not tested")
    if pooled is not None:
        notes.append(pooled["note"])
        if sigma is None:
            notes.append("no standard errors for the level estimates: no intervals")
    cis = DR.intervals(fit, x) if (pooled is None or sigma is not None) else {}
    if fit is not None and fit.model != "none" and not cis:
        notes.append("too few points for intervals")
    if d["ec50"] is not None and not (ux.min() <= d["ec50"] <= ux.max()):
        notes.append("EC50 outside the tested range (extrapolated)")
    row = dict(base)
    row.update({"model": fit.model if fit else "none", "threshold": thr, "threshold_bound": bound,
                "threshold_criterion": crit[1], "criterion_value": crit[0],
                "direction": direction if direction in ("increase", "decrease") else "",
                "ec50": d["ec50"], "top": d["top"], "bottom": d["bottom"], "slope": d["slope"],
                "r2": fit.r2 if fit else None, "n": len(y), "n_levels": len(ux), "aic": fit.aic if fit else None,
                **{f"aic_{m}": aics.get(m) for m in DR.ALL_MODELS}, **cis,
                "levels": "|".join(DR.fmt_level(float(u)) for u in ux),
                "level_means": "|".join(f"{m:.4g}" for m in means), "note": "; ".join(notes),
                "_fit": fit, "_x": x, "_y": y, "_ux": ux, "_means": means})
    return row


def suppress_for_fulcrum(row: dict) -> None:
    """The response changes sign along x: a monotone curve's slope, top and EC50 do not describe it."""
    for k in ("ec50", "top", "bottom", "slope", "ec50_ci_low", "ec50_ci_high", "top_ci_low", "top_ci_high",
              "slope_ci_low", "slope_ci_high"):
        row[k] = None
    row["note"] = "; ".join(n for n in (row["note"], "the response changes sign along x (fulcrum.csv): the "
                                        "monotone fit's slope, top and EC50 are left empty") if n)
    row["sign_change"] = True


def fit_all(rows: list[dict], events_by_rec: dict, prm: dict, crits: dict) -> tuple[list, list, list]:
    """Per-subject and pooled fits, surfaces and fulcrums."""
    fits, surfaces, fulcrums = [], [], []
    by_subject = defaultdict(list)
    for rid, evs in events_by_rec.items():
        for e in evs:
            by_subject[(e.params.get("_subject"), e.trial_type)].append(e)
    resp = defaultdict(dict)    # (variable) -> {(rid, event_id): row}
    for r in rows:
        if r["included"]:
            resp[r["variable"]][(r["recording_id"], r["event_id"])] = r
    variables = sorted({r["variable"] for r in rows})

    def designs(evs):
        numeric, varying, levels = DR.design([e.params for e in evs], I.to_float)
        return numeric, varying, levels

    subject_sets = defaultdict(list)    # (trial_type, species) -> [(subject, events)]
    for (subject, tt), evs in sorted(by_subject.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        species = evs[0].params.get("_species", "")
        subject_sets[(tt, species)].append((subject, evs))
        numeric, varying, levels = designs(evs)
        xs = [k for k in numeric if len({lv[k] for lv in levels if lv[k] is not None}) >= 2]
        for x in xs:
            others = [k for k in varying if k != x]
            groups = defaultdict(list)
            for e, lv in zip(evs, levels):
                if lv[x] is not None:
                    groups[DR.group_label(lv, others)].append((e, lv))
            for g, members in sorted(groups.items()):
                tested = sorted({lv[x] for _, lv in members})
                for var in variables:
                    pts = [(lv[x], resp[var][(e.recording_id, e.event_id)]["delta"], e) for e, lv in members
                           if (e.recording_id, e.event_id) in resp[var]]
                    base = {"subject": subject, "recording_id": "|".join(sorted({e.recording_id for e, _ in members})),
                            "variable": var, "x": x, "x_unit": x_unit(x, [e.params for e, _ in members]), "group": g,
                            "unit": PS.unit_of(var), "trial_type": tt, "n_subjects": 1, "species": species}
                    if len({p[0] for p in pts}) < 2:
                        # one level (or none) analysed in this group: a row saying so, no curve
                        why = ("only one level of x was tested in this group" if len(tested) < 2 else
                               f"{len({p[0] for p in pts})} of {len(tested)} tested levels have included trains")
                        if not any(r["variable"] == var and r["subject"] == subject for r in rows):
                            continue
                        fits.append({**base, "model": "none", "n": len(pts), "n_levels": len({p[0] for p in pts}),
                                     "levels": "|".join(DR.fmt_level(float(p)) for p in sorted({p[0] for p in pts})),
                                     "level_means": "|".join(f"{np.mean([q[1] for q in pts if q[0] == u]):.4g}"
                                                             for u in sorted({p[0] for p in pts})),
                                     "note": why + ": no fit, no threshold", "_fit": None})
                        continue
                    xa = np.array([p[0] for p in pts], float)
                    ya = np.array([p[1] for p in pts], float)
                    crit = crits.get((subject, tt, var), (None, "no baseline"))
                    row = _fit_row(base, xa, ya, prm, crit, tested)
                    fits.append(row)
                    if x == "amplitude" and var.startswith(PS.FULCRUM_PREFIX) and crit[0] is not None:
                        xf, note = DR.fulcrum(row["_ux"], row["_means"], crit[0])
                        if xf is not None:
                            suppress_for_fulcrum(row)
                            freqs = {lv.get("frequency_hz") for _, lv in members if lv.get("frequency_hz") is not None}
                            cis = DR.bootstrap_fulcrum(xa, ya, crit[0], prm["n_boot"],
                                                       fit_seed(prm["seed"], "fulcrum", subject, tt, var, x, g))
                            fulcrums.append({"subject": subject, "variable": var, "species": species,
                                             "frequency_hz": freqs.pop() if len(freqs) == 1 else None,
                                             "amplitude_at_fulcrum": xf,
                                             "method": "linear interpolation between the bracketing level means",
                                             "ci_low": cis.get("ci_low"), "ci_high": cis.get("ci_high"), "group": g,
                                             "recording_id": base["recording_id"], "levels": row["levels"],
                                             "level_means": row["level_means"],
                                             "note": "; ".join(n for n in (note, "" if cis else
                                                               "no CI: one train per level") if n)})
        # response surface: amplitude x frequency
        if "amplitude" in xs and "frequency_hz" in xs:
            others = [k for k in varying if k not in ("amplitude", "frequency_hz")]
            for var in variables:
                cells = defaultdict(list)
                for e, lv in zip(evs, levels):
                    r = resp[var].get((e.recording_id, e.event_id))
                    if r is not None and lv["amplitude"] is not None and lv["frequency_hz"] is not None:
                        cells[(DR.group_label(lv, others), lv["amplitude"], lv["frequency_hz"])].append(r["delta"])
                for (g, a, f), ds in sorted(cells.items()):
                    surfaces.append({"subject": subject, "variable": var, "amplitude": a, "frequency_hz": f,
                                     "mean_delta": float(np.mean(ds)),
                                     "sd": float(np.std(ds, ddof=1)) if len(ds) > 1 else None, "n": len(ds),
                                     "group": g, "trial_type": tt, "unit": PS.unit_of(var),
                                     "species": species})

    # pooled, over subjects (>= 3) of one species, on a design computed over all of them; only the levels
    # tested in >= 3 subjects
    for (tt, species), sets in subject_sets.items():
        if len(sets) < MIN_SUBJECTS_POOLED:
            continue
        evs = [e for _, es in sets for e in es]
        numeric, varying, levels = designs(evs)
        xs = [k for k in numeric if len({lv[k] for lv in levels if lv[k] is not None}) >= 2]
        for x in xs:
            others = [k for k in varying if k != x]
            groups = defaultdict(list)
            for e, lv in zip(evs, levels):
                if lv[x] is not None:
                    groups[DR.group_label(lv, others)].append((e, lv))
            for g, members in sorted(groups.items()):
                for var in variables:
                    pts = [(lv[x], resp[var][(e.recording_id, e.event_id)]["delta"], e.params.get("_subject"))
                           for e, lv in members if (e.recording_id, e.event_id) in resp[var]]
                    n_sub = defaultdict(set)
                    for p in pts:
                        n_sub[p[0]].add(p[2])
                    dropped = sorted(lv for lv, ss in n_sub.items() if len(ss) < MIN_SUBJECTS_POOLED)
                    pts = [p for p in pts if p[0] not in dropped]
                    subs = {p[2] for p in pts}
                    if len(subs) < MIN_SUBJECTS_POOLED or len({p[0] for p in pts}) < 2:
                        continue
                    xa = np.array([p[0] for p in pts], float)
                    ya = np.array([p[1] for p in pts], float)
                    sa = np.array([str(p[2]) for p in pts])
                    ux, est, se, note = DR.mixed_level_estimates(xa, ya, sa)
                    cvals = [crits.get((s, tt, var), (None, ""))[0] for s in subs]
                    cvals = [c for c in cvals if c is not None]
                    crit = ((float(np.mean(cvals)), "mean of the subjects' criteria") if cvals else (None, "no baseline"))
                    base = {"subject": "pooled", "recording_id": "pooled", "variable": var, "x": x,
                            "x_unit": x_unit(x, [e.params for e, _ in members]), "group": g,
                            "unit": PS.unit_of(var), "trial_type": tt, "n_subjects": len(subs), "species": species}
                    row = _fit_row(base, ux, est, prm, crit, sorted({lv[x] for _, lv in members}),
                                   pooled={"se": se, "note": note + f"; n = {len(ux)} level estimates from "
                                                                     f"{len(ya)} trains"})
                    if dropped:
                        row["note"] = "; ".join(n for n in (row["note"], "levels tested in fewer than 3 subjects "
                                                            "left out: " + ",".join(DR.fmt_level(d) for d in dropped)) if n)
                    fits.append(row)
                    if x == "amplitude" and var.startswith(PS.FULCRUM_PREFIX) and crit[0] is not None:
                        xf, fnote = DR.fulcrum(ux, est, crit[0])
                        if xf is not None:
                            suppress_for_fulcrum(row)
                            freqs = {lv.get("frequency_hz") for _, lv in members if lv.get("frequency_hz") is not None}
                            fulcrums.append({"subject": "pooled", "variable": var,
                                             "frequency_hz": freqs.pop() if len(freqs) == 1 else None,
                                             "amplitude_at_fulcrum": xf,
                                             "method": "linear interpolation between the bracketing mixed-model level estimates",
                                             "group": g, "recording_id": "pooled", "levels": row["levels"],
                                             "level_means": row["level_means"], "note": fnote, "species": species})
        if "amplitude" in xs and "frequency_hz" in xs:
            subj_cells = defaultdict(lambda: defaultdict(list))
            for s in surfaces:
                if s["trial_type"] == tt and s["subject"] != "pooled" and s.get("species") == species:
                    subj_cells[(s["variable"], s["group"], s["amplitude"], s["frequency_hz"])][s["subject"]].append(
                        s["mean_delta"])
            for (var, g, a, f), by_s in sorted(subj_cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
                if len(by_s) < MIN_SUBJECTS_POOLED:
                    continue
                ms = [float(np.mean(v)) for v in by_s.values()]
                surfaces.append({"subject": "pooled", "variable": var, "amplitude": a, "frequency_hz": f,
                                 "mean_delta": float(np.mean(ms)), "sd": float(np.std(ms, ddof=1)), "n": len(ms),
                                 "group": g, "trial_type": tt, "unit": PS.unit_of(var), "species": species})
    return fits, surfaces, fulcrums


# --------------------------------------------------------------------------- main

def main() -> int:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc)
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    try:
        raw, prm = read_params()
    except ValueError as e:
        log.error("parameters: %s", e)
        return 2
    out = output_dir / APP_NAME
    out.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s threads=%d (%s)", APP_NAME, app_version(), input_dir, out,
             _THREADS, _THREADS_WHY)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s does not exist", input_dir)
        return 1
    found = I.discover(input_dir)

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(input_dir))
        except ValueError:
            return str(p)

    for p in found.ignored_events:
        log.info("ignored %s: physio-import's markers (stimulation comes from stim-events/events.tsv)", rel(p))
    if not found.events:
        log.error("no stim-events events.tsv under %s: link the events output of stim-events", input_dir)
        return 1
    if not found.series:
        log.error("no series.csv under %s: link the hrv output of ecg-hrv and/or the bp output of bp-resp", input_dir)
        return 1
    warnings_out: list[str] = []
    ev_rows = I.read_events(found.events)
    events_all, w = RS.make_events(ev_rows, I.to_float)
    warnings_out += w
    series, w = I.read_series(found.series)
    warnings_out += w
    recs = I.read_recordings(found.recordings)
    prov_species = I.species_from_provenance(found.provenance)
    log.info("inputs: %d events.tsv (%d trains), %d series.csv (%d recordings, %d variables), %d recordings.csv",
             len(found.events), len(events_all), len(found.series), series["recording_id"].nunique(),
             series["variable"].nunique(), len(found.recordings))

    events_by_rec = defaultdict(list)
    for e in events_all:
        events_by_rec[e.recording_id].append(e)
    rec_ids = sorted(set(series["recording_id"].unique()) | set(events_by_rec) |
                     {r["recording_id"] for r in ev_rows if r.get("recording_id")})
    per_rec, all_rows, courses_all, settings_used = {}, [], {}, {}
    n_ok = n_failed = 0
    series_groups = {rid: g for rid, g in series.groupby("recording_id", sort=False)}
    for i, rid in enumerate(rec_ids, 1):
        t0 = time.monotonic()
        subject = I.subject_of(rid, recs)
        species, sp_src = recording_species(rid, prm, recs, prov_species)
        evs = events_by_rec.get(rid, [])
        sg = series_groups.get(rid)
        if not evs:
            per_rec[rid] = {"status": "skipped", "reason": "no stimulation or block events for this recording",
                            "subject": subject, "species": species}
            log.info("[%d/%d] %s: skipped (no stimulation events)", i, len(rec_ids), rid)
            continue
        if sg is None or not len(sg):
            per_rec[rid] = {"status": "skipped", "reason": "no series.csv rows for this recording "
                            "(no ECG/heart rate or blood pressure analysed upstream)", "subject": subject,
                            "species": species}
            warnings_out.append(f"{rid}: {len(evs)} trains but no series.csv for this recording: skipped")
            log.info("[%d/%d] %s: skipped (no series)", i, len(rec_ids), rid)
            continue
        variables = sorted(sg["variable"].unique())
        if prm["variables"]:
            variables = [v for v in variables if v in prm["variables"]]
        if not variables:
            per_rec[rid] = {"status": "skipped", "reason": "none of VARIABLES in this recording's series",
                            "subject": subject, "species": species}
            continue
        settings, used = settings_for(species, prm)
        settings_used[rid] = {**used, "species_source": sp_src}
        for e in evs:
            e.params["_subject"] = subject
            e.params["_species"] = species
        try:
            rows, courses = process_recording(rid, evs, sg, settings, subject, variables)
        except Exception as e:  # noqa: BLE001
            log.error("%s: failed: %s", rid, e)
            log.info(traceback.format_exc())
            per_rec[rid] = {"status": "failed", "error": str(e), "subject": subject, "species": species}
            warnings_out.append(f"{rid}: failed: {e}")
            n_failed += 1
            continue
        all_rows += rows
        courses_all[rid] = courses
        n_inc = len({r["event_id"] for r in rows if r["included"]})
        secs = time.monotonic() - t0
        per_rec[rid] = {"status": "ok", "seconds": round(secs, 3), "species": species, "subject": subject,
                        "n_trains": len(evs), "n_trains_included": n_inc, "variables": variables}
        n_ok += 1
        log.info("[%d/%d] %s: subject %s, %s (%s), %d trains (%d included in fits for at least one variable), "
                 "%d variables, baseline %gs after %gs, %.2fs", i, len(rec_ids), rid, subject, species, sp_src,
                 len(evs), n_inc, len(variables), settings.baseline_s, settings.after_s, secs)

    crits = subject_criteria(all_rows, prm["criterion"])
    t_fit = time.monotonic()
    fits, surfaces, fulcrums = fit_all(all_rows, {r: [e for e in events_by_rec[r] if per_rec.get(r, {}).get("status")
                                                       == "ok"] for r in events_by_rec}, prm, crits)
    log.info("fits: %d rows (%d pooled), %d surface cells, %d fulcrum(s), %.1fs", len(fits),
             sum(1 for f in fits if f["subject"] == "pooled"), len(surfaces), len(fulcrums), time.monotonic() - t_fit)

    # ---- tables
    param_cols = []
    for e in events_all:
        for k in e.params:
            if k not in param_cols and not k.startswith("_") and k not in ("label", "source", "trial_group"):
                param_cols.append(k)
    param_cols = [k for k in EVENT_PARAM_ORDER if k in param_cols] + [k for k in param_cols
                                                                       if k not in EVENT_PARAM_ORDER]
    resp_rows = []
    for r in all_rows:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        for k in param_cols:
            v = r["_params"].get(k)
            rr[k] = None if v is None or str(v).lower() == "n/a" else v
        resp_rows.append(rr)
    write_csv(out / "responses.csv", RESP_HEADER + param_cols, resp_rows)
    write_csv(out / "fits.csv", FIT_HEADER, fits)
    write_csv(out / "surface.csv", SURF_HEADER, surfaces)
    write_csv(out / "fulcrum.csv", FULC_HEADER, fulcrums)

    features = []
    for rid, info in per_rec.items():
        if info["status"] != "ok":
            continue
        rws = [r for r in all_rows if r["recording_id"] == rid]
        features.append({"recording_id": rid, "feature": "dose_n_trains", "value": info["n_trains"]})
        features.append({"recording_id": rid, "feature": "dose_n_trains_included", "value": info["n_trains_included"]})
        for var in info["variables"]:
            ds = [r["delta"] for r in rws if r["variable"] == var and r["included"]]
            if ds:
                j = int(np.argmax(np.abs(ds)))
                features.append({"recording_id": rid, "feature": f"resp_max_delta_{var}", "value": ds[j]})
    for f in fits:
        if f["subject"] == "pooled" or "|" in f["recording_id"]:
            continue
        tag = f"{f['variable']}_{f['x']}" + (f"_{slug(f['group'])}" if f["group"] else "")
        for k in ("threshold", "ec50"):
            if f.get(k) is not None:
                features.append({"recording_id": f["recording_id"], "feature": f"dose_{k}_{tag}", "value": f[k]})
    for fu in fulcrums:
        if fu["subject"] != "pooled" and "|" not in fu["recording_id"]:
            features.append({"recording_id": fu["recording_id"], "feature":
                             f"fulcrum_amplitude_{fu['variable']}" + (f"_{slug(fu['group'])}" if fu["group"] else ""),
                             "value": fu["amplitude_at_fulcrum"]})
    write_csv(out / "features.csv", FEATURE_HEADER, features)

    # ---- figures
    figures = []
    if prm["figures"] and all_rows:
        from processor import figures as FG  # noqa: PLC0415 - matplotlib only when figures are wanted

        all_vars = sorted({r["variable"] for r in all_rows})
        if prm["figure_variables"] == "all":
            figvars = all_vars
        elif prm["figure_variables"] == "auto":
            figvars = [v for v in PS.FIGURE_DEFAULT if v in all_vars]
            if "hr_bpm" in figvars and "hr_device_bpm" in figvars:
                figvars.remove("hr_device_bpm")
            figvars = figvars or all_vars[:2]
        else:
            figvars = [v for v in prm["figure_variables"] if v in all_vars]
        try:
            figures = FG.make_all(out / "figures", all_rows, courses_all, fits, surfaces, fulcrums, figvars,
                                  settings_used, found, recs, input_dir)
        except Exception as e:  # noqa: BLE001
            log.error("figures failed: %s", e)
            log.info(traceback.format_exc())
            warnings_out.append(f"figures failed: {e}")

    # ---- summary
    if not events_all:
        warnings_out.append("events.tsv has no stimulation or block rows: nothing to analyse (resting recordings?)")
    if n_ok == 0 and not n_failed:
        warnings_out.append("no recording had both stimulation events and a series: empty tables")
    headline = {}
    for rid, info in per_rec.items():
        h = {"subject": info.get("subject"), "species": info.get("species"), "status": info["status"]}
        if info["status"] == "ok":
            h.update(n_trains=info["n_trains"], n_trains_included=info["n_trains_included"],
                     variables=info["variables"])
            for var in info["variables"]:
                ds = [r["delta"] for r in all_rows if r["recording_id"] == rid and r["variable"] == var and r["included"]]
                if ds:
                    h[f"max_abs_delta_{var}"] = num(max(ds, key=abs))
        else:
            h["reason"] = info.get("reason") or info.get("error")
        headline[rid] = h
    subjects = defaultdict(lambda: {"fits": [], "fulcrum": []})
    for f in fits:
        subjects[f["subject"]]["fits"].append(
            {k: (num(f[k]) if isinstance(f.get(k), float) else f.get(k)) for k in
             ("variable", "x", "group", "trial_type", "model", "threshold", "threshold_criterion", "ec50",
              "ec50_ci_low", "ec50_ci_high", "top", "r2", "n", "n_levels", "unit", "direction")})
    for fu in fulcrums:
        subjects[fu["subject"]]["fulcrum"].append({k: (num(fu[k]) if isinstance(fu.get(k), float) else fu.get(k))
                                                   for k in ("variable", "frequency_hz", "amplitude_at_fulcrum",
                                                             "ci_low", "ci_high", "group")})
    notice = RESEARCH_NOTICE + (" " + FULCRUM_CAVEAT if fulcrums else "")
    tables = [
        {"path": "responses.csv", "caption": "One row per train and variable: baseline, during and after means, the "
         "change (delta, pct_change), peak change, time to peak, recovery time, the share of valid samples, whether "
         "the train was used in the fits (and why not), and the train's stimulation parameters."},
        {"path": "fits.csv", "caption": "Dose-response per subject (and pooled over three or more subjects): for "
         "each parameter that varies with the others held fixed, the chosen model (Hill, Boltzmann, linear or none, "
         "by AICc), threshold (with whether it is bracketed or an upper bound), EC50, top, slope, r², AICc of every "
         "model and 95% intervals (t-based per subject, from the mixed model's standard errors when pooled)."},
    ]
    if surfaces:
        tables.append({"path": "surface.csv", "caption": "Mean change per amplitude x frequency cell (response "
                       "surface), per subject and pooled."})
    if fulcrums:
        tables.append({"path": "fulcrum.csv", "caption": "Neural fulcrum: the amplitude where the heart-rate "
                       "response changes sign at a fixed frequency (see the notice for its limits)."})
    summary = {"app": APP_NAME, "version": app_version(), "recordings": headline,
               "subjects": dict(subjects), "figures": figures, "tables": tables,
               "warnings": warnings_out, "notice": notice}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))

    citations = [DOI_SCIPY, DOI_AIC, DOI_AICC]
    if any(f.get("ci_low") is not None for f in fulcrums):
        citations.append(DOI_BOOTSTRAP)
    if any(f["subject"] == "pooled" for f in fits):
        citations.append(DOI_STATSMODELS)
    if fulcrums:
        citations.append(DOI_FULCRUM)
    finished = datetime.now(timezone.utc)
    prov = {"app": APP_NAME, "version": app_version(), "variant": None, "tools": tool_versions(),
            "parameters": raw,
            "inputs": sorted(rel(p) for p in found.events + found.series + found.recordings),
            "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "finished": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_seconds": round(time.monotonic() - t_start, 3), "peak_rss_mb": peak_rss_mb(),
            "threads": {"requested": int(raw["THREADS"]), "used": _THREADS, "why": _THREADS_WHY},
            "recordings": per_rec, "windows": settings_used,
            "species_presets": PS.PRESETS, "citations": citations, "licence_notice": None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=_json_default))
    log.info("done: %d ok, %d skipped, %d failed; %d response rows, %d fits; peak %.0f MB, %.1fs", n_ok,
             sum(1 for v in per_rec.values() if v["status"] == "skipped"), n_failed, len(resp_rows), len(fits),
             prov["peak_rss_mb"], prov["wall_seconds"])
    if n_ok == 0 and n_failed:
        return 1
    return 0


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return num(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
