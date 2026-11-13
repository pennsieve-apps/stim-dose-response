"""
Finding and reading the inputs, by name and content, recursively, following symlinks.

events port   stim-events' `stim-events/events.tsv` (one row per train). physio-import also writes a
              file called events.tsv (raw markers, at the root of its physio port, next to
              recordings.csv): it is never read as stimulation events.
hrv, bp ports any number of `series.csv` files (recording_id, time_s, variable, value, valid), from
              ecg-hrv, bp-resp or any future step that writes the same shape.
physio port   optional: physio-import's recordings.csv (subject, species) and NWB files (raw traces
              for one figure per recording).
Upstream provenance.json files give the species of a recording when there is no recordings.csv.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("stim-dose-response")

SERIES_COLUMNS = ("recording_id", "time_s", "variable", "value", "valid")
EVENT_REQUIRED = ("recording_id", "onset", "trial_type")
NA = ("", "n/a", "na", "nan", "none", "null")


def walk(root: Path):
    """Every file under root, recursively, following symlinks (INPUT_DIR entries are symlinks); no loops."""
    seen: set = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
            key = (st.st_dev, st.st_ino)
        except OSError:
            continue
        if key in seen:
            dirnames[:] = []
            continue
        seen.add(key)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith((".", "~$")):
                yield Path(dirpath) / f


def _header(path: Path) -> list[str]:
    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return []
    delim = "\t" if first.count("\t") > first.count(",") else ","
    return [c.strip() for c in first.rstrip("\r\n").split(delim)]


@dataclass
class Found:
    events: list[Path] = field(default_factory=list)          # stim-events events.tsv
    ignored_events: list[Path] = field(default_factory=list)  # physio-import's markers events.tsv
    series: list[Path] = field(default_factory=list)
    recordings: list[Path] = field(default_factory=list)
    provenance: list[Path] = field(default_factory=list)
    nwb: list[Path] = field(default_factory=list)


def discover(root: Path) -> Found:
    f = Found()
    for p in walk(root):
        name = p.name.lower()
        if name == "events.tsv":
            hdr = _header(p)
            if not all(c in hdr for c in EVENT_REQUIRED):
                continue
            # physio-import's markers sit at its port root, next to recordings.csv; stim-events' file is
            # stim-events/events.tsv. A file in neither place is accepted when it has stimulation columns.
            if p.parent.name == "stim-events":
                f.events.append(p)
            elif (p.parent / "recordings.csv").exists() or "event_id" not in hdr:
                f.ignored_events.append(p)
            else:
                f.events.append(p)
        elif name == "series.csv":
            hdr = _header(p)
            if all(c in hdr for c in SERIES_COLUMNS):
                f.series.append(p)
        elif name == "recordings.csv":
            hdr = _header(p)
            if "recording_id" in hdr:
                f.recordings.append(p)
        elif name == "provenance.json":
            f.provenance.append(p)
        elif name.endswith(".nwb"):
            f.nwb.append(p)
    return f


def _na(v) -> bool:
    return v is None or (isinstance(v, float) and not np.isfinite(v)) or str(v).strip().lower() in NA


def to_float(v) -> float | None:
    if _na(v):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def read_events(paths: list[Path]) -> list[dict]:
    """Rows of every stim-events events.tsv (de-duplicated on recording_id + event_id)."""
    out, seen = [], set()
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r = {(k or "").strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
                key = (r.get("recording_id"), r.get("event_id") or r.get("onset"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
    return out


def source_app(path: Path) -> str:
    """The app that wrote a series.csv: the name of its top folder (ecg-hrv/series.csv), else the file's
    parent folder name."""
    return path.parent.name or "series"


def read_series(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """All series.csv files as one long table with a `source` column. A recording/variable pair that
    appears in two files is kept from the first file only (warning)."""
    frames, warnings, owner = [], [], {}
    for p in paths:
        df = pd.read_csv(p, usecols=list(SERIES_COLUMNS), dtype={"recording_id": str, "variable": str},
                         na_values=list(NA), keep_default_na=True)
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["valid"] = df["valid"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        df = df.dropna(subset=["recording_id", "variable", "time_s"])
        src = source_app(p)
        keep = []
        for (rid, var), _ in df.groupby(["recording_id", "variable"], sort=False):
            if (rid, var) in owner:
                warnings.append(f"{rid} {var}: also in {p} (kept the one from {owner[(rid, var)]})")
            else:
                owner[(rid, var)] = str(p)
                keep.append((rid, var))
        if keep:
            k = pd.MultiIndex.from_tuples(keep)
            df = df[pd.MultiIndex.from_frame(df[["recording_id", "variable"]]).isin(k)]
            df = df.assign(source=src)
            frames.append(df)
        log.info("series %s (%s): %d rows, %d recording/variable pairs", p, src, len(df), len(keep))
    if not frames:
        return pd.DataFrame(columns=[*SERIES_COLUMNS, "source"]), warnings
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["recording_id", "variable", "time_s"], kind="stable")
    return out.reset_index(drop=True), warnings


def read_recordings(paths: list[Path]) -> dict[str, dict]:
    """recording_id -> row of recordings.csv (subject, species, path, ...)."""
    out: dict[str, dict] = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
            for r in csv.DictReader(fh):
                rid = (r.get("recording_id") or "").strip()
                if rid and rid not in out:
                    out[rid] = {k: (v or "").strip() for k, v in r.items() if k}
                    out[rid]["_root"] = str(p.parent)
    return out


def species_from_provenance(paths: list[Path]) -> dict[str, str]:
    """recording_id -> species from upstream provenance.json files (ecg-hrv, bp-resp write it)."""
    out: dict[str, str] = {}
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        recs = d.get("recordings") if isinstance(d, dict) else None
        if not isinstance(recs, dict):
            continue
        for rid, info in recs.items():
            if isinstance(info, dict) and info.get("species") and rid not in out:
                out[rid] = str(info["species"])
    return out


_SUB = re.compile(r"^sub-([A-Za-z0-9-]+?)(?:_|$)")


def subject_of(recording_id: str, recordings: dict[str, dict]) -> str:
    r = recordings.get(recording_id)
    if r and r.get("subject"):
        return r["subject"]
    m = _SUB.match(recording_id)
    return m.group(1) if m else recording_id
