"""
Raw traces around one train, from physio-import's NWB files (h5py, read by slice; no pynwb at run time).
Only used for one figure per recording when the physio port is linked.
"""
from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np

TRACE_TYPES = ("ecg", "bp", "resp", "hr", "ppg")
MAX_RATE = 5000.0


def _search(ds, value: float) -> int:
    """searchsorted on a 1-D h5py dataset without reading it whole (binary search, ~log2(n) reads)."""
    lo, hi = 0, ds.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if float(ds[mid]) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def channel_types(recordings_csv_dir: Path, recording_id: str) -> dict[str, str]:
    """channel -> type from channels.csv next to recordings.csv; empty when there is none. Channels that
    physio-import derived from another channel (`derived_from`, e.g. a filtered copy of the ECG) get the
    type "derived" and are not drawn: the raw signal is shown."""
    p = Path(recordings_csv_dir) / "channels.csv"
    out = {}
    if p.exists():
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("recording_id") == recording_id:
                    derived = (r.get("derived_from") or "").strip() not in ("", "n/a")
                    out[r.get("channel", "")] = "derived" if derived else (r.get("type") or "").lower()
    return out


def read_window(nwb: Path, t0: float, t1: float, types: dict[str, str]) -> list[dict]:
    """[{name, type, unit, t, y}] of the trace channels in [t0, t1]."""
    out = []
    with h5py.File(nwb, "r") as f:
        acq = f.get("acquisition")
        if acq is None:
            return out
        for name in acq:
            g = acq[name]
            if not isinstance(g, h5py.Group) or "data" not in g:
                continue
            typ = types.get(name) or name.split("-")[0].lower()
            if typ not in TRACE_TYPES:
                continue
            data = g["data"]
            if data.ndim != 1:
                continue
            conv = float(data.attrs.get("conversion", 1.0))
            off = float(data.attrs.get("offset", 0.0))
            unit = data.attrs.get("unit", "")
            unit = unit.decode() if isinstance(unit, bytes) else str(unit)
            if "timestamps" in g:
                ts = g["timestamps"]
                i0, i1 = _search(ts, t0), _search(ts, t1)
                if i1 - i0 < 2:
                    continue
                t = np.asarray(ts[i0:i1], float)
                rate = (i1 - i0) / max(t[-1] - t[0], 1e-9)
            elif "starting_time" in g:
                st = g["starting_time"]
                rate = float(st.attrs.get("rate", 0)) or 1.0
                start = float(st[()])
                i0 = max(0, int(np.floor((t0 - start) * rate)))
                i1 = min(data.shape[0], int(np.ceil((t1 - start) * rate)))
                if i1 - i0 < 2:
                    continue
                t = start + np.arange(i0, i1) / rate
            else:
                continue
            if rate > MAX_RATE:
                continue
            y = np.asarray(data[i0:i1], float) * conv + off
            out.append({"name": name, "type": typ, "unit": unit, "t": t, "y": y})
    order = {k: i for i, k in enumerate(TRACE_TYPES)}
    out.sort(key=lambda c: (order.get(c["type"], 99), c["name"]))
    return out
