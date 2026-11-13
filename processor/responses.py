"""
Baseline, during and after each stimulation, for every variable of every series.csv.

For each train (events.tsv row with trial_type stimulation or block) and each variable:

  baseline  mean of the valid samples in the BASELINE_S before onset
  during    mean over the response window: the train (RESPONSE_WINDOW=train, default), a PEAK_WINDOW_S
            window centred on the largest deviation from baseline during the train and AFTER_S after it
            (peak), or a fixed window relative to onset (fixed:<a>,<b>, seconds)
  after     mean over the AFTER_S after the train ends
  delta = during - baseline, pct_change = 100 * delta / baseline
  peak_delta, time_to_peak_s: the largest deviation of the PEAK_WINDOW_S moving mean from baseline,
            from onset to AFTER_S after the end, and when it happened (seconds after onset)
  recovery_s: seconds after the train ends (or after the peak, if later) until the moving mean is back
            within RECOVERY_FRACTION x |peak_delta| of baseline; searched up to the next train or the end
            of the series; empty when it never got back
  The after window, the peak search and the recovery search end where the next train starts (the
  moving-mean searches half a PEAK_WINDOW_S earlier), so another train's effect never enters them;
  after_s_used says how much of AFTER_S was left.
  valid_fraction: the smaller of the baseline's and the response window's valid share (valid samples /
            samples a full grid would have); each window must reach MIN_VALID

An event is left out of the fits (included=false, with the reason) when its baseline or response window
overlaps another train (OVERLAP=exclude), or when the baseline or the response window has a valid share
below MIN_VALID. With
OVERLAP=clip the baseline is shortened to start after the previous train ends instead, and the event
is left out only when less than half of BASELINE_S remains.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STIM_TYPES = ("stimulation", "block")


@dataclass
class Event:
    recording_id: str
    event_id: str
    onset: float
    end: float
    duration_known: bool
    trial_type: str
    params: dict = field(default_factory=dict)   # every other column, as given


@dataclass
class Settings:
    baseline_s: float
    after_s: float
    response_window: str            # train | peak | fixed
    fixed: tuple[float, float] | None
    peak_window_s: float
    recovery_fraction: float
    min_valid: float
    overlap: str                    # exclude | clip


def make_events(rows: list[dict], to_float) -> tuple[list[Event], list[str]]:
    out, warnings = [], []
    for r in rows:
        tt = (r.get("trial_type") or "").strip().lower()
        if tt not in STIM_TYPES:
            continue
        onset = to_float(r.get("onset"))
        if onset is None:
            warnings.append(f"{r.get('recording_id')} {r.get('event_id')}: no onset, left out")
            continue
        dur = to_float(r.get("duration"))
        if dur is None:
            dur = to_float(r.get("train_duration_s"))
        known = dur is not None and dur > 0
        params = {k: v for k, v in r.items() if k not in ("recording_id", "onset", "duration", "trial_type",
                                                          "event_id")}
        out.append(Event(r["recording_id"], r.get("event_id") or f"ev-{len(out) + 1:04d}", onset,
                         onset + (dur if known else 0.0), known, tt, params))
    out.sort(key=lambda e: (e.recording_id, e.onset))
    return out, warnings


def _mean_valid(t, v, ok, a, b):
    i0, i1 = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "left")
    sel = ok[i0:i1]
    n = int(sel.sum())
    return (float(np.mean(v[i0:i1][sel])) if n else None), n, (float(np.var(v[i0:i1][sel])) if n > 1 else None)


def moving_mean(v: np.ndarray, ok: np.ndarray, width: int) -> np.ndarray:
    """Centred moving mean over `width` samples of the valid samples (NaN where none is valid)."""
    width = max(1, int(width))
    x = np.where(ok, v, 0.0).astype(float)
    w = ok.astype(float)
    k = np.ones(width)
    num = np.convolve(x, k, "same")
    den = np.convolve(w, k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < max(1, width / 2)] = np.nan
    return out


def series_rate(t: np.ndarray) -> float:
    if len(t) < 2:
        return 1.0
    d = np.diff(t)
    d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else 1.0


def overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def event_windows(ev: Event, others: list[Event], s: Settings) -> dict:
    """Window bounds and overlap flags of one event (independent of the variable, except `peak`)."""
    b0, b1 = ev.onset - s.baseline_s, ev.onset
    reasons = []
    end = ev.end if ev.duration_known else ev.onset + s.baseline_s
    if not ev.duration_known:
        reasons.append("duration unknown: the response window is BASELINE_S after onset")
    if s.response_window == "fixed" and s.fixed:
        r0, r1 = ev.onset + s.fixed[0], ev.onset + s.fixed[1]
    else:
        r0, r1 = ev.onset, end
    excluded = False
    prev_end = max([o.end for o in others if o.end <= ev.onset + 1e-9 and o.onset < ev.onset], default=None)
    if s.overlap == "clip" and prev_end is not None and prev_end > b0:
        b0 = min(prev_end, b1)
        if b1 - b0 < 0.5 * s.baseline_s:
            excluded = True
            reasons.append(f"baseline clipped to {b1 - b0:.1f} s after the previous train (< half of BASELINE_S)")
    hit_b = [o.event_id for o in others if overlaps(b0, b1, o.onset, max(o.end, o.onset + 1e-6))]
    hit_r = [o.event_id for o in others if overlaps(r0, r1, o.onset, max(o.end, o.onset + 1e-6))]
    if hit_b:
        excluded = True
        reasons.append("baseline overlaps " + ",".join(hit_b))
    if hit_r:
        excluded = True
        reasons.append("response window overlaps " + ",".join(hit_r))
    nxt = min([o.onset for o in others if o.onset > ev.onset], default=None)
    # the after window, the peak search and the recovery search stop where the next train starts
    a1 = end + s.after_s if nxt is None else max(end, min(end + s.after_s, nxt))
    return {"b0": b0, "b1": b1, "r0": r0, "r1": r1, "end": end, "a0": end, "a1": a1,
            "next_onset": nxt, "excluded": excluded, "reasons": reasons}


def respond(ev: Event, win: dict, t: np.ndarray, v: np.ndarray, ok: np.ndarray, smooth: np.ndarray,
            rate: float, s: Settings) -> dict:
    """The response of one variable to one event."""
    b0, b1, r0, r1 = win["b0"], win["b1"], win["r0"], win["r1"]
    base, nb, var_b = _mean_valid(t, v, ok, b0, b1)
    out = {"baseline": base, "baseline_sd": (np.sqrt(var_b) if var_b is not None else None), "n_baseline": nb}
    # peak: the largest deviation of the moving mean from baseline between onset and AFTER_S after the end,
    # stopping half a smoothing window before the next train so its samples never enter the moving mean
    nxt = win["next_onset"]
    p0, p1 = ev.onset, win["a1"] if nxt is None else min(win["a1"], nxt - s.peak_window_s / 2)
    i0, i1 = np.searchsorted(t, p0, "left"), np.searchsorted(t, p1, "left")
    peak_delta = t_peak = None
    if base is not None and i1 > i0:
        seg = smooth[i0:i1] - base
        if np.isfinite(seg).any():
            j = int(np.nanargmax(np.abs(seg)))
            peak_delta, t_peak = float(seg[j]), float(t[i0 + j])
    if s.response_window == "peak" and t_peak is not None:
        r0, r1 = t_peak - s.peak_window_s / 2, t_peak + s.peak_window_s / 2
    during, nd, _ = _mean_valid(t, v, ok, r0, r1)
    after, _, _ = _mean_valid(t, v, ok, win["a0"], win["a1"]) if win["a1"] > win["a0"] else (None, 0, None)
    # valid share per window (baseline and response separately; the smaller one is valid_fraction)
    vf_b = min(1.0, nb / max(1.0, (b1 - b0) * rate))
    vf_r = min(1.0, nd / max(1.0, (r1 - r0) * rate))
    valid_fraction = min(vf_b, vf_r)
    delta = (during - base) if (during is not None and base is not None) else None
    pct = (100.0 * delta / base) if (delta is not None and base not in (None, 0.0)) else None
    recovery = None
    if peak_delta is not None and base is not None:
        start = max(win["end"], t_peak)
        stop = (nxt - s.peak_window_s / 2) if nxt is not None else t[-1] + 1
        k0, k1 = np.searchsorted(t, start, "left"), np.searchsorted(t, stop, "left")
        tol = s.recovery_fraction * abs(peak_delta)
        seg = smooth[k0:k1]
        hit = np.where(np.isfinite(seg) & (np.abs(seg - base) <= tol))[0]
        if len(hit):
            recovery = float(t[k0 + hit[0]] - win["end"])
    out.update({"during": during, "after": after, "delta": delta, "pct_change": pct, "peak_delta": peak_delta,
                "time_to_peak_s": (t_peak - ev.onset) if t_peak is not None else None,
                "recovery_s": recovery, "valid_fraction": valid_fraction, "valid_fraction_baseline": vf_b,
                "valid_fraction_response": vf_r, "after_s_used": win["a1"] - win["a0"], "window_start_s": r0,
                "window_end_s": r1})
    return out


def aligned(ev: Event, t: np.ndarray, v: np.ndarray, ok: np.ndarray, base: float | None, grid: np.ndarray):
    """The series minus baseline on a grid of seconds relative to onset (NaN where not valid)."""
    if base is None:
        return np.full(len(grid), np.nan)
    tt = ev.onset + grid
    i0 = max(0, np.searchsorted(t, tt[0], "left") - 1)
    i1 = min(len(t), np.searchsorted(t, tt[-1], "right") + 1)
    ts, vs, oks = t[i0:i1], v[i0:i1], ok[i0:i1]
    if len(ts) < 2:
        return np.full(len(grid), np.nan)
    y = np.interp(tt, ts, np.where(oks, vs, np.nan), left=np.nan, right=np.nan)
    # nearest-sample validity: a grid point is valid when its nearest series sample is
    idx = np.clip(np.searchsorted(ts, tt), 1, len(ts) - 1)
    near = np.where(np.abs(ts[idx - 1] - tt) <= np.abs(ts[idx] - tt), idx - 1, idx)
    y[~oks[near]] = np.nan
    return y - base
