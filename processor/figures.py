"""
Figures (PNG, 150 dpi), with captions for summary.json:

  <subject>_<variable>_timecourse.png    change from baseline around the trains, mean ± SD per level
  <subject>_<variable>_dose_response.png responses per train, the chosen fit, threshold and EC50
  <subject>_<variable>_surface.png       mean change per amplitude x frequency cell (heat map)
  pooled_<variable>_dose_response.png    pooled level estimates (>= 3 subjects)
  <recording_id>_raw.png                 raw ECG / BP / respiration around the strongest response
                                         (only when the physio port is linked)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from processor import doseresp as DR  # noqa: E402
from processor import inputs as I  # noqa: E402
from processor import nwbtrace as NT  # noqa: E402
from processor import presets as PS  # noqa: E402

log = logging.getLogger("stim-dose-response")

DPI = 150
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
           "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("bluegrayred", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"])
TRAIN_SHADE = "#f0efec"
MAX_PANELS = 6

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "legend.labelcolor": INK2,
})


def ordinal(n: int) -> list[str]:
    if n <= 1:
        return [ORDINAL[5]]
    idx = np.linspace(0, len(ORDINAL) - 1, n).round().astype(int)
    return [ORDINAL[i] for i in idx]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _fname(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in str(s))


def natural(label: str):
    """Sort key: group labels with numbers in numeric order (frequency_hz=2 before frequency_hz=10)."""
    import re  # noqa: PLC0415

    return [(0, float(t), "") if re.fullmatch(r"-?\d+(\.\d+)?", t) else (1, 0.0, t)
            for t in re.split(r"(-?\d+(?:\.\d+)?)", str(label)) if t]


def _plain_log(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: PLC0415

    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def _ylabel(var: str) -> str:
    return f"Δ {PS.label_of(var)} ({PS.unit_of(var)})"


def make_all(figdir: Path, rows, courses_all, fits, surfaces, fulcrums, figvars, settings_used, found, recs,
             input_dir) -> list[dict]:
    figs: list[dict] = []
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    for subject, srows in sorted(by_subject.items(), key=lambda kv: str(kv[0])):
        for var in figvars:
            vrows = [r for r in srows if r["variable"] == var]
            if not vrows:
                continue
            try:
                f = timecourse(figdir, subject, var, vrows, courses_all, settings_used)
                if f:
                    figs.append(f)
            except Exception as e:  # noqa: BLE001
                log.warning("time course figure %s %s: %s", subject, var, e)
            sfits = [f for f in fits if f["subject"] == subject and f["variable"] == var and f.get("_fit") is not None]
            if sfits:
                figs.append(dose_response(figdir, subject, var, sfits,
                                          [fu for fu in fulcrums if fu["subject"] == subject and fu["variable"] == var]))
            surf = [s for s in surfaces if s["subject"] == subject and s["variable"] == var]
            if surf:
                figs.append(surface(figdir, subject, var, surf))
    pooled_species = sorted({str(f.get("species") or "") for f in fits if f["subject"] == "pooled"})
    for sp in pooled_species:
        label = "pooled" + (f"-{sp}" if sp else "")
        for var in figvars:
            pf = [f for f in fits if f["subject"] == "pooled" and str(f.get("species") or "") == sp
                  and f["variable"] == var and f.get("_fit") is not None]
            if pf:
                figs.append(dose_response(figdir, label, var, pf, [
                    fu for fu in fulcrums if fu["subject"] == "pooled" and str(fu.get("species") or "") == sp
                    and fu["variable"] == var]))
            surf = [s for s in surfaces if s["subject"] == "pooled" and str(s.get("species") or "") == sp
                    and s["variable"] == var]
            if surf:
                figs.append(surface(figdir, label, var, surf))
    if found.nwb and figvars:
        figs += raw_traces(figdir, rows, figvars, found, recs, input_dir, settings_used)
    return figs


# --------------------------------------------------------------------------- time course

def timecourse(figdir, subject, var, vrows, courses_all, settings_used):
    inc = [r for r in vrows if r["included"]]
    if not inc:
        return None
    params = [r["_params"] for r in inc]
    numeric, varying, levels = DR.design(params, I.to_float)
    xs = [k for k in numeric if k in varying]
    x = "amplitude" if "amplitude" in xs else (xs[0] if xs else None)
    others = [k for k in varying if k != x]
    panels = defaultdict(list)
    for r, lv in zip(inc, levels):
        panels[DR.group_label(lv, others)].append((r, lv.get(x) if x else None))
    keys = sorted(panels, key=lambda g: -len(panels[g]))[:MAX_PANELS]
    keys.sort(key=natural)
    rid0 = inc[0]["recording_id"]
    base_s = settings_used[rid0]["baseline_s"]
    after_s = settings_used[rid0]["after_s"]
    durs = [r["duration"] or 0.0 for r in inc]
    step = None
    for r in inc:
        g = courses_all[r["recording_id"]].get(("_grid", var))
        if g is not None and len(g) > 1:
            step = float(g[1] - g[0])
            break
    if step is None:
        return None
    grid = np.arange(-base_s, max(durs + [0.0]) + after_s + step / 2, step)
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False, sharey=True)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, g in zip(axes.flat, keys):
        members = panels[g]
        lvs = sorted({m[1] for m in members}, key=lambda v: (v is None, v if v is not None else 0))
        cols = ordinal(len(lvs))
        pdurs = {round(r["duration"] or 0.0, 3) for r, _ in members}
        if len(pdurs) == 1:
            ax.axvspan(0, pdurs.pop(), color=TRAIN_SHADE, zorder=0, lw=0)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        for lv, c in zip(lvs, cols):
            curves, ends = [], []
            for r, l in members:
                if l != lv:
                    continue
                cg = courses_all[r["recording_id"]].get(("_grid", var))
                cv = courses_all[r["recording_id"]].get((var, r["event_id"]))
                if cg is None or cv is None:
                    continue
                y = np.interp(grid, cg, cv, left=np.nan, right=np.nan)
                y[grid > (r["duration"] or 0.0) + after_s] = np.nan     # each train up to its own after window
                curves.append(y)
                ends.append(r["duration"] or 0.0)
            if not curves:
                continue
            arr = np.vstack(curves)
            with np.errstate(all="ignore"), _quiet():
                m = np.nanmean(arr, axis=0)
                sd = np.nanstd(arr, axis=0) if len(curves) > 1 else None
            label = f"{DR.fmt_level(lv)}" + (f" (n={len(curves)})" if len(curves) > 1 else "")
            ax.plot(grid, m, color=c, lw=1.6, label=label, zorder=3)
            if sd is not None:
                ax.fill_between(grid, m - sd, m + sd, color=c, alpha=0.15, lw=0, zorder=2)
            if len({round(e, 3) for e in ends}) == 1 and len({round(r["duration"] or 0.0, 3) for r, _ in members}) > 1:
                ax.axvline(ends[0], color=c, lw=0.8, ls=":", zorder=1)
        ax.set_title(g.replace(";", ", ") or "all trains", fontsize=9)
        ax.set_xlabel("time from train onset (s)")
        if ax in axes[:, 0]:
            ax.set_ylabel(_ylabel(var))
        ax.legend(title=x or "", fontsize=7, title_fontsize=7, loc="best")
    fig.suptitle(f"{subject}: {PS.label_of(var)} around the trains", color=INK, fontsize=11)
    fig.tight_layout()
    name = f"{_fname(subject)}_{var}_timecourse.png"
    _save(fig, figdir / name)
    more = f" Only the {MAX_PANELS} largest parameter groups are shown." if len(panels) > MAX_PANELS else ""
    return {"path": f"figures/{name}", "caption": (
        f"Subject {subject}: change in {PS.label_of(var)} from its {base_s:g} s baseline, aligned to train onset "
        f"(grey band: the train; where trains of different lengths share a panel, a dotted line in each level's colour marks its end), each train drawn up to {after_s:g} s after its end, one line per {x or 'train'} level (mean across trains, shaded ± SD where a level "
        f"was repeated), one panel per combination of the other parameters. Look for the size, direction and "
        f"timing of the response and whether it returns to baseline after the train. Trains left out of the fits "
        f"(overlapping windows, too few valid samples) are not drawn.{more}")}


class _quiet:
    def __enter__(self):
        import warnings  # noqa: PLC0415

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


# --------------------------------------------------------------------------- dose-response

def dose_response(figdir, subject, var, sfits, fulc):
    curve = [f for f in sfits if "MIN_LEVELS" not in (f.get("note") or "")]
    xs = sorted({f["x"] for f in curve}, key=lambda k: (k != "amplitude", k))
    if not xs:
        xs = [max({f["x"] for f in sfits}, key=lambda k: max(f["n_levels"] for f in sfits if f["x"] == k))]
    fig, axes = plt.subplots(1, len(xs), figsize=(4.6 * len(xs), 3.4), squeeze=False, sharey=True)
    for ax, x in zip(axes[0], xs):
        rows = [f for f in sfits if f["x"] == x]
        rows.sort(key=lambda f: natural(f["group"]))
        rows = rows[:len(CATEGORICAL)]
        allx = np.concatenate([f["_x"] for f in rows])
        logx = bool(np.all(allx > 0) and allx.max() / max(allx.min(), 1e-12) >= 10)
        ax.axhline(0, color=AXIS, lw=0.8)
        for c, f in zip(CATEGORICAL, rows):
            few = "MIN_LEVELS" in (f.get("note") or "")
            lbl = (f["group"].replace(";", ", ") or "all") + (": too few levels" if few else f": {f['model']}")
            if subject.startswith("pooled"):
                ax.plot(f["_ux"], f["_means"], "o", color=c, ms=6, mec=SURFACE, mew=1.2, zorder=3)
            else:
                ax.plot(f["_x"], f["_y"], "o", color=c, ms=5, mec=SURFACE, mew=1.0, alpha=0.9, zorder=3)
            fit = f["_fit"]
            lo, hi = float(np.min(f["_x"])), float(np.max(f["_x"]))
            xx = np.geomspace(lo, hi, 200) if logx else np.linspace(lo, hi, 200)
            if fit is not None and not few:
                ax.plot(xx, DR.FUNCS[fit.model](xx, *fit.params), color=c, lw=2, label=lbl, zorder=2)
            else:
                ax.plot(f["_ux"], f["_means"], color=c, lw=1, ls="--", label=lbl, zorder=2)
            if f.get("threshold") is not None:
                ax.axvline(f["threshold"], color=c, lw=1, ls=":", zorder=1)
            if f.get("ec50") is not None and lo <= f["ec50"] <= hi:
                y50 = float(DR.FUNCS[fit.model](np.array([f["ec50"]]), *fit.params)[0])
                ax.plot([f["ec50"]], [y50], marker="D", color=c, ms=6, mec=INK, mew=0.8, zorder=4)
        for fu in fulc:
            if x == "amplitude":
                ax.axvline(fu["amplitude_at_fulcrum"], color=INK2, lw=1, ls="--", zorder=1)
        if logx:
            ax.set_xscale("log")
            _plain_log(ax)
            ux = np.unique(allx)
            if len(ux) <= 8:
                ax.set_xticks(ux)
        units = {f.get("x_unit") for f in rows if f.get("x_unit")}
        ax.set_xlabel(x.replace("_", " ") + (f" ({units.pop()})" if len(units) == 1 else ""))
        if ax is axes[0][0]:
            ax.set_ylabel(_ylabel(var))
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{subject}: dose-response of {PS.label_of(var)}", color=INK, fontsize=11)
    fig.tight_layout()
    name = f"{_fname(subject)}_{var}_dose_response.png"
    _save(fig, figdir / name)
    who = (f"Pooled over the {subject.removeprefix('pooled-') or ''} subjects: dots are the mixed-model estimates per level (subject as a random effect)"
           if subject.startswith("pooled") else f"Subject {subject}: one dot per train")
    fl = (" The dashed grey line is the amplitude where the heart-rate response changes sign (neural fulcrum; see "
          "the notice for what it does and does not mean).") if fulc else ""
    return {"path": f"figures/{name}", "caption": (
        f"{who}, change in {PS.label_of(var)} during the train against the parameter that varies, one colour per "
        f"combination of the other parameters; lines are the model chosen by AIC (Hill, Boltzmann, linear, or none "
        f"when the response does not depend on the parameter); thin dashed lines join the level means where fewer than "
        f"MIN_LEVELS levels were tested and no curve was fitted. Dotted lines mark the threshold (lowest level whose "
        f"mean change exceeds the criterion in fits.csv), diamonds the EC50. Look for where the response starts, how "
        f"steeply it grows and whether it saturates.{fl}")}


# --------------------------------------------------------------------------- surface

def surface(figdir, subject, var, surf):
    groups = sorted({s["group"] for s in surf})
    g = max(groups, key=lambda k: sum(1 for s in surf if s["group"] == k))
    cells = [s for s in surf if s["group"] == g]
    amps = sorted({s["amplitude"] for s in cells})
    freqs = sorted({s["frequency_hz"] for s in cells})
    z = np.full((len(freqs), len(amps)), np.nan)
    for s in cells:
        z[freqs.index(s["frequency_hz"]), amps.index(s["amplitude"])] = s["mean_delta"]
    vmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    vmax = vmax or 1.0
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(amps), 0.9 + 0.6 * len(freqs)))
    im = ax.imshow(z, cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto", origin="lower")
    ax.grid(False)
    for i in range(len(freqs)):
        for j in range(len(amps)):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:.3g}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(amps)), [DR.fmt_level(a) for a in amps])
    ax.set_yticks(range(len(freqs)), [DR.fmt_level(f) for f in freqs])
    ax.set_xlabel("amplitude")
    ax.set_ylabel("frequency (Hz)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(_ylabel(var), color=INK2)
    ax.set_title(f"{subject}: {PS.label_of(var)} response surface" + (f"\n{g.replace(';', ', ')}" if g else ""))
    fig.tight_layout()
    name = f"{_fname(subject)}_{var}_surface.png"
    _save(fig, figdir / name)
    return {"path": f"figures/{name}", "caption": (
        f"{'Pooled over subjects' if subject.startswith('pooled') else 'Subject ' + str(subject)}: mean change in "
        f"{PS.label_of(var)} for every amplitude x frequency combination tested (blue: decrease, red: increase, "
        f"grey: no change; empty cells were not tested or had no usable train). Look for the region of the "
        f"parameter space that gives the largest response and where the response changes direction.")}


# --------------------------------------------------------------------------- raw traces

def raw_traces(figdir, rows, figvars, found, recs, input_dir, settings_used) -> list[dict]:
    out = []
    nwb_by_rid = {}
    for rid, r in recs.items():
        p = Path(r.get("_root", "")) / r.get("path", "")
        if r.get("path") and p.exists():
            nwb_by_rid[rid] = p
    for p in found.nwb:
        stem = p.stem
        nwb_by_rid.setdefault(stem, p)
    by_rec = defaultdict(list)
    for r in rows:
        if r["included"] and r["variable"] in figvars and r["delta"] is not None:
            by_rec[r["recording_id"]].append(r)
    for rid, rr in sorted(by_rec.items()):
        nwb = nwb_by_rid.get(rid)
        if nwb is None:
            continue
        var = next(v for v in figvars if any(r["variable"] == v for r in rr))
        best = max((r for r in rr if r["variable"] == var), key=lambda r: abs(r["delta"]))
        sw = settings_used.get(rid, {})
        t0 = best["onset"] - sw.get("baseline_s", 10)
        t1 = best["onset"] + (best["duration"] or 0) + sw.get("after_s", 10)
        try:
            types = NT.channel_types(Path(recs.get(rid, {}).get("_root", nwb.parent)), rid)
            chans = NT.read_window(nwb, t0, t1, types)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: raw traces not read (%s)", rid, e)
            continue
        if not chans:
            continue
        chans = chans[:4]
        fig, axes = plt.subplots(len(chans), 1, figsize=(8, 1.6 * len(chans) + 0.6), sharex=True, squeeze=False)
        for ax, c in zip(axes[:, 0], chans):
            ax.axvspan(0, best["duration"] or 0, color=TRAIN_SHADE, lw=0, zorder=0)
            ax.plot(c["t"] - best["onset"], c["y"], color=CATEGORICAL[0], lw=0.6)
            ax.set_ylabel(f"{c['name']}\n({c['unit']})", fontsize=8)
        axes[-1, 0].set_xlabel("time from train onset (s)")
        fig.suptitle(f"{rid}: {best['event_id']} ({PS.label_of(var)} change {best['delta']:.3g} {PS.unit_of(var)})",
                     color=INK, fontsize=10)
        fig.tight_layout()
        name = f"{_fname(rid)}_raw.png"
        _save(fig, figdir / name)
        out.append({"path": f"figures/{name}", "caption": (
            f"{rid}: the recorded signals around the train with the largest change in {PS.label_of(var)} "
            f"({best['event_id']}; grey band: the train). Check that the beats and pressure waves look clean in "
            f"the baseline and during the train, and that the stimulation artefact did not corrupt them.")})
    return out
