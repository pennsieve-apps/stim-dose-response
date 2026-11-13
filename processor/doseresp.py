"""
Dose-response: which stimulation parameter varies, model fits, thresholds, EC50, response surfaces and
the neural fulcrum. In-house implementation of published, short methods (no licensed tool exists):

- Models (scipy.optimize.curve_fit with bounds, several starting points). The response is the change
  from baseline (delta), so the curves start at 0 (bottom fixed at 0, the baseline):
    hill       y = top * x^n / (ec50^n + x^n)                 (x >= 0; slope = n)
    boltzmann  y = top / (1 + exp((ec50 - x) / slope))
    linear     y = a + b * x                                  (slope = b)
    none       y = c                                          (no dependence on x)
  chosen by AIC = n ln(RSS / n) + 2 k (k fitted parameters + 1 for the residual variance), among the
  models with at least 2 residual degrees of freedom, when x has at least MIN_LEVELS distinct levels;
  else `none`.
- Threshold: the lowest level of x whose mean response exceeds the criterion in absolute value; the
  criterion (THRESHOLD_CRITERION=auto) is the larger of 5 % of the baseline and 3 SD of the baseline
  variability (the within-window SD of the baseline samples, pooled over the subject's trains).
- Bootstrap CIs (N_BOOT, SEED): per subject, residual bootstrap of the chosen model (trains are rarely
  repeated at every level, so resampling trains within a level would not vary); pooled, a bootstrap of
  subjects.
- Pooled fits (>= 3 subjects): the response at each level is estimated with a linear mixed model with
  the level as a fixed effect and subject as a random intercept (statsmodels MixedLM), and the models
  are fitted to those estimates.
- Neural fulcrum (Ardell et al. 2017): for heart-rate variables, at a fixed frequency, the amplitude
  where the response changes sign (tachycardia to bradycardia or back), found by linear interpolation
  between the two bracketing level means, when one level exceeds the criterion in each direction.
"""
from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

LEVEL_TOLERANCE = 0.02          # numeric parameter values within 2 % are one level (3.7819 vs 3.7826)
NUMERIC_PARAMS = ("amplitude", "frequency_hz", "pulse_width_us", "train_duration_s", "train_on_s", "train_off_s")
CATEGORICAL_PARAMS = ("amplitude_unit", "polarity", "waveform", "site", "contact", "condition")
NOT_PARAMS = {"label", "source", "trial_group", "n_pulses", "event_id", "recording_id", "onset", "duration",
              "trial_type"}
ALL_MODELS = ("hill", "boltzmann", "linear", "none")
N_PARAMS = {"hill": 3, "boltzmann": 3, "linear": 2, "none": 1}


# --------------------------------------------------------------------------- models

def f_hill(x, top, ec50, n):
    x = np.clip(np.asarray(x, float), 0, None)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r = np.power(x / ec50, n)
        return top * r / (1.0 + r)


def f_boltzmann(x, top, ec50, slope):
    with np.errstate(over="ignore"):
        return top / (1.0 + np.exp(np.clip((ec50 - np.asarray(x, float)) / slope, -500, 500)))


def f_linear(x, a, b):
    return a + b * np.asarray(x, float)


def f_none(x, c):
    return np.full(np.shape(x), c, float)


FUNCS = {"hill": f_hill, "boltzmann": f_boltzmann, "linear": f_linear, "none": f_none}


@dataclass
class Fit:
    model: str
    params: tuple
    rss: float
    aic: float
    r2: float | None
    note: str = ""


def _aic(n: int, rss: float, k: int) -> float:
    rss = max(rss, 1e-12 * max(1, n))
    return n * math.log(rss / n) + 2 * (k + 1)


def _bounds(model: str, x: np.ndarray, y: np.ndarray):
    ymax = float(np.max(np.abs(y))) + 1e-9
    xmin, xmax = float(np.min(x)), float(np.max(x))
    rng = max(xmax - xmin, 1e-9)
    if model == "hill":
        pos = x[x > 0]
        lo = float(np.min(pos)) / 10 if len(pos) else 1e-6
        return ([-10 * ymax, lo, 0.5], [10 * ymax, xmax * 10, 8.0])
    if model == "boltzmann":
        return ([-10 * ymax, xmin - rng, rng / 200], [10 * ymax, xmax + rng, rng * 2])
    return (-np.inf, np.inf)


def _starts(model: str, x: np.ndarray, y: np.ndarray):
    ux = np.unique(x)
    order = np.argsort(x)
    y_hi = float(np.mean(y[order][-max(1, len(y) // 4):]))
    top0 = y_hi * 1.2 if abs(y_hi) > 1e-9 else (float(np.max(np.abs(y))) or 1.0)
    if model == "hill":
        pos = ux[ux > 0]
        cands = np.geomspace(pos.min(), pos.max(), 4) if len(pos) > 1 else (pos if len(pos) else [1.0])
        return [(top0, float(c), n) for c in cands for n in (1.0, 3.0)]
    if model == "boltzmann":
        rng = max(float(np.ptp(x)), 1e-9)
        cands = np.linspace(x.min(), x.max(), 4)
        return [(top0, float(c), s) for c in cands for s in (rng / 10, rng / 3)]
    return [None]


def fit_model(model: str, x: np.ndarray, y: np.ndarray, p0=None) -> Fit | None:
    n = len(x)
    k = N_PARAMS[model]
    if n < k + 2:
        return None
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if model == "none":
        c = float(np.mean(y))
        rss = tss
        return Fit(model, (c,), rss, _aic(n, rss, k), None)
    if model == "linear":
        b, a = np.polyfit(x, y, 1)
        rss = float(np.sum((y - f_linear(x, a, b)) ** 2))
        return Fit(model, (float(a), float(b)), rss, _aic(n, rss, k), (1 - rss / tss) if tss > 0 else None)
    lo, hi = _bounds(model, x, y)
    best = None
    for start in ([p0] if p0 is not None else _starts(model, x, y)):
        s = np.clip(np.asarray(start, float), np.asarray(lo) + 1e-12, np.asarray(hi) - 1e-12)
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", OptimizeWarning)
                _warnings.simplefilter("ignore", RuntimeWarning)
                p, _ = curve_fit(FUNCS[model], x, y, p0=s, bounds=(lo, hi), maxfev=4000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum((y - FUNCS[model](x, *p)) ** 2))
        if np.isfinite(rss) and (best is None or rss < best[1]):
            best = (tuple(float(v) for v in p), rss)
    if best is None:
        return None
    p, rss = best
    note = ""
    if model in ("hill", "boltzmann"):
        if abs(p[1] - lo[1]) <= 1e-6 * max(1, abs(lo[1])) or abs(p[1] - hi[1]) <= 1e-6 * max(1, abs(hi[1])):
            note = "EC50 at the fit bound: not determined"
    return Fit(model, p, rss, _aic(n, rss, k), (1 - rss / tss) if tss > 0 else None, note)


def choose(x: np.ndarray, y: np.ndarray, models, min_levels: int) -> tuple[Fit | None, dict, str]:
    """Best model by AIC. Returns (fit, {model: aic}, note)."""
    n_levels = len(np.unique(x))
    aics = {}
    if n_levels < min_levels:
        f = fit_model("none", x, y)
        return f, ({"none": f.aic} if f else {}), f"{n_levels} level(s) < MIN_LEVELS {min_levels}: no curve fitted"
    fits = []
    for m in models:
        f = fit_model(m, x, y)
        if f is not None:
            aics[m] = f.aic
            fits.append(f)
    if not fits:
        return None, aics, "too few trains for any model"
    best = min(fits, key=lambda f: f.aic)
    return best, aics, best.note


def describe(fit: Fit | None, x: np.ndarray) -> dict:
    """ec50, top, bottom, slope of a fit, in the fits.csv sense."""
    d = {"ec50": None, "top": None, "bottom": None, "slope": None}
    if fit is None:
        return d
    p = fit.params
    if fit.model in ("hill", "boltzmann"):
        d.update(top=p[0], bottom=0.0, slope=p[2], ec50=None if fit.note else p[1])
    elif fit.model == "linear":
        lo, hi = float(np.min(x)), float(np.max(x))
        d.update(slope=p[1], bottom=float(f_linear(lo, *p)), top=float(f_linear(hi, *p)))
    elif fit.model == "none":
        d.update(top=p[0], bottom=p[0])
    return d


def bootstrap_residual(fit: Fit, x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    """Percentile 95 % CIs of ec50, top and slope by resampling the residuals of the chosen model."""
    if fit is None or fit.model == "none" or n_boot <= 0:
        return {}
    yhat = FUNCS[fit.model](x, *fit.params)
    res = y - yhat
    vals = {"ec50": [], "top": [], "slope": []}
    for _ in range(n_boot):
        yb = yhat + rng.choice(res, size=len(res), replace=True)
        fb = fit_model(fit.model, x, yb, p0=fit.params) if fit.model != "linear" else fit_model("linear", x, yb)
        if fb is None:
            continue
        d = describe(fb, x)
        for k in vals:
            if d[k] is not None:
                vals[k].append(d[k])
    return ci(vals, n_boot)


def ci(vals: dict, n_boot: int) -> dict:
    out = {}
    for k, v in vals.items():
        if len(v) >= max(20, n_boot // 2):
            out[f"{k}_ci_low"], out[f"{k}_ci_high"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


# --------------------------------------------------------------------------- design: what varies

def cluster_levels(values: list[float]) -> dict[float, float]:
    """Map each value to its level (the median of values within LEVEL_TOLERANCE of each other)."""
    u = sorted(set(values))
    groups: list[list[float]] = []
    for v in u:
        if groups and abs(v - groups[-1][-1]) <= LEVEL_TOLERANCE * max(abs(v), abs(groups[-1][-1]), 1e-12):
            groups[-1].append(v)
        else:
            groups.append([v])
    out = {}
    for g in groups:
        lv = float(np.median(g))
        lv = float(f"{lv:.4g}")
        for v in g:
            out[v] = lv
    return out


def design(events: list[dict], to_float) -> tuple[list[str], list[str], list[dict]]:
    """For a list of event param dicts: the numeric parameters (candidate x), the varying parameters
    (for group labels), and each event's levels {param: level}."""
    keys = set()
    for e in events:
        keys.update(k for k in e if k not in NOT_PARAMS and not k.startswith("_"))
    numeric, categorical = [], []
    for k in sorted(keys):
        vals = [e.get(k) for e in events]
        nums = [to_float(v) for v in vals]
        present = [v for v in vals if to_float(v) is not None or (v not in (None, "") and str(v).lower() != "n/a")]
        if not present:
            continue
        if k in NUMERIC_PARAMS or (all(n is not None for n in nums) and k not in CATEGORICAL_PARAMS):
            numeric.append(k)
        else:
            distinct = {str(v) for v in vals}
            if k not in CATEGORICAL_PARAMS and len(distinct) == len(events) and len(events) > 2:
                continue        # unique per event (a time stamp, a free-text note): not a parameter
            categorical.append(k)
    levels = [dict() for _ in events]
    for k in numeric:
        raw = [to_float(e.get(k)) for e in events]
        cl = cluster_levels([v for v in raw if v is not None])
        for i, v in enumerate(raw):
            levels[i][k] = cl[v] if v is not None else None
    for k in categorical:
        for i, e in enumerate(events):
            v = e.get(k)
            levels[i][k] = None if (v in (None, "") or str(v).lower() == "n/a") else str(v)
    varying = [k for k in numeric + categorical if len({lv[k] for lv in levels}) > 1]
    return numeric, varying, levels


def fmt_level(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def group_label(levels: dict, keys: list[str]) -> str:
    return ";".join(f"{k}={fmt_level(levels.get(k))}" for k in keys)


# --------------------------------------------------------------------------- thresholds

def criterion(spec: str, baseline_mean: float | None, baseline_sd: float | None, unit: str) -> tuple[float | None, str]:
    """The threshold criterion in the variable's unit, and how it was decided."""
    pct_val = abs(baseline_mean) * 0.05 if baseline_mean is not None else None
    sd_val = 3 * baseline_sd if baseline_sd is not None else None
    kind, _, arg = spec.partition(":")
    if kind == "auto":
        cands = [(v, d) for v, d in ((pct_val, f"5% of baseline ({_g(pct_val)} {unit})"),
                                     (sd_val, f"3 SD of baseline ({_g(sd_val)} {unit})")) if v is not None]
        if not cands:
            return None, "no baseline"
        v, d = max(cands, key=lambda c: c[0])
        other = [c for c in cands if c[1] != d]
        return v, f"auto: {d}" + (f" > {other[0][1]}" if other else "")
    a = float(arg)
    if kind == "pct":
        return (abs(baseline_mean) * a / 100 if baseline_mean is not None else None), f"{a:g}% of baseline"
    if kind == "sd":
        return (a * baseline_sd if baseline_sd is not None else None), f"{a:g} SD of baseline"
    if kind == "abs":
        return a, f"{a:g} {unit}"
    raise ValueError(spec)


def _g(v):
    return "n/a" if v is None else f"{v:.3g}"


def level_means(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ux = np.unique(x)
    m = np.array([float(np.mean(y[x == u])) for u in ux])
    n = np.array([int(np.sum(x == u)) for u in ux])
    return ux, m, n


def threshold(ux: np.ndarray, means: np.ndarray, crit: float | None) -> tuple[float | None, str]:
    if crit is None:
        return None, ""
    above = np.where(np.abs(means) >= crit)[0]
    if not len(above):
        return None, "no level exceeded the criterion"
    i = int(above[0])
    return float(ux[i]), ("increase" if means[i] > 0 else "decrease")


# --------------------------------------------------------------------------- pooled (mixed model)

def mixed_level_estimates(x: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Response per level with subject as a random intercept (statsmodels MixedLM, REML); falls back to
    the mean of the subjects' level means when the model cannot be fitted."""
    ux = np.unique(x)
    exog = (x[:, None] == ux[None, :]).astype(float)
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM  # noqa: PLC0415

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            res = MixedLM(y, exog, groups=subjects).fit(reml=True)
        est = np.asarray(res.fe_params, float)
        if est.shape == ux.shape and np.all(np.isfinite(est)):
            re_var = float(np.asarray(res.cov_re).ravel()[0])
            return ux, est, f"MixedLM, subject random intercept (variance {re_var:.3g})"
    except Exception as e:  # noqa: BLE001 - singular fits, convergence
        note = f"MixedLM failed ({type(e).__name__}); mean of subject means"
    else:
        note = "MixedLM gave no estimate; mean of subject means"
    est = []
    for u in ux:
        per = [float(np.mean(y[(x == u) & (subjects == s)])) for s in np.unique(subjects[x == u])]
        est.append(float(np.mean(per)))
    return ux, np.array(est), note


def subject_level_means(x, y, subjects):
    """{subject: (levels, means)}"""
    out = {}
    for s in np.unique(subjects):
        m = subjects == s
        out[s] = level_means(x[m], y[m])[:2]
    return out


def bootstrap_subjects(fit: Fit, per_subject: dict, n_boot: int, rng: np.random.Generator) -> dict:
    """Pooled CIs: resample subjects with replacement, average their level means, refit the chosen model."""
    if fit is None or fit.model == "none" or n_boot <= 0:
        return {}
    subs = list(per_subject)
    vals = {"ec50": [], "top": [], "slope": []}
    for _ in range(n_boot):
        pick = rng.choice(len(subs), size=len(subs), replace=True)
        xs = np.concatenate([per_subject[subs[i]][0] for i in pick])
        ys = np.concatenate([per_subject[subs[i]][1] for i in pick])
        ux, m, _ = level_means(xs, ys)
        fb = fit_model(fit.model, ux, m, p0=fit.params if fit.model != "linear" else None)
        if fb is None:
            continue
        d = describe(fb, ux)
        for k in vals:
            if d[k] is not None:
                vals[k].append(d[k])
    return ci(vals, n_boot)


# --------------------------------------------------------------------------- neural fulcrum

def fulcrum(ux: np.ndarray, means: np.ndarray, crit: float | None) -> tuple[float | None, str]:
    """Amplitude where the response changes sign: from the first level whose mean exceeds the criterion in
    one direction to the first later level that exceeds it in the other direction, the last sign change of
    the level means between them, by linear interpolation between the two bracketing levels. Sign flips of
    responses smaller than the criterion alone do not count."""
    if crit is None or len(ux) < 2:
        return None, ""
    sig = [i for i in range(len(ux)) if abs(means[i]) >= crit]
    if not sig:
        return None, ""
    i0 = sig[0]
    s0 = np.sign(means[i0])
    later = [j for j in sig if j > i0 and np.sign(means[j]) == -s0]
    if not later:
        return None, ""
    j0 = later[0]
    k = max(k for k in range(i0, j0) if np.sign(means[k]) == s0 and np.sign(means[k + 1]) != s0)
    x0, x1, y0, y1 = ux[k], ux[k + 1], means[k], means[k + 1]
    xf = float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    note = "tachycardia to bradycardia" if s0 > 0 else "bradycardia to tachycardia"
    flips = sum(1 for a, b in zip(sig, sig[1:]) if np.sign(means[a]) != np.sign(means[b]))
    if flips > 1:
        note += f"; the response changes direction {flips} times, the first change is reported"
    return xf, note


def bootstrap_fulcrum(x: np.ndarray, y: np.ndarray, crit: float, n_boot: int, rng: np.random.Generator) -> dict:
    """Resample trains within each level; only meaningful when levels have repeated trains."""
    ux = np.unique(x)
    if n_boot <= 0 or all(np.sum(x == u) < 2 for u in ux):
        return {}
    vals = []
    for _ in range(n_boot):
        m = np.array([float(np.mean(rng.choice(y[x == u], size=int(np.sum(x == u)), replace=True))) for u in ux])
        xf, _ = fulcrum(ux, m, crit)
        if xf is not None:
            vals.append(xf)
    if len(vals) < max(20, n_boot // 2):
        return {}
    return {"ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))}
