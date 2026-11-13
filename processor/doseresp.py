"""
Dose-response: which stimulation parameter varies, model fits, thresholds, EC50, response surfaces and
the neural fulcrum. In-house implementation of published, short methods (no licensed tool exists):

- Models (scipy.optimize.curve_fit with bounds, several starting points). The response is the change
  from baseline (delta), so the lower asymptote (bottom) is fixed at 0, the baseline. The Hill curve
  passes through 0 at x = 0; the Boltzmann curve only approaches 0 (at x = 0 it is top / (1 + e^(ec50/slope))):
    hill       y = top * x^n / (ec50^n + x^n)                 (x >= 0; slope = n)
    boltzmann  y = top / (1 + exp((ec50 - x) / slope))
    linear     y = a + b * x                                  (slope = b)
    none       y = c                                          (no dependence on x)
  chosen by AICc (Hurvich & Tsai 1989; AIC of Akaike 1974 with the small-sample correction):
  AICc = n ln(RSS / n) + 2K + 2K(K + 1) / (n - K - 1), K = fitted parameters + 1 for the residual
  variance. A model takes part only when n >= K + 2, i.e. at least 4 points for `none`, 5 for `linear`
  and 6 for a sigmoid; with one train per level and 4-5 levels the choice is between linear and none.
  Fewer than MIN_LEVELS distinct levels: `none` (level means and threshold only).
- Threshold: the lowest level of x whose mean response exceeds the criterion in absolute value; an
  upper bound (threshold_bound=upper) unless the next lower tested level was analysed and stayed below
  the criterion. The criterion (THRESHOLD_CRITERION=auto) is the larger of 5 % of the baseline (a
  convention, no published source) and 3 SD of the baseline
  variability: the train-to-train SD of the baseline means of the subject's included trains, or, with
  fewer than 3 such trains, the within-window SD of the baseline samples pooled over them
  (main.subject_criteria). That SD pools all of a subject's trains (all frequencies and other
  parameters) and also contains slow drift and any carry-over from the previous train.
- 95 % intervals: per subject, from the fit's covariance with t quantiles on the residual degrees of
  freedom (Hill EC50 on the log scale); pooled, from the mixed model's standard errors of the level
  estimates (weighted fit, normal quantiles). The fulcrum interval is a bootstrap of trains within
  levels (N_BOOT, a seed per fit derived from SEED), only when levels were repeated.
- Pooled fits (>= 3 subjects): the response at each level is estimated with a linear mixed model with
  the level as a fixed effect and subject as a random intercept (statsmodels MixedLM), and the models
  are fitted to those estimates weighted by their standard errors; n is then the number of levels.
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
    aic: float                      # AICc
    r2: float | None
    note: str = ""
    cov: np.ndarray | None = None   # parameter covariance (residual-scaled, or from sigma when absolute)
    df: int = 0                     # residual degrees of freedom
    absolute: bool = False          # cov from known standard errors (pooled fits) rather than residuals


def aicc(n: int, rss: float, k: int) -> float | None:
    """Corrected AIC (Hurvich & Tsai 1989) with K = k + 1 parameters (k fitted + the residual variance);
    None when n - K - 1 < 1 (the correction is undefined)."""
    kk = k + 1
    if n - kk - 1 < 1:
        return None
    rss = max(rss, 1e-12 * max(1, n))
    return n * math.log(rss / n) + 2 * kk + 2 * kk * (kk + 1) / (n - kk - 1)


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


def fit_model(model: str, x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None) -> Fit | None:
    """Least-squares fit (weighted by 1/sigma^2 when sigma is given: pooled fits, where sigma are the
    mixed model's standard errors and the covariance is taken as absolute). None when AICc is undefined
    for this n (n < k + 3)."""
    n = len(x)
    k = N_PARAMS[model]
    if n < k + 3:
        return None
    w = np.ones(n) if sigma is None else 1.0 / np.asarray(sigma, float) ** 2
    absolute = sigma is not None
    ybar = float(np.sum(w * y) / np.sum(w))
    tss = float(np.sum(w * (y - ybar) ** 2))
    df = n - k
    if model in ("none", "linear"):
        X = np.ones((n, 1)) if model == "none" else np.column_stack([np.ones(n), x])
        W = np.diag(w)
        xtwx_inv = np.linalg.pinv(X.T @ W @ X)
        beta = xtwx_inv @ X.T @ W @ y
        rss = float(np.sum(w * (y - X @ beta) ** 2))
        cov = xtwx_inv if absolute else xtwx_inv * (rss / df)
        r2 = None if model == "none" else ((1 - rss / tss) if tss > 0 else None)
        return Fit(model, tuple(float(b) for b in beta), rss, aicc(n, rss, k), r2, cov=cov, df=df, absolute=absolute)
    lo, hi = _bounds(model, x, y)
    best = None
    for start in _starts(model, x, y):
        s0 = np.clip(np.asarray(start, float), np.asarray(lo) + 1e-12, np.asarray(hi) - 1e-12)
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", OptimizeWarning)
                _warnings.simplefilter("ignore", RuntimeWarning)
                p, pcov = curve_fit(FUNCS[model], x, y, p0=s0, bounds=(lo, hi), maxfev=4000,
                                    sigma=sigma, absolute_sigma=absolute)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum(w * (y - FUNCS[model](x, *p)) ** 2))
        if np.isfinite(rss) and (best is None or rss < best[1]):
            best = (tuple(float(v) for v in p), rss, pcov)
    if best is None:
        return None
    p, rss, pcov = best
    note = ""
    if abs(p[1] - lo[1]) <= 1e-6 * max(1, abs(lo[1])) or abs(p[1] - hi[1]) <= 1e-6 * max(1, abs(hi[1])):
        note = "EC50 at the fit bound: not determined"
    return Fit(model, p, rss, aicc(n, rss, k), (1 - rss / tss) if tss > 0 else None, note, cov=pcov, df=df,
               absolute=absolute)


def choose(x: np.ndarray, y: np.ndarray, models, min_levels: int, sigma=None) -> tuple[Fit | None, dict, str]:
    """Best model by AICc. Returns (fit, {model: aicc}, note)."""
    n_levels = len(np.unique(x))
    aics = {}
    if n_levels < min_levels:
        f = fit_model("none", x, y, sigma)
        return f, ({"none": f.aic} if f else {}), f"{n_levels} level(s) < MIN_LEVELS {min_levels}: no curve fitted"
    fits = []
    for m in models:
        f = fit_model(m, x, y, sigma)
        if f is not None:
            aics[m] = f.aic
            fits.append(f)
    if not fits:
        return None, aics, f"too few points ({len(x)}) for AICc with any model"
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


def intervals(fit: Fit | None, x: np.ndarray) -> dict:
    """95 % intervals of ec50, top and slope from the fit's covariance: t quantiles with the residual
    degrees of freedom (per subject), normal quantiles when the covariance comes from known standard
    errors (pooled). EC50 of the Hill model on the log scale (it is positive). None below 2 residual
    degrees of freedom or when the covariance is not finite."""
    from scipy.stats import norm, t as t_dist  # noqa: PLC0415

    if fit is None or fit.model == "none" or fit.cov is None:
        return {}
    cov = np.asarray(fit.cov, float)
    if not np.all(np.isfinite(cov)) or (not fit.absolute and fit.df < 2):
        return {}
    q = float(norm.ppf(0.975)) if fit.absolute else float(t_dist.ppf(0.975, fit.df))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    p = fit.params
    out = {}
    if fit.model in ("hill", "boltzmann"):
        out["top_ci_low"], out["top_ci_high"] = p[0] - q * se[0], p[0] + q * se[0]
        out["slope_ci_low"], out["slope_ci_high"] = p[2] - q * se[2], p[2] + q * se[2]
        if not fit.note:
            if fit.model == "hill" and p[1] > 0:
                f = math.exp(q * se[1] / p[1])
                out["ec50_ci_low"], out["ec50_ci_high"] = p[1] / f, p[1] * f
            else:
                out["ec50_ci_low"], out["ec50_ci_high"] = p[1] - q * se[1], p[1] + q * se[1]
    elif fit.model == "linear":
        out["slope_ci_low"], out["slope_ci_high"] = p[1] - q * se[1], p[1] + q * se[1]
        g = np.array([1.0, float(np.max(x))])
        se_top = math.sqrt(max(float(g @ cov @ g), 0.0))
        top = p[0] + p[1] * g[1]
        out["top_ci_low"], out["top_ci_high"] = top - q * se_top, top + q * se_top
    return {k: float(v) for k, v in out.items()}


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


def threshold(ux: np.ndarray, means: np.ndarray, crit: float | None, tested=None) -> tuple[float | None, str, str]:
    """(threshold, direction or note, bound). bound is `bracketed` when the next lower tested level was
    analysed and stayed below the criterion, else `upper` (the true threshold may be lower: the level
    below was not tested, or its trains were all left out)."""
    if crit is None:
        return None, "", ""
    above = np.where(np.abs(means) >= crit)[0]
    if not len(above):
        return None, "no level exceeded the criterion", ""
    i = int(above[0])
    thr = float(ux[i])
    tested = sorted(set(float(t) for t in (tested if tested is not None else ux)))
    lower = [t for t in tested if t < thr - 1e-9]
    bound = "bracketed" if lower and any(abs(lower[-1] - u) <= 1e-9 for u in ux) else "upper"
    return thr, ("increase" if means[i] > 0 else "decrease"), bound


# --------------------------------------------------------------------------- pooled (mixed model)

def mixed_level_estimates(x: np.ndarray, y: np.ndarray, subjects: np.ndarray):
    """Response per level and its standard error, with subject as a random intercept (statsmodels
    MixedLM, REML); falls back to the mean of the subjects' level means (SE = SD / sqrt(n subjects))
    when the model cannot be fitted. Returns (levels, estimates, standard errors, note)."""
    ux = np.unique(x)
    exog = (x[:, None] == ux[None, :]).astype(float)
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM  # noqa: PLC0415

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            res = MixedLM(y, exog, groups=subjects).fit(reml=True)
        est = np.asarray(res.fe_params, float)
        se = np.asarray(res.bse_fe, float)
        if est.shape == ux.shape and np.all(np.isfinite(est)) and np.all(np.isfinite(se)) and np.all(se > 0):
            re_var = float(np.asarray(res.cov_re).ravel()[0])
            return ux, est, se, f"MixedLM, subject random intercept (variance {re_var:.3g})"
    except Exception as e:  # noqa: BLE001 - singular fits, convergence
        note = f"MixedLM failed ({type(e).__name__}); mean of subject means"
    else:
        note = "MixedLM gave no estimate; mean of subject means"
    est, se = [], []
    for u in ux:
        per = [float(np.mean(y[(x == u) & (subjects == s)])) for s in np.unique(subjects[x == u])]
        est.append(float(np.mean(per)))
        se.append(float(np.std(per, ddof=1) / np.sqrt(len(per))) if len(per) > 1 else np.nan)
    se = np.array(se)
    if not np.all(np.isfinite(se) & (se > 0)):
        se = None
    return ux, np.array(est), se, note


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
