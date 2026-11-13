"""The synthetic fixture through the processor (no Docker): the planted thresholds, EC50s and fulcrum are
recovered, species presets apply, and missing inputs take the documented paths."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from processor import doseresp as DR
from processor import main as M
from processor import responses as RS
from tests import make_fixture


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(monkeypatch, input_dir: Path, output_dir: Path, **env) -> int:
    for k in M.DEFAULTS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("N_BOOT", "100")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session")
def fixture(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("fixture")
    make_fixture.main(d)
    return d


@pytest.fixture(scope="session")
def truth(fixture) -> dict:
    return json.loads((fixture / "truth.json").read_text())


@pytest.fixture(scope="session")
def out(fixture, tmp_path_factory) -> Path:
    o = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture, o) == 0
    finally:
        mp.undo()
    return o / "stim-dose-response"


def fit(out, subject, variable, x="amplitude", group=None):
    hits = [r for r in rows(out / "fits.csv") if r["subject"] == subject and r["variable"] == variable
            and r["x"] == x and (group is None or r["group"] == group)]
    assert len(hits) == 1, (subject, variable, x, group, [h["group"] for h in hits])
    return hits[0]


def test_outputs_exist(out):
    for f in ("responses.csv", "fits.csv", "surface.csv", "fulcrum.csv", "features.csv", "summary.json",
              "provenance.json"):
        assert (out / f).exists(), f
    s = json.loads((out / "summary.json").read_text())
    assert s["app"] == "stim-dose-response"
    assert s["figures"] and all((out / f["path"]).exists() and f["caption"] for f in s["figures"])
    assert "Research outputs" in s["notice"] and "Ardell" in s["notice"]
    p = json.loads((out / "provenance.json").read_text())
    assert set(p) >= {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                      "wall_seconds", "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"}
    assert M.DOI_FULCRUM in p["citations"] and M.DOI_STATSMODELS in p["citations"]


@pytest.mark.parametrize("subject", ["p1", "p3", "p4"])
@pytest.mark.parametrize("freq", ["20", "5"])
def test_planted_threshold_and_ec50(out, truth, subject, freq):
    t = truth["subjects"][subject]
    f = fit(out, subject, "hr_bpm", group=f"frequency_hz={freq}")
    assert float(f["threshold"]) == t["threshold_hr_uA"][freq]
    assert f["direction"] == "decrease"
    assert f["model"] in ("hill", "boltzmann")
    assert abs(float(f["ec50"]) / t["ec50_hr_uA"] - 1) < 0.15
    assert float(f["ec50_ci_low"]) <= float(f["ec50"]) <= float(f["ec50_ci_high"])
    assert "5% of baseline" in f["threshold_criterion"]
    if freq == "20":
        # top of the fitted curve = planted plateau (-30 bpm) x the train-mean factor
        assert float(f["top"]) == pytest.approx(-30 * truth["train_mean_factor"], rel=0.1)
        assert float(f["r2"]) > 0.98


def test_linear_map(out):
    f = fit(out, "p1", "map_mmhg", group="frequency_hz=20")
    assert f["model"] == "linear" or "outside the tested range" in f["note"]
    assert float(f["threshold"]) == 800      # -0.008 * 800 * 0.80 = -5.1 >= 4.5 mmHg (5 % of 90)


def test_rat_preset(out, truth):
    p = json.loads((out / "provenance.json").read_text())
    w = p["windows"]["sub-r1_rec-sweep"]
    assert (w["species"], w["baseline_s"], w["after_s"]) == ("rat", 5.0, 5.0)
    assert p["windows"]["sub-p1_rec-grid"]["baseline_s"] == 10.0
    f = fit(out, "r1", "hr_bpm")
    assert float(f["threshold"]) == truth["subjects"]["r1"]["threshold_hr_uA"]["20"]
    assert abs(float(f["ec50"]) / 300 - 1) < 0.15


def test_pooled(out):
    f = fit(out, "pooled", "hr_bpm", group="frequency_hz=20")
    assert f["species"] == "pig" and int(f["n_subjects"]) == 3 and f["recording_id"] == "pooled"
    assert "MixedLM" in f["note"]
    assert abs(float(f["ec50"]) / 600 - 1) < 0.15
    assert float(f["threshold"]) == 400
    assert not any(r["subject"] == "pooled" and r["species"] == "rat" for r in rows(out / "fits.csv"))


def test_fulcrum(out, truth):
    fu = [r for r in rows(out / "fulcrum.csv") if r["subject"] == "p2"]
    assert len(fu) == 1
    a = float(fu[0]["amplitude_at_fulcrum"])
    lo, hi = truth["subjects"]["p2"]["fulcrum_bracket"]
    assert lo < a < hi
    assert abs(a / truth["subjects"]["p2"]["fulcrum_hr_uA"] - 1) < 0.05
    assert fu[0]["frequency_hz"] == "10" and fu[0]["ci_low"] and "tachycardia to bradycardia" in fu[0]["note"]
    # no fulcrum where the response never changes sign
    assert not any(r["subject"] in ("p1", "p3", "p4", "r1") for r in rows(out / "fulcrum.csv"))


def test_surface(out):
    s = [r for r in rows(out / "surface.csv") if r["subject"] == "p1" and r["variable"] == "hr_bpm"]
    assert len(s) == 12
    cell = next(r for r in s if r["amplitude"] == "3200" and r["frequency_hz"] == "20")
    assert float(cell["mean_delta"]) == pytest.approx(-30 * 3200 ** 2 / (600 ** 2 + 3200 ** 2) * 0.8013, abs=1.5)


def test_responses(out, truth):
    r = rows(out / "responses.csv")
    assert not any(x["event_id"] == "mk-0001" for x in r)            # physio-import's markers are not read
    assert {x["recording_id"] for x in r} == {"sub-p1_rec-grid", "sub-p2_rec-fulcrum", "sub-p3_rec-grid",
                                              "sub-p4_rec-grid", "sub-r1_rec-sweep"}
    p2 = {x["variable"] for x in r if x["recording_id"] == "sub-p2_rec-fulcrum"}
    assert p2 == {"hr_bpm", "rr_ms"}                                   # no BP for this recording: not a failure
    big = [x for x in r if x["recording_id"] == "sub-p1_rec-grid" and x["variable"] == "hr_bpm"
           and x["amplitude"] == "3200" and x["frequency_hz"] == "20"][0]
    assert float(big["baseline"]) == pytest.approx(80, abs=1)
    assert float(big["peak_delta"]) < float(big["delta"]) < 0
    assert 3 < float(big["time_to_peak_s"]) < 7
    assert big["recovery_s"] and 0 < float(big["recovery_s"]) < 15
    assert big["included"] == "true" and float(big["valid_fraction"]) > 0.95
    assert big["amplitude_unit"] == "uA" and big["pulse_width_us"] == "500"
    p = json.loads((out / "provenance.json").read_text())
    assert p["recordings"]["sub-h1_rec-rest"]["status"] == "skipped"


def test_overlap_rules():
    s = RS.Settings(10, 10, "train", None, 2, 0.1, 0.8, "exclude")
    a = RS.Event("r", "a", 100, 105, True, "stimulation")
    b = RS.Event("r", "b", 112, 117, True, "stimulation")       # starts 7 s after a ends
    w = RS.event_windows(b, [a], s)
    assert w["excluded"] and "baseline overlaps a" in w["reasons"][0]
    s.overlap = "clip"
    w = RS.event_windows(b, [a], s)
    assert not w["excluded"] and w["b0"] == 105
    c = RS.Event("r", "c", 108, 113, True, "stimulation")       # 3 s of baseline left
    w = RS.event_windows(c, [a], s)
    assert w["excluded"]


def test_model_choice():
    rng = np.random.default_rng(0)
    x = np.repeat([1, 2, 4, 8, 16, 32.0], 2)
    y = DR.f_hill(x, -20, 6, 2) + rng.normal(0, 0.3, len(x))
    best, aics, _ = DR.choose(x, y, DR.ALL_MODELS, 4)
    assert best.model in ("hill", "boltzmann") and abs(best.params[1] / 6 - 1) < 0.2
    flat = rng.normal(0, 1, len(x))
    best, _, _ = DR.choose(x, flat, DR.ALL_MODELS, 4)
    assert best.model in ("none", "linear")
    best, _, note = DR.choose(np.array([1, 2, 3.0]), np.array([1, 2, 3.0]), DR.ALL_MODELS, 4)
    assert best.model == "none" and "MIN_LEVELS" in note
    lv = DR.cluster_levels([3.7819, 3.7826, 15.131, 30.13])
    assert lv[3.7819] == lv[3.7826] and lv[15.131] != lv[30.13]


def test_criterion():
    v, d = DR.criterion("auto", 80, 0.5, "bpm")
    assert v == pytest.approx(4) and d.startswith("auto: 5% of baseline")
    v, d = DR.criterion("auto", 80, 2, "bpm")
    assert v == pytest.approx(6) and d.startswith("auto: 3 SD")
    assert DR.criterion("abs:2", 80, 2, "bpm")[0] == 2
    assert DR.criterion("pct:10", 80, 2, "bpm")[0] == pytest.approx(8)


def test_no_series(monkeypatch, fixture, tmp_path):
    d = tmp_path / "in"
    shutil.copytree(fixture / "events", d / "events")
    assert run(monkeypatch, d, tmp_path / "o") == 1


def test_no_events(monkeypatch, fixture, tmp_path):
    d = tmp_path / "in"
    shutil.copytree(fixture / "hrv", d / "hrv")
    shutil.copytree(fixture / "physio", d / "physio")     # physio-import's events.tsv is not stim-events'
    assert run(monkeypatch, d, tmp_path / "o") == 1


def test_resting_only(monkeypatch, fixture, tmp_path):
    """stim-events found no stimulation (resting recordings): exit 0, header-only tables, a warning."""
    d = tmp_path / "in"
    shutil.copytree(fixture / "hrv", d / "hrv")
    (d / "events" / "stim-events").mkdir(parents=True)
    (d / "events" / "stim-events" / "events.tsv").write_text(
        "\t".join(make_fixture.EVENT_COLS) + "\n")
    assert run(monkeypatch, d, tmp_path / "o") == 0
    o = tmp_path / "o" / "stim-dose-response"
    assert rows(o / "responses.csv") == [] and rows(o / "fits.csv") == []
    assert json.loads((o / "summary.json").read_text())["warnings"]


def test_merged_root_and_clip(monkeypatch, fixture, tmp_path):
    """Everything side by side at the root of INPUT_DIR; OVERLAP=clip and RESPONSE_WINDOW=peak run."""
    d = tmp_path / "in"
    for port in ("events", "hrv", "bp", "physio"):
        for p in (fixture / port).iterdir():
            shutil.copytree(p, d / p.name) if p.is_dir() else shutil.copy(p, d / p.name)
    assert run(monkeypatch, d, tmp_path / "o", OVERLAP="clip", RESPONSE_WINDOW="peak", FIGURES="false") == 0
    o = tmp_path / "o" / "stim-dose-response"
    f = [r for r in rows(o / "fits.csv") if r["subject"] == "p1" and r["variable"] == "hr_bpm"
         and r["group"] == "frequency_hz=20" and r["x"] == "amplitude"][0]
    assert float(f["threshold"]) == 400


@pytest.mark.parametrize("env", [{"RESPONSE_WINDOW": "fixed:5,2"}, {"SPECIES": "cat"}, {"MODELS": "spline"},
                                 {"THRESHOLD_CRITERION": "pct"}, {"MIN_VALID": "2"}, {"OVERLAP": "maybe"}])
def test_bad_parameters(monkeypatch, fixture, tmp_path, env):
    assert run(monkeypatch, fixture, tmp_path / "o", **env) == 2


def test_fulcrum_ignores_small_sign_flips():
    """Dataset 349 at 30 Hz: a -0.3 bpm wobble at 50 uA is not a sign change; the crossing is between the
    last tachycardic and the bradycardic level."""
    ux = np.array([50, 500, 1000, 1500, 2500.0])
    xf, note = DR.fulcrum(ux, np.array([-0.33, 5.08, 4.95, 1.76, -5.25]), 3.5)
    assert 1500 < xf < 2500 and note == "tachycardia to bradycardia"
    assert DR.fulcrum(ux, np.array([-0.3, 5.0, 4.0, 3.0, 2.0]), 3.5) == (None, "")
