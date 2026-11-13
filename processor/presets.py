"""
Species presets for the response windows.

stim-dose-response takes two numbers from the species: how long the baseline before each train is
(BASELINE_S) and how long the window after it is (AFTER_S). Neither has a published standard; they are
a design rule of this app, shown here, in the README and in provenance.json, and each can be overridden
by its own parameter:

  the window must hold at least two breaths and about ten heart beats at the species' resting rates,
  so that the respiratory oscillation of heart rate and blood pressure (respiratory sinus arrhythmia)
  largely averages out, and should be short enough to stay clear of the previous train in a typical
  protocol.

Large animals and humans breathe 10-20 times a minute (10 s holds 2-3 breaths); rats and mice breathe
much faster (5 s holds several breaths and dozens of beats). The pig preset equals the protocol of the
Discover 349 demo dataset (10 s recorded before and after each 5 s train; Armitage et al.,
doi:10.26275/dm5l-tspb), which is the only number here taken from a source.
"""
from __future__ import annotations

SPECIES = ("human", "pig", "dog", "rabbit", "rat", "mouse")

PRESETS = {
    "human": {"baseline_s": 10.0, "after_s": 10.0,
              "source": "design rule (>= 2 breaths, ~10 beats at resting rates)"},
    "pig": {"baseline_s": 10.0, "after_s": 10.0,
            "source": "design rule; equals the 10 s before/after each train of Discover 349 (Armitage et al.)"},
    "dog": {"baseline_s": 10.0, "after_s": 10.0,
            "source": "design rule (>= 2 breaths, ~10 beats at resting rates)"},
    "rabbit": {"baseline_s": 5.0, "after_s": 5.0,
               "source": "design rule (>= 2 breaths, ~10 beats at resting rates)"},
    "rat": {"baseline_s": 5.0, "after_s": 5.0,
            "source": "design rule (>= 2 breaths, ~10 beats at resting rates)"},
    "mouse": {"baseline_s": 5.0, "after_s": 5.0,
              "source": "design rule (>= 2 breaths, ~10 beats at resting rates)"},
}

# Variables whose sign change across amplitude defines the neural fulcrum (Ardell et al. 2017): heart
# rate, from the ECG (ecg-hrv hr_bpm) or a device channel (hr_device_bpm).
FULCRUM_PREFIX = "hr_"

# Variables shown in figures by default (FIGURE_VARIABLES=auto), in this order, when present.
FIGURE_DEFAULT = ("hr_bpm", "hr_device_bpm", "map_mmhg", "resp_rate_bpm")

UNITS = {"hr_bpm": "bpm", "hr_device_bpm": "bpm", "rr_ms": "ms", "sbp_mmhg": "mmHg", "dbp_mmhg": "mmHg",
         "map_mmhg": "mmHg", "pp_mmhg": "mmHg", "resp_rate_bpm": "breaths/min", "resp_amp": "a.u."}

LABELS = {"hr_bpm": "heart rate (ECG)", "hr_device_bpm": "heart rate (device)", "rr_ms": "RR interval",
          "sbp_mmhg": "systolic pressure", "dbp_mmhg": "diastolic pressure", "map_mmhg": "mean arterial pressure",
          "pp_mmhg": "pulse pressure", "resp_rate_bpm": "respiratory rate", "resp_amp": "respiratory amplitude"}


def normalise_species(s: str | None) -> str:
    s = (s or "").strip().lower()
    aliases = {"homo sapiens": "human", "sus scrofa": "pig", "sus scrofa domesticus": "pig", "swine": "pig",
               "porcine": "pig", "canis familiaris": "dog", "canis lupus familiaris": "dog", "canine": "dog",
               "oryctolagus cuniculus": "rabbit", "rattus norvegicus": "rat", "mus musculus": "mouse",
               "rats": "rat", "mice": "mouse", "pigs": "pig"}
    s = aliases.get(s, s)
    return s if s in SPECIES else ""


def unit_of(variable: str) -> str:
    if variable in UNITS:
        return UNITS[variable]
    tail = variable.rsplit("_", 1)[-1]
    return {"bpm": "bpm", "ms": "ms", "mmhg": "mmHg", "s": "s", "uv": "µV", "mv": "mV", "pct": "%",
            "hz": "Hz"}.get(tail, "a.u.")


def label_of(variable: str) -> str:
    return LABELS.get(variable, variable.replace("_", " "))
