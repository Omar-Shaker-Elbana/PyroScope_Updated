"""Live confidence: how far to trust the newest real-time reading (x%), and how much more
once our forecast independently-ish corroborates it (y%).

Why this module exists
----------------------
forecast_by_country.csv is OVERWRITTEN on every predict_countries run and always starts the day
AFTER the newest observed day, so today's forecast can never be compared with reality -- the
future has not happened yet. To score a forecast you have to keep it until the day it
predicted arrives. This module therefore (1) archives every forecast the day it is made in
OUTPUTS_DIR/forecast_history.csv, (2) scores archived forecasts against the harmonized
observations that have arrived since, and (3) turns that into two numbers per country.

The two numbers (all inputs are files already written by the pipeline; nothing is downloaded)
--------------------------------------------------------------------------------------------
x  data_confidence_pct      Trust in the real-time reading ITSELF, before looking at any forecast.
                            = STATUS_FACTOR[harmonized_status of that day]
                              * TRUST_FACTOR[core.validation.confidence's harmonization flag]
                            i.e. a complete, VIIRS-observed day on a well-validated MODIS scale
                            scores high; a partial day, or a day the reference VIIRS satellite did
                            not observe (e.g. everything after Suomi NPP's 2026-11-02 retirement),
                            or a LOW-trust MODIS k, scores lower.

R  forecast_track_record    How often past forecasts for this country landed inside their own
                            10-90% band when the real data arrived (a Beta-binomial average that
                            starts at the held-out coverage from train_countries' metrics.json and
                            moves toward this country's live hit rate as scored forecasts
                            accumulate; SHRINK_K controls how fast).

A  agreement                How well THIS reading matches what we had predicted for that day:
                            1.0 inside the predicted band, falling linearly to 0 one band-width
                            outside it.

y  corroborated_confidence_pct
                            = x + (100 - x) * EVIDENCE_WEIGHT * R * A
                            A forecast can only ADD confidence, and only in proportion to how
                            well it has performed (R) and how well it matches (A). A reading
                            outside the band is NOT marked down: real fire surges and real data
                            problems both look like that, so it is reported as "outside forecast
                            range" and y stays at x.

Read this before quoting the numbers
------------------------------------
* x is a transparent heuristic index, not a statistical probability. The factors below are
  judgement calls, kept in one place so they are easy to tune.
* The forecast is built from the same satellite stream it is compared with, so agreement is a
  PLAUSIBILITY check (no glitch, dropout or spike relative to recent history and season), not
  independent ground truth -- which is why EVIDENCE_WEIGHT is 0.5, not 1.0. For independent
  evidence use core.src.external_validation (burned area).
* R is the only fully empirical piece; with few scored forecasts it mostly echoes the held-out
  prior, and `n_scored` says how many forecasts it rests on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.src.config import OUTPUTS_DIR, PROCESSED_DIR

HISTORY_NAME = "forecast_history.csv"
OUTPUT_JSON = "live_confidence.json"
OUTPUT_CSV = "live_confidence.csv"

# --- tunable policy (judgement, not statistics) ----------------------------------------------
STATUS_FACTOR = {
    "measured": 1.00,
    "measured_zero": 0.95,
    "measured_low_coverage": 0.60,   # a lower bound, not a measurement (harmonize.py)
    "viirs_not_observed": 0.50,      # reference VIIRS absent: leans on other sensors
    "observed_obscured": 0.40,
    "partial_day": 0.35,             # window not complete yet
    "demo_partial_window": 0.30,
}
DEFAULT_STATUS_FACTOR = 0.50
TRUST_FACTOR = {"OK": 1.00, "CAUTION": 0.90, "LOW_TRUST": 0.75, "UNKNOWN": 0.70}
EVIDENCE_WEIGHT = 0.5      # forecast shares its data source with the reading -> partial evidence only
SHRINK_K = 20              # pseudo-observations behind the held-out prior
DEFAULT_PRIOR = 0.80       # nominal coverage of the p10-p90 band
TRACK_RECORD_DAYS = 120    # only recent scored forecasts count toward R
KEEP_HISTORY_DAYS = 400


def _out_dir() -> Path:
    return Path(OUTPUTS_DIR)


# --------------------------------------------------------------------------- forecast archive
def archive_forecast(forecast_csv: str | Path | None = None,
                     history_csv: str | Path | None = None) -> pd.DataFrame:
    """Append the current forecast to the history file (idempotent: re-running on the same
    forecast replaces its own rows instead of duplicating them). Returns the full history."""
    forecast_csv = Path(forecast_csv) if forecast_csv else _out_dir() / "forecast_by_country.csv"
    history_csv = Path(history_csv) if history_csv else _out_dir() / HISTORY_NAME
    if not forecast_csv.exists():
        raise FileNotFoundError(f"{forecast_csv} not found. Run `python manage.py predict_countries` first.")

    fc = pd.read_csv(forecast_csv, parse_dates=["date"])
    keep = ["country", "date", "horizon_days", "predicted", "low_p10", "high_p90"]
    missing = set(keep) - set(fc.columns)
    if missing:
        raise ValueError(f"{forecast_csv.name} is missing column(s): {sorted(missing)}")
    new = fc[keep + ([c for c in ("name", "target") if c in fc.columns])].copy()
    # issued_on = the last observed day the forecast was built from (target date minus horizon)
    new["issued_on"] = new["date"] - pd.to_timedelta(new["horizon_days"].astype(int), unit="D")

    if history_csv.exists():
        old = pd.read_csv(history_csv, parse_dates=["date", "issued_on"])
        hist = pd.concat([old, new], ignore_index=True)
    else:
        hist = new
    hist = hist.drop_duplicates(["country", "date", "horizon_days"], keep="last")
    cutoff = hist["date"].max() - pd.Timedelta(days=KEEP_HISTORY_DAYS)
    hist = hist[hist["date"] >= cutoff].sort_values(["country", "date", "horizon_days"]).reset_index(drop=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(history_csv, index=False)
    return hist


# --------------------------------------------------------------------------- scoring
def load_observed(recent_csv: str | Path | None = None) -> pd.DataFrame:
    """recent_by_country.csv -> long frame (date, country, observed), NaN rows dropped."""
    recent_csv = Path(recent_csv) if recent_csv else _out_dir() / "recent_by_country.csv"
    if not recent_csv.exists():
        raise FileNotFoundError(f"{recent_csv} not found. Run `python manage.py predict_countries` first.")
    obs = pd.read_csv(recent_csv, parse_dates=["date"])
    obs["observed"] = pd.to_numeric(obs["observed"], errors="coerce")
    return obs.dropna(subset=["observed"])[["date", "country", "observed"]]


def agreement_score(observed, low, high, predicted) -> tuple[float, str]:
    """(A in [0,1], 'within'|'above'|'below'). 1.0 inside [low, high]; linear fall-off to 0 at
    one band-width outside it (band-width floored so a razor-thin band cannot make A brittle)."""
    scale = max(float(high) - float(low), 0.1 * abs(float(predicted)), 1.0)
    if observed < low:
        return max(0.0, 1.0 - (float(low) - observed) / scale), "below"
    if observed > high:
        return max(0.0, 1.0 - (observed - float(high)) / scale), "above"
    return 1.0, "within"


def score_history(history: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Archived forecasts joined to the observations that have since arrived."""
    s = history.merge(observed, on=["country", "date"], how="inner")
    if s.empty:
        return s.assign(in_band=pd.Series(dtype=bool), abs_error=pd.Series(dtype=float))
    s["in_band"] = (s["observed"] >= s["low_p10"]) & (s["observed"] <= s["high_p90"])
    s["abs_error"] = (s["observed"] - s["predicted"]).abs()
    return s


def forecast_track_record(scored: pd.DataFrame, prior: float = DEFAULT_PRIOR,
                          k: int = SHRINK_K, days: int = TRACK_RECORD_DAYS) -> pd.DataFrame:
    """Per-country R (see module docstring) and the n_scored behind it."""
    if scored.empty:
        return pd.DataFrame(columns=["country", "n_scored", "hit_rate_raw", "track_record"])
    recent = scored[scored["date"] >= scored["date"].max() - pd.Timedelta(days=days)]
    g = recent.groupby("country")["in_band"]
    out = pd.DataFrame({"n_scored": g.size(), "hits": g.sum()}).reset_index()
    out["hit_rate_raw"] = out["hits"] / out["n_scored"]
    out["track_record"] = (out["hits"] + k * prior) / (out["n_scored"] + k)
    return out.drop(columns="hits")


# --------------------------------------------------------------------------- x and y
def data_confidence_pct(status: str, trust_flag: str) -> float:
    return 100.0 * STATUS_FACTOR.get(status, DEFAULT_STATUS_FACTOR) * TRUST_FACTOR.get(trust_flag, 0.70)


def corroborated_pct(x: float, track_record: float, agreement: float) -> float:
    return x + (100.0 - x) * EVIDENCE_WEIGHT * track_record * agreement


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _prior_from_metrics() -> float:
    cov = _read_json(_out_dir() / "metrics.json").get("interval_80pct_coverage")
    return float(cov) if isinstance(cov, (int, float)) and 0 < cov <= 1 else DEFAULT_PRIOR


def build_live_confidence(write: bool = True) -> dict:
    """Archive the current forecast, score history, and return/write the per-country x% / y%."""
    from core.validation.confidence import harmonization_confidence

    out_dir = _out_dir()
    history = archive_forecast()
    observed = load_observed()
    if observed.empty:
        raise ValueError("recent_by_country.csv has no observed values yet.")

    cal_path = Path(PROCESSED_DIR) / "calendar_daily.csv"
    status_by_day: dict = {}
    if cal_path.exists():
        cal = pd.read_csv(cal_path, parse_dates=["date"], usecols=["date", "harmonized_status"])
        status_by_day = dict(zip(cal["date"], cal["harmonized_status"]))

    mapping = _read_json(Path(PROCESSED_DIR) / "frp_mapping.json")
    countries = sorted(observed["country"].unique())
    hc = harmonization_confidence(mapping, countries).set_index("country")

    names = {}
    if "name" in history.columns:
        names = history.dropna(subset=["name"]).drop_duplicates("country").set_index("country")["name"].to_dict()

    prior = _prior_from_metrics()
    scored = score_history(history, observed)
    record = forecast_track_record(scored, prior=prior).set_index("country") if not scored.empty else None

    # headline reading = newest COMPLETE day (today is almost always a partial day)
    complete_days = sorted(d for d in observed["date"].unique() if status_by_day.get(d) != "partial_day")
    rows = []
    for c in countries:
        obs_c = observed[observed["country"] == c]
        days_c = obs_c[obs_c["date"].isin(complete_days)] if complete_days else obs_c
        if days_c.empty:
            days_c = obs_c
        d = days_c["date"].max()
        reading = float(days_c.loc[days_c["date"] == d, "observed"].iloc[0])
        status = status_by_day.get(d, "unknown")
        flag = hc.loc[c, "confidence_flag"]
        x = data_confidence_pct(status, flag)

        n = int(record.loc[c, "n_scored"]) if record is not None and c in record.index else 0
        R = float(record.loc[c, "track_record"]) if n else prior

        f = history[(history["country"] == c) & (history["date"] == d)].sort_values("horizon_days")
        if f.empty:
            y, A, label, fc_rec = x, None, "no_prior_forecast", None
        else:
            fc_rec = f.iloc[0]   # shortest horizon = the forecast made closest to that day
            A, label = agreement_score(reading, fc_rec["low_p10"], fc_rec["high_p90"], fc_rec["predicted"])
            y = corroborated_pct(x, R, A)
            label = {"within": "within_forecast_range", "above": "above_forecast_range",
                     "below": "below_forecast_range"}[label]
        rows.append({
            "country": c, "name": names.get(c, c), "date": str(pd.Timestamp(d).date()),
            "observed": round(reading, 2), "harmonized_status": status,
            "harmonization_flag": flag,
            "data_confidence_pct": round(x, 1),
            "corroborated_confidence_pct": round(y, 1),
            "agreement": label,
            "agreement_score": None if A is None else round(A, 3),
            "forecast_track_record_pct": round(100 * R, 1),
            "n_scored_forecasts": n,
            "forecast_predicted": None if fc_rec is None else round(float(fc_rec["predicted"]), 2),
            "forecast_low_p10": None if fc_rec is None else round(float(fc_rec["low_p10"]), 2),
            "forecast_high_p90": None if fc_rec is None else round(float(fc_rec["high_p90"]), 2),
            "forecast_horizon_days": None if fc_rec is None else int(fc_rec["horizon_days"]),
            "forecast_issued_on": None if fc_rec is None else str(pd.Timestamp(fc_rec["issued_on"]).date()),
        })

    df = pd.DataFrame(rows)
    w = df["observed"].clip(lower=0)
    wmean = (lambda col: float(np.average(df[col], weights=w)) if w.sum() > 0 else float(df[col].mean()))
    compared = df["agreement"] != "no_prior_forecast"
    result = {
        "success": True,
        "generated_for_date": str(pd.to_datetime(df["date"]).max().date()),
        "overall": {
            "data_confidence_pct": round(wmean("data_confidence_pct"), 1),
            "corroborated_confidence_pct": round(wmean("corroborated_confidence_pct"), 1),
            "weighting": "countries weighted by their observed FRP (MW)",
            "countries": int(len(df)),
            "countries_with_forecast_comparison": int(compared.sum()),
            "within_forecast_range": int((df["agreement"] == "within_forecast_range").sum()),
            "scored_forecasts_total": int(len(scored)),
            "forecast_history_days": int(history["issued_on"].nunique()),
        },
        "method": {
            "data_confidence": "STATUS_FACTOR[harmonized_status] x TRUST_FACTOR[harmonization flag] "
                               "(heuristic index, not a probability)",
            "forecast_track_record": f"share of past forecasts that landed in their own 10-90% band, "
                                     f"shrunk toward the held-out prior ({prior:.0%}) with k={SHRINK_K}",
            "corroborated": f"x + (100-x) * {EVIDENCE_WEIGHT} * track_record * agreement; "
                            "agreement can raise confidence but never lowers it",
            "caveat": "the forecast is built from the same satellite stream, so agreement is a "
                      "plausibility check, not independent ground truth",
        },
        "countries": df.sort_values("observed", ascending=False).to_dict(orient="records"),
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / OUTPUT_JSON).write_text(json.dumps(result, indent=2, default=str))
        df.to_csv(out_dir / OUTPUT_CSV, index=False)
    return result
