"""python manage.py auto_update [--force] [--sources ...] [--bbox west,south,east,north]
                                [--initial-lookback-days N] [--skip-predict] [--skip-anomalies]

Idempotent daily refresh. On its FIRST run in a UTC day it:

  1. looks at the newest day already in data/processed/calendar_daily.csv (or, if the
     calendar does not exist yet, starts --initial-lookback-days before today);
  2. downloads NASA FIRMS **near-real-time (NRT)** data for every day after that through
     today, UTC (core.src.download_firms -- NRT, not the ~5-month-delayed *_SP archive
     sources, see README section 7);
  3. re-runs preprocess + harmonize so the calendar covers the new day(s);
  4. (re-)runs detect_anomalies: the global anomaly episodes, critical burning windows and
     season-by-year report the dashboard's "Anomalies & explanation" card reads (needs >= 2
     years of valid calendar days; skip with --skip-anomalies);
  5. re-runs predict_countries so forecast_by_country.csv / recent_by_country.csv reflect
     the new data;
  6. runs live_confidence: archives that forecast, scores earlier archived forecasts against
     the real-time data that has arrived since, and writes the per-country confidence numbers
     the dashboard shows (core.validation.live_confidence).

Steps 4 and 6 are never fatal: a failure there is reported but does not fail the refresh or
leave the day unmarked, since the calendar and forecast (steps 1-3 and 5) are already fine at
that point -- it only costs one day of anomaly/comparison history. Both are also (re)attempted
on every run, pulled new data or not, so a previous failure there actually gets retried instead
of silently staying broken (the same reasoning as step 5's predict_countries retry below).

Re-running it again the same UTC day is a no-op (checked against
data/processed/auto_update_state.json) -- pass --force to pull again anyway (this also
re-downloads today's NRT file even if it already exists, since NRT data streams in
throughout the day).

This does NOT replace the one-time archive backfill (README section 7): run
    python manage.py download_firms --start ... --end ... --sources MODIS_SP VIIRS_SNPP_SP
once first to build real multi-year history (and `train_countries` once on top of it), then
let this command keep the calendar and the forecast current from then on. Needs
FIRMS_MAP_KEY set, same as download_firms.

Intended to be run once a day (cron / Windows Task Scheduler / a CI schedule); it is also
always safe to run more often by hand -- the state file is what limits it to one real pull
per UTC day.

Runs on its own, too: dashboard/apps.py starts an in-process scheduler
(start_background_scheduler, below) the moment `python manage.py runserver` comes up. It does
one refresh immediately, then one again a few minutes after every UTC midnight for as long as
the server process keeps running, so a server left running across midnight picks up each new
day by itself -- no cron job needed on top of it. Set PYRO_AUTO_UPDATE_ON_START=0 to turn that
off and rely only on running this command yourself (or your own scheduler).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from core.src.config import PROCESSED_DIR, OUTPUTS_DIR, ensure_dirs

# core/src/auto_update.py -> parents[2] is the repo root (where manage.py lives), the same
# depth config.py uses for its default ROOT -- computed independently of PYRO_DATA_ROOT so
# this still finds manage.py even when PYRO_DATA_ROOT points data at a different folder.
MANAGE_PY = Path(__file__).resolve().parents[2] / "manage.py"
STATE_FILE_NAME = "auto_update_state.json"

# SP (standard, science-quality) archive sources and their NRT (near-real-time) counterparts.
# _SP lags real time by ~5 months (README section 7), so the daily top-up has to use _NRT,
# on the same satellites the archive was built with -- an _SP name here is mapped, not used as-is.
NRT_OF = {
    "MODIS_SP": "MODIS_NRT", "MODIS_NRT": "MODIS_NRT",
    # S-NPP's NRT feed is dead (see note above); redirect it to NOAA-20 rather than a source
    # that will only ever return 0 rows from here on.
    "VIIRS_SNPP_SP": "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT": "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA20_SP": "VIIRS_NOAA20_NRT", "VIIRS_NOAA20_NRT": "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_SP": "VIIRS_NOAA21_NRT", "VIIRS_NOAA21_NRT": "VIIRS_NOAA21_NRT",
}
DEFAULT_SOURCES = ["MODIS_NRT", "VIIRS_NOAA20_NRT"]
# Suomi NPP (S-NPP)'s NRT feed was permanently retired on 2026-11-02 13:00 UTC -- NASA's own
# guidance is to move FIRMS API callers from VIIRS_SNPP_NRT/_SP to VIIRS_NOAA20_NRT or
# VIIRS_NOAA21_NRT (see https://www.earthdata.nasa.gov/news/blog/transitioning-from-suomi-npp-viirs-omps-what-firms-users-need-know).
# An explicit --sources VIIRS_SNPP_NRT/_SP is therefore redirected to VIIRS_NOAA20_NRT below
# rather than left to return 0 rows.


def _state_path() -> Path:
    return PROCESSED_DIR / STATE_FILE_NAME


def _load_state() -> dict:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    ensure_dirs()
    _state_path().write_text(json.dumps(state, indent=2, default=str))


def _last_calendar_date():
    """Newest UTC day already in calendar_daily.csv, or None if it does not exist / is empty."""
    path = PROCESSED_DIR / "calendar_daily.csv"
    if not path.exists():
        return None
    dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date()


def _run_step(args: list[str]) -> None:
    """Run one `python manage.py <args>` step, streaming its output, and raise if it fails."""
    print(f"\n$ {sys.executable} manage.py {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(MANAGE_PY)] + args, check=True, cwd=str(MANAGE_PY.parent))


def _run_optional_step(command: str, context: str) -> None:
    """Run one step that must never fail the whole refresh: report the failure loudly and carry
    on. `context` says, in one clause, what stays fine regardless (shown in the message)."""
    try:
        _run_step([command])
    except subprocess.CalledProcessError as e:
        print(f"\n[auto_update] {command} failed (exit {e.returncode}); {context}.",
              file=sys.stderr, flush=True)


def run(force=False, sources=None, bbox=None, initial_lookback_days=10,
        skip_predict=False, with_anomalies=True) -> bool:
    """Do one idempotent refresh attempt -- the logic shared by main() (the CLI) and the
    background scheduler below. Returns True if new FIRMS data was actually pulled, False if
    the calendar already reached today (a short-circuit on the download/preprocess/harmonize
    steps only -- predict_countries is still (re)attempted either way, unless skip_predict or
    no trained model). Returns False without doing anything if auto_update already completed
    today. Raises RuntimeError if a step fails (today is left unmarked, so the next attempt
    retries it)."""
    today = datetime.now(timezone.utc).date()
    state = _load_state()

    if not force and state.get("last_run_date") == today.isoformat() and state.get("status") == "ok":
        print(f"auto_update already completed today ({today.isoformat()} UTC); nothing to do "
              "(force=True / --force to pull again).")
        return False

    last = _last_calendar_date()
    if last is None:
        start = today - timedelta(days=initial_lookback_days)
        print(f"No calendar_daily.csv yet: fetching the last {initial_lookback_days} day(s) of NRT "
              "data to start. For real multi-year history, run the one-time archive backfill first "
              "(README section 7) before relying on this for anything but a demo.")
    else:
        start = last + timedelta(days=1)
    if force and start > today:
        start = today  # re-pull at least today even if the calendar already reaches it

    sources = [NRT_OF.get(s.upper(), s.upper()) for s in (sources or DEFAULT_SOURCES)]
    pulled = start <= today

    try:
        if pulled:
            dl_args = ["download_firms", "--start", start.isoformat(), "--end", today.isoformat(),
                       "--sources", *sources]
            if bbox:
                dl_args += ["--bbox", bbox]
            if force:
                dl_args.append("--force")
            _run_step(dl_args)

            _run_step(["preprocess"])
            _run_step(["harmonize"])
        else:
            print(f"Calendar already covers through {last} (today is {today}); nothing new to pull.")

        # Always (re)attempt detect_anomalies, pulled or not, and never let it fail the refresh:
        # a previous failure (or a day with nothing new to pull) must not leave it silently stale.
        if with_anomalies:
            _run_optional_step("detect_anomalies",
                               "the calendar is fine, but anomalies/critical-windows were not refreshed")
        else:
            print("\nskip_anomalies set: anomaly detection left untouched.")

        # Always (re)attempt predict_countries, pulled or not: it is cheap and idempotent, and
        # this is also what retries a predict_countries failure from an earlier attempt today
        # even when there is no new calendar data this time -- skipping it here would silently
        # mask that failure instead of actually retrying it.
        if skip_predict:
            print("\nskip_predict set: forecast left untouched.")
        elif not (OUTPUTS_DIR / "model_countries.joblib").exists():
            print(f"\nNo trained model at {OUTPUTS_DIR / 'model_countries.joblib'}; skipping "
                  "predict_countries. Run `python manage.py train_countries ...` once first.")
        else:
            _run_step(["predict_countries"])
            _run_optional_step("live_confidence",
                               "the forecast is fine, but today's forecast was not archived for comparison")
    except subprocess.CalledProcessError as e:
        _save_state({**state, "status": "error", "failed_step": list(e.cmd),
                     "failed_at": datetime.now(timezone.utc).isoformat()})
        raise RuntimeError(
            f"auto_update FAILED at step {list(e.cmd)!r} (exit {e.returncode}); today is NOT "
            "marked done, so the next attempt will retry from here."
        ) from e

    state_update = {"last_run_date": today.isoformat(), "status": "ok",
                     "completed_at": datetime.now(timezone.utc).isoformat()}
    if pulled:
        state_update.update(last_pulled_start=start.isoformat(), last_pulled_end=today.isoformat(),
                             sources=sources)
    else:
        state_update["note"] = "already current"
    _save_state(state_update)
    print(f"\nDone: calendar through {today.isoformat()} (UTC)"
          + (" refreshed and forecast updated." if pulled else "; forecast (re)checked."))
    return pulled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                     help="pull again even if auto_update already completed today (UTC); "
                          "also re-downloads today's NRT file if it already exists")
    ap.add_argument("--sources", nargs="+", default=None,
                     help=f"FIRMS sources to pull (default: {DEFAULT_SOURCES}); an _SP name "
                          "is mapped to its _NRT equivalent automatically")
    ap.add_argument("--bbox", default=os.getenv("PYRO_BBOX"),
                     help="west,south,east,north for the FIRMS request (default: $PYRO_BBOX, else global)")
    ap.add_argument("--initial-lookback-days", type=int, default=10,
                     help="days of NRT history to fetch when calendar_daily.csv does not exist yet (default: 10)")
    ap.add_argument("--skip-predict", action="store_true",
                     help="refresh the calendar but do not run predict_countries")
    ap.add_argument("--skip-anomalies", action="store_true",
                     help="skip detect_anomalies (it runs by default on every refresh)")
    args = ap.parse_args()
    try:
        run(force=args.force, sources=args.sources, bbox=args.bbox,
            initial_lookback_days=args.initial_lookback_days,
            skip_predict=args.skip_predict, with_anomalies=not args.skip_anomalies)
    except RuntimeError as e:
        sys.exit(f"\n{e}")


# ---------------------------------------------------------------------------------------
# In-process scheduler: keeps running automatically for as long as the server is up.
# Started from dashboard/apps.py's AppConfig.ready() (see there for the runserver / reloader
# guards); PYRO_AUTO_UPDATE_ON_START=0 disables it. Calling this more than once is harmless --
# only the first call starts a thread.
# ---------------------------------------------------------------------------------------
_scheduler_started = False
_scheduler_lock = threading.Lock()
RETRY_DELAY_S = 30 * 60   # a failed scheduled refresh is retried after this long ...
MAX_QUICK_RETRIES = 3     # ... up to this many times in a row, then it waits for the next UTC midnight


def start_background_scheduler(initial_lookback_days: int = 10) -> None:
    """Run one refresh now, in a daemon thread, then one again a few minutes after every UTC
    midnight for as long as this process stays alive."""
    global _scheduler_started
    with _scheduler_lock:
        if _scheduler_started:
            return
        _scheduler_started = True
    threading.Thread(target=_scheduler_loop, args=(initial_lookback_days,),
                      name="pyroscope-auto-update", daemon=True).start()


def _seconds_until_next_run(buffer_minutes: int = 5) -> float:
    """Seconds until a few minutes after the next UTC midnight -- a small buffer so the FIRMS
    NRT feed for the new day has had a moment to start populating before we fetch it."""
    now = datetime.now(timezone.utc)
    next_run = (now + timedelta(days=1)).replace(hour=0, minute=buffer_minutes, second=0, microsecond=0)
    return max((next_run - now).total_seconds(), 60.0)


def _scheduler_loop(initial_lookback_days: int) -> None:
    failures = 0
    while True:
        print(f"[auto_update] running ({datetime.now(timezone.utc).isoformat()})", flush=True)
        sleep_s = _seconds_until_next_run()
        try:
            run(initial_lookback_days=initial_lookback_days)
            failures = 0
        except Exception:
            print("[auto_update] refresh attempt failed:", file=sys.stderr, flush=True)
            traceback.print_exc()
            failures += 1
            if failures <= MAX_QUICK_RETRIES:
                # transient problems (network, a locked file, a FIRMS hiccup) are common: try again
                # soon instead of leaving the data stale until the next midnight
                sleep_s = min(RETRY_DELAY_S, sleep_s)
                print(f"[auto_update] retry {failures}/{MAX_QUICK_RETRIES} in {sleep_s / 60:.0f} min", flush=True)
            else:
                failures = 0   # give up for now; the next scheduled day starts fresh
        next_at = datetime.now(timezone.utc) + timedelta(seconds=sleep_s)
        print(f"[auto_update] next refresh in {sleep_s / 3600:.1f}h (~{next_at.isoformat()})", flush=True)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()