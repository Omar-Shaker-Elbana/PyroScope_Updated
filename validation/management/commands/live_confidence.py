"""python manage.py live_confidence

Archives the forecast predict_countries just wrote, scores earlier archived forecasts against
the real-time observations that have arrived since, and writes live_confidence.json / .csv into
core.src.config.OUTPUTS_DIR -- the per-country "data confidence x%" and "corroborated
confidence y%" the dashboard shows (core.validation.live_confidence).

Run it after every predict_countries (auto_update does this for you): the archive step is what
makes tomorrow's comparison possible, so skipping a day leaves a gap in the track record.
"""
from django.core.management.base import BaseCommand, CommandError

from core.validation import live_confidence


class Command(BaseCommand):
    help = "Score forecasts against real-time data and write per-country confidence (live_confidence)."

    def handle(self, *args, **options):
        try:
            res = live_confidence.build_live_confidence()
        except (FileNotFoundError, ValueError) as exc:
            raise CommandError(str(exc))
        o = res["overall"]
        self.stdout.write(
            f"{o['countries']} countries, reading date {res['generated_for_date']}\n"
            f"Overall (FRP-weighted): data confidence {o['data_confidence_pct']}%  ->  "
            f"with forecast corroboration {o['corroborated_confidence_pct']}%\n"
            f"{o['countries_with_forecast_comparison']} countries compared with an archived forecast "
            f"({o['within_forecast_range']} inside the forecast range); "
            f"{o['scored_forecasts_total']} past forecasts scored over {o['forecast_history_days']} issue days."
        )
