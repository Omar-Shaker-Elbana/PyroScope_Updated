"""python manage.py auto_update [--force] [--sources ...] [--bbox ...] [--skip-predict] [--with-anomalies]

Forwards every argument after the command name straight through to core.src.auto_update's own
argparse (no `--` separator needed), same pattern as download_firms and predict_countries.

Idempotent daily refresh: pull new NASA FIRMS NRT data since the last run, re-harmonize, and
refresh the country forecast -- but only once per UTC day (core.src.auto_update). Schedule it
once a day (cron / Task Scheduler) or run it by hand any time; re-running the same day is a
no-op unless --force is given.
"""
import sys

from django.core.management.base import BaseCommand

from core.src import auto_update


class Command(BaseCommand):
    help = ("Daily idempotent refresh: download_firms (NRT) + preprocess + harmonize + "
            "predict_countries, at most once per UTC day (core.src.auto_update).")

    def run_from_argv(self, argv):
        # Hand everything after `manage.py auto_update` to the real argparse untouched,
        # instead of Django's option parser intercepting it.
        sys.argv = [argv[0]] + argv[2:]
        # Django's execute() reads these options, which its own parser would normally supply.
        self.execute(force_color=False, no_color=False, skip_checks=True)

    def handle(self, *args, **options):
        auto_update.main()
