import os
import sys

from django.apps import AppConfig


class DashboardConfig(AppConfig):
    name = 'dashboard'

    def ready(self):
        # ready() runs for every management command (migrate, test, shell, ...), not just
        # the server, and -- with the autoreloader on (the default) -- twice for runserver
        # itself: once in the watcher process and once in the actual child process Django
        # restarts with RUN_MAIN=true. Only start the scheduler in the one process that is
        # actually going to serve requests.
        if os.environ.get("PYRO_AUTO_UPDATE_ON_START", "1") != "1":
            return
        if "runserver" not in sys.argv:
            return
        if "--noreload" not in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        from core.src import auto_update
        auto_update.start_background_scheduler()
