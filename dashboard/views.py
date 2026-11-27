from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie


def landing(request):
    return render(request, "dashboard/landing.html")


@ensure_csrf_cookie  # the dashboard JS posts the user's FIRMS key and needs the CSRF cookie
def workspace(request):
    return render(request, "dashboard/workspace.html")