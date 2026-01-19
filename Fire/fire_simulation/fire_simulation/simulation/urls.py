from django.urls import path

from . import views

app_name = "simulation"

urlpatterns = [
    path("", views.upload_view, name="upload"),
    path("run/", views.run_view, name="run"),

    # API（結果ページから呼ぶ）
    path("api/frame/", views.api_frame, name="api_frame"),
    path("api/toggle_pause/", views.api_toggle_pause, name="api_toggle_pause"),
    path("api/toggle_water/", views.api_toggle_water, name="api_toggle_water"),
    path("api/place_water/", views.api_place_water, name="api_place_water"),
    path("api/status/", views.api_status, name="api_status"),
    path("api/relief/", views.api_relief, name="api_relief"),
]