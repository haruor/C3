from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import json

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

from .forms import UploadCSVForm
from .engine.sim import SimConfig, SimState, Simulator

SESSION_KEY = "sim_session"

# 開発用：セッションID -> Simulator をメモリ保持
# （本番を考えるならRedis等にしたいが、まず動かすため）
_SIMULATORS: dict[str, Simulator] = {}


def _save_upload(f, subdir: str) -> str:
    name = f"{subdir}/{uuid4()}_{f.name}"
    return default_storage.save(name, f)


def _save_uploaded_csv(uploaded_file, prefix: str) -> Path:
    """
    アップロードCSVをMEDIA配下へ保存して Path を返す。
    既存の保存処理があるならそれを優先してOK。
    """
    name = getattr(uploaded_file, "name", "upload.csv")
    ext = ".csv" if not name.lower().endswith(".csv") else ""
    filename = f"{prefix}_{uuid4().hex}{ext}"
    rel_path = Path("uploads") / filename
    saved_path = default_storage.save(str(rel_path), uploaded_file)
    return Path(settings.MEDIA_ROOT) / saved_path


def _get_or_create_sim_id(request: HttpRequest) -> str:
    sim = request.session.get(SESSION_KEY)
    if sim and "id" in sim:
        return sim["id"]
    sim_id = uuid4().hex
    request.session[SESSION_KEY] = {"id": sim_id}
    request.session.modified = True
    return sim_id


def upload_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            elev_f = form.cleaned_data["elevation_csv"]
            vege_f = form.cleaned_data["vegetation_csv"]
            ign_f = form.cleaned_data.get("ignition_csv")  # 任意

            elev_path = _save_uploaded_csv(elev_f, "elevation")
            vege_path = _save_uploaded_csv(vege_f, "vegetation")
            ign_path = _save_uploaded_csv(ign_f, "ignition") if ign_f else None

            request.session[SESSION_KEY] = {
                "id": _get_or_create_sim_id(request),
                "elev_rel": str(elev_path),
                "vege_rel": str(vege_path),
                "ign_rel": str(ign_path) if ign_path else "",
                "t": 0,
                "is_paused": False,
                "water_mode": False,
            }
            request.session.modified = True

            # ★ここが重要：ignition_path を SimState に渡す
            sim_state = SimState(
                elev_path=Path(str(elev_path)),
                vege_path=Path(str(vege_path)),
                ignition_path=Path(str(ign_path)) if ign_path else None,
            )
            _SIMULATORS[request.session[SESSION_KEY]["id"]] = Simulator(state=sim_state, config=SimConfig())

            return redirect("simulation:run")
    else:
        form = UploadCSVForm()

    return render(request, "simulation/upload.html", {"form": form})


def run_view(request: HttpRequest) -> HttpResponse:
    sim = request.session.get(SESSION_KEY)
    if not sim or "id" not in sim:
        return redirect("simulation:upload")

    context = {
        "poll_interval_ms": 200,  # 後で調整
    }
    return render(request, "simulation/run.html", context)


def _get_simulator(request: HttpRequest) -> Simulator:
    sim = request.session.get(SESSION_KEY)
    if not sim or "id" not in sim:
        raise ValueError("セッションがありません。")
    sim_id = sim["id"]
    if sim_id not in _SIMULATORS:
        # サーバ再起動などで消えた場合：アップロードからやり直し
        raise ValueError("シミュレータが見つかりません。アップロードからやり直してください。")
    return _SIMULATORS[sim_id]


@require_GET
def api_frame(request: HttpRequest) -> HttpResponse:
    try:
        sim = _get_simulator(request)
    except ValueError as e:
        # フロント側で「アップロードからやり直し」を促せるようにする
        return JsonResponse({"ok": False, "error": str(e), "code": "SIM_NOT_FOUND"}, status=409)

    STEPS_PER_TICK = 1
    sim.step(n=STEPS_PER_TICK)

    kind = request.GET.get("kind", "fire")
    png = sim.render_elev_png() if kind == "elev" else sim.render_fire_png()
    return HttpResponse(png, content_type="image/png")


@require_POST
def api_toggle_pause(request: HttpRequest) -> JsonResponse:
    sim = _get_simulator(request)
    is_paused = sim.toggle_pause()
    return JsonResponse({"ok": True, "is_paused": is_paused})


@require_POST
def api_toggle_water(request: HttpRequest) -> JsonResponse:
    sim = _get_simulator(request)
    water_mode = sim.toggle_water()
    return JsonResponse({"ok": True, "water_mode": water_mode})


@require_POST
def api_place_water(request: HttpRequest) -> JsonResponse:
    sim = _get_simulator(request)

    try:
        payload = json.loads(request.body.decode("utf-8"))
        i = int(payload["i"])
        j = int(payload["j"])
        radius = int(payload.get("radius", 1))
    except Exception:
        return JsonResponse({"ok": False, "error": "invalid payload"}, status=400)

    if not sim.state.water_mode:
        return JsonResponse(
            {"ok": True, "skipped": True, "reason": "water_mode_off", "water_remaining_l": sim.state.water_remaining_l}
        )

    before = sim.state.water_remaining_l
    placed = sim.place_water(i=i, j=j, radius=radius)
    after = sim.state.water_remaining_l

    return JsonResponse(
        {
            "ok": True,
            "placed": placed,
            "water_used_l": before - after,
            "water_remaining_l": after,
            "out_of_water": after < sim.WATER_COST_PER_CELL_L,
        }
    )


@require_GET
def api_status(request: HttpRequest) -> JsonResponse:
    sim = _get_simulator(request)

    cooldown_remaining = max(0, int(sim.state.relief_next_available_t - sim.state.t))

    return JsonResponse(
        {
            "ok": True,
            "t": sim.state.t,
            "is_paused": sim.state.is_paused,
            "water_mode": sim.state.water_mode,
            "water_remaining_l": sim.state.water_remaining_l,
            "water_cost_per_cell_l": sim.WATER_COST_PER_CELL_L,

            # ★救援
            "relief_available": sim.can_relief(),
            "relief_cooldown_remaining_steps": cooldown_remaining,
        }
    )


@require_POST
def api_relief(request: HttpRequest) -> JsonResponse:
    sim = _get_simulator(request)
    return JsonResponse(sim.relief())
