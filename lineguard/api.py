"""LineGuard web API + operator UI.

    uvicorn lineguard.api:app --port 8080

The plant is replayed from simulated telemetry; the "plant clock" (minute t) is chosen by
the UI, and every answer only uses data up to t.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import impact
from .copilot import Copilot
from .detector import BASELINE_MIN, fit
from .knowledge import KnowledgeBase
from .llm import LLM
from .simulator import simulate_plant

ROOT = Path(__file__).resolve().parent.parent
START = dt.datetime(2026, 10, 12, 6, 0)          # Monday, start of early shift
SHIFT_STARTS = (6, 14, 22)
SEED = int(os.environ.get("LINEGUARD_SEED", "7"))
WO_FILE = Path(os.environ.get("LINEGUARD_WO_FILE", ROOT / "data" / "workorders.json"))

ASSETS = simulate_plant(seed=SEED)
MODELS = {a.spec.id: fit(a) for a in ASSETS}
HORIZON = len(ASSETS[0].values)
KB = KnowledgeBase(ROOT / "knowledge_base")
LLM_CLIENT = LLM()
COPILOT = Copilot(KB, LLM_CLIENT)
LOCK = threading.Lock()


def clock(t: int) -> str:
    return (START + dt.timedelta(minutes=int(t))).strftime("%a %d %b %H:%M")


def deadline(t: int, s: dict) -> str | None:
    """Plant time at which the machine is predicted to trip."""
    return clock(t + int(s["rul_h"] * 60)) if s.get("rul_h") else None


def asset_info(a) -> dict:
    return {"id": a.spec.id, "name": a.spec.name, "kind": a.spec.kind, "kind_label": a.kind.label, "area": a.spec.area}


# ---------------------------------------------------------------------------- work orders
def _load_wos() -> list[dict]:
    try:
        return json.loads(WO_FILE.read_text())
    except Exception:
        return []


WORK_ORDERS: list[dict] = _load_wos()


def _save_wos():
    WO_FILE.parent.mkdir(parents=True, exist_ok=True)
    WO_FILE.write_text(json.dumps(WORK_ORDERS, indent=1))


def _active_repair(asset_id: str, t: int) -> dict | None:
    """Approved work order that starts before the machine would have tripped."""
    for wo in WORK_ORDERS:
        if wo["asset_id"] == asset_id and wo["status"] == "APPROVED" and wo["planned_start"] <= t:
            trip = next(a.trip_time for a in ASSETS if a.spec.id == asset_id)
            if trip is None or wo["planned_start"] < trip:
                return wo
    return None


def asset_state(a, t: int) -> dict:
    m = MODELS[a.spec.id]
    wo = _active_repair(a.spec.id, t)
    if wo:
        end = wo["planned_start"] + int(wo["duration_h"] * 60)
        frozen = m.state(wo["planned_start"])
        if t < end:
            return {**frozen, "level": "MAINTENANCE", "health": frozen["health"], "work_order": wo["id"]}
        return {"level": "OK", "health": 100, "deviations": [], "diagnosis": None, "rul_h": None,
                "alert_since": None, "repaired_by": wo["id"], "repaired_at": clock(end)}
    return m.state(t)


# ---------------------------------------------------------------------------- app
app = FastAPI(title="LineGuard", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {"horizon": HORIZON, "baseline_min": BASELINE_MIN, "start": START.isoformat(), "seed": SEED,
            "llm": LLM_CLIENT.info(), "assets": [asset_info(a) for a in ASSETS],
            "kb_chunks": len(KB.chunks)}


@app.get("/api/plant")
def plant(t: int):
    t = int(np.clip(t, 0, HORIZON - 1))
    rows = []
    for a in ASSETS:
        s = asset_state(a, t)
        diag = s.get("diagnosis") or {}
        rows.append({**asset_info(a), "level": s["level"], "health": s["health"], "rul_h": s.get("rul_h"),
                     "diagnosis": diag.get("label"), "top": (s["deviations"] or [{}])[0].get("label"),
                     "alert_since": clock(s["alert_since"]) if s.get("alert_since") is not None else None,
                     "repaired_by": s.get("repaired_by"), "work_order": s.get("work_order")})
    for a, r in zip(ASSETS, rows):
        if r["level"] in ("WATCH", "WARNING", "CRITICAL"):
            s = asset_state(a, t)
            COPILOT.prefetch(asset_info(a), s, clock(t), deadline(t, s))
    return {"t": t, "clock": clock(t), "assets": rows, "kpi": kpis(t), "llm": LLM_CLIENT.info()}


def kpis(t: int) -> dict:
    hours_ahead, saved_h, saved_cost, avoided = 0.0, 0.0, 0, 0
    for a in ASSETS:
        m = MODELS[a.spec.id]
        if m.first_watch is not None and m.first_watch <= t:
            fixed = m.static_alarm if (m.static_alarm is not None and m.static_alarm <= t) else t
            hours_ahead += max(0, fixed - m.first_watch) / 60
        wo = _active_repair(a.spec.id, t)
        if wo and a.trip_time is not None:
            static_lead = None if m.static_alarm is None else (a.trip_time - m.static_alarm) / 60
            c = impact.compare(a.spec.kind, float("inf"), static_lead)  # repair was planned before the trip
            saved_h += c["downtime_saved_h"]
            saved_cost += c["cost_saved"]
            avoided += 1
    states = [asset_state(a, t)["level"] for a in ASSETS]
    return {"running": sum(s not in ("TRIPPED", "MAINTENANCE") for s in states),
            "at_risk": sum(s in ("WATCH", "WARNING", "CRITICAL") for s in states),
            "tripped": states.count("TRIPPED"), "hours_ahead_of_fixed_alarms": round(hours_ahead, 1),
            "trips_avoided": avoided, "downtime_saved_h": round(saved_h, 1), "cost_saved": saved_cost,
            "currency": impact.CURRENCY}


def _asset(asset_id: str):
    a = next((a for a in ASSETS if a.spec.id == asset_id), None)
    if a is None:
        raise HTTPException(404, "unknown asset")
    return a


@app.get("/api/asset/{asset_id}")
def asset_detail(asset_id: str, t: int, window_h: int = 48):
    a = _asset(asset_id)
    t = int(np.clip(t, 0, HORIZON - 1))
    m = MODELS[asset_id]
    s = asset_state(a, t)
    wo = _active_repair(asset_id, t)
    end = min(t, wo["planned_start"]) if wo else t        # after a repair the old trend is no longer valid
    lo = max(0, end - window_h * 60)
    step = max(1, (end - lo) // 400)
    idx = np.arange(lo, end + 1, step)
    hours = (idx // 60) % 24
    series = []
    for j, sn in enumerate(a.sensors):
        series.append({"key": sn.key, "label": sn.label, "unit": sn.unit, "alarm": sn.alarm, "trip": sn.trip,
                       "direction": sn.direction,
                       "values": np.round(a.values[idx, j], 3).tolist(),
                       "normal": np.round(m.profile[hours, j], 3).tolist()})
    marks = []
    if m.first_watch is not None and m.first_watch <= t:
        marks.append({"t": m.first_watch, "label": "LineGuard early warning", "kind": "lineguard"})
    if m.static_alarm is not None and m.static_alarm <= t:
        marks.append({"t": m.static_alarm, "label": "Fixed alarm limit", "kind": "static"})
    if a.trip_time is not None and a.trip_time <= t and not wo:
        marks.append({"t": a.trip_time, "label": "Trip / breakdown", "kind": "trip"})
    if wo:
        marks.append({"t": wo["planned_start"], "label": f"Planned repair {wo['id']}", "kind": "repair"})
    return {**asset_info(a), "t": t, "clock": clock(t), "state": s, "series": series,
            "times": idx.tolist(), "time_labels": [clock(i) for i in idx[:: max(1, len(idx) // 6)]],
            "marks": marks, "repair_slots": repair_slots(t, s.get("rul_h")),
            "work_orders": [w for w in WORK_ORDERS if w["asset_id"] == asset_id]}


def repair_slots(t: int, rul_h: float | None) -> list[dict]:
    """Start now, or at one of the next shift changes (cheapest moment to stop a line)."""
    slots = [{"t": t, "label": f"Now ({clock(t)})"}]
    cur = START + dt.timedelta(minutes=t)
    day = cur.replace(hour=0, minute=0)
    for d in range(3):
        for h in SHIFT_STARTS:
            ts = day + dt.timedelta(days=d, hours=h)
            m = int((ts - START).total_seconds() // 60)
            if m > t and len(slots) < 5:
                late = rul_h is not None and m > t + rul_h * 60
                slots.append({"t": m, "label": f"Shift change {clock(m)}" + (" - after predicted trip!" if late else "")})
    return slots


@app.post("/api/asset/{asset_id}/analyze")
def analyze(asset_id: str, t: int):
    a = _asset(asset_id)
    s = asset_state(a, t)
    if s["level"] in ("OK", "LEARNING"):
        raise HTTPException(400, "no open alert on this machine")
    return COPILOT.analyze(asset_info(a), s, clock(t), deadline(t, s))


class AskBody(BaseModel):
    question: str
    t: int
    role: str = "operator"


@app.post("/api/ask")
def ask(body: AskBody):
    p = plant(body.t)
    status = [{k: r[k] for k in ("id", "name", "level", "health", "rul_h", "diagnosis")} for r in p["assets"]]
    try:
        return COPILOT.ask(body.question[:1000], status, body.role)
    except Exception as e:
        raise HTTPException(502, f"AI model call failed: {type(e).__name__}")


class WOBody(BaseModel):
    asset_id: str
    t: int
    title: str
    priority: str = "P2"
    duration_h: float | None = None
    actions: list = []
    parts: list = []
    safety: list = []
    citations: list = []


@app.get("/api/workorders")
def list_wos():
    return WORK_ORDERS


@app.post("/api/workorders")
def create_wo(body: WOBody):
    a = _asset(body.asset_id)
    dur = body.duration_h or impact.ASSUMPTIONS[a.spec.kind]["planned_h"]
    with LOCK:
        wo = {"id": f"WO-2026-{1101 + len(WORK_ORDERS)}", "asset_id": body.asset_id, "created_at": body.t,
              "created_label": clock(body.t), "title": body.title, "priority": body.priority,
              "duration_h": float(dur), "actions": body.actions, "parts": body.parts, "safety": body.safety,
              "citations": body.citations, "status": "PENDING_APPROVAL", "planned_start": None}
        WORK_ORDERS.append(wo)
        _save_wos()
    return wo


class ApproveBody(BaseModel):
    planned_start: int
    approver: str = "Maintenance planner"


@app.post("/api/workorders/{wo_id}/approve")
def approve_wo(wo_id: str, body: ApproveBody):
    with LOCK:
        wo = next((w for w in WORK_ORDERS if w["id"] == wo_id), None)
        if wo is None:
            raise HTTPException(404, "unknown work order")
        wo.update(status="APPROVED", planned_start=int(body.planned_start),
                  planned_label=clock(body.planned_start), approver=body.approver)
        _save_wos()
    return wo


@app.post("/api/reset")
def reset():
    with LOCK:
        WORK_ORDERS.clear()
        _save_wos()
    return {"ok": True}
