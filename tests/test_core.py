import json

import httpx
import numpy as np
import pytest

from lineguard.detector import alert_episodes, fit
from lineguard.simulator import AssetSpec, simulate_plant


@pytest.fixture(scope="module")
def plant():
    return {a.spec.id: (a, fit(a)) for a in simulate_plant(seed=7)}


def test_faults_detected_well_before_fixed_alarm(plant):
    for a, m in plant.values():
        if a.trip_time is None:
            continue
        assert m.first_watch is not None and a.fault_onset <= m.first_watch < a.trip_time
        assert m.static_alarm - m.first_watch > 12 * 60, a.spec.id   # >12 h earlier than today's alarms


def test_healthy_machines_stay_quiet(plant):
    for a, m in plant.values():
        if a.spec.fault is None:
            assert alert_episodes(m) == []


def test_detector_is_causal():
    """Changing the future must not change the state at time t."""
    spec = [AssetSpec("P-101", "pump", "pump", "x", "bearing_wear", 3.0, 2.0)]
    a = simulate_plant(seed=1, plant=spec)[0]
    t = 4 * 1440
    before = fit(a).state(t)
    a.values[t + 1:] += 50
    a.trip_time = None
    after = fit(a).state(t)
    assert before["level"] == after["level"] and before["health"] == after["health"]


def test_diagnosis_matches_injected_fault(plant):
    for a, m in plant.values():
        if a.trip_time is not None:
            assert m.diagnose(a.trip_time - 120)["mode"] == a.spec.fault


def test_copilot_with_llm_endpoint(monkeypatch):
    """Full alert -> retrieval -> LLM -> guarded plan, against a fake OpenAI-compatible server."""
    monkeypatch.setenv("LLM_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["body"] = json
        content = '```json\n{"headline": "Bearing wearing [KB7]", "explanation": "x [KB7] [KB999]", ' \
                  '"actions": [{"step": "replace bearing", "who": "mechanic", "when": "this shift"}], "safety": []}\n```'
        return httpx.Response(200, json={"model": "test-model", "choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 120}},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    import importlib

    import lineguard.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    api.reset()
    r = c.post("/api/asset/P-101/analyze", params={"t": 6000}).json()
    a = r["analysis"]
    assert r["meta"]["mode"] == "llm" and r["meta"]["tokens_per_s"]
    assert "KB7" in a["citations"] and a["unverified_citations"] == ["KB999"]
    assert "lock" in a["safety"][0].lower()                      # LOTO enforced
    assert "SOURCES" in seen["body"]["messages"][1]["content"]

    # Approving a work order before the trip turns the breakdown into a planned repair
    wo = c.post("/api/workorders", json={"asset_id": "P-101", "t": 6000, "title": "bearing"}).json()
    c.post(f"/api/workorders/{wo['id']}/approve", json={"planned_start": 6000 + 600})
    end = c.get("/api/plant", params={"t": 10079}).json()
    p101 = next(x for x in end["assets"] if x["id"] == "P-101")
    assert p101["level"] == "OK" and p101["repaired_by"] == wo["id"]
    assert end["kpi"]["trips_avoided"] == 1 and end["kpi"]["cost_saved"] > 0
    api.reset()
