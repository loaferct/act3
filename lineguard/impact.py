"""Business impact: what earlier warning is worth.

A warning only saves money if it arrives early enough to act: order the part, book the
technician and do the job in a planned stop instead of after a trip. The assumptions are
explicit and editable - typical mid-size plant figures, not measured values.
"""
from __future__ import annotations

ASSUMPTIONS = {
    # kind: downtime cost/h of the line it stops, hours lost if it trips unplanned,
    #       hours needed for the same repair when planned, extra damage if run to failure,
    #       warning needed to plan it (part lead time + next planned stop, from the manuals)
    "pump":       {"cost_per_h": 3500, "unplanned_h": 9.0, "planned_h": 2.5, "secondary_damage": 6000, "lead_needed_h": 24},
    "compressor": {"cost_per_h": 5000, "unplanned_h": 7.0, "planned_h": 2.0, "secondary_damage": 9000, "lead_needed_h": 8},
    "press":      {"cost_per_h": 8000, "unplanned_h": 12.0, "planned_h": 4.0, "secondary_damage": 15000, "lead_needed_h": 24},
    "conveyor":   {"cost_per_h": 4000, "unplanned_h": 5.0, "planned_h": 1.5, "secondary_damage": 2500, "lead_needed_h": 8},
    "cnc":        {"cost_per_h": 1500, "unplanned_h": 30.0, "planned_h": 8.0, "secondary_damage": 18000, "lead_needed_h": 120},
}
CURRENCY = "€"


def outcome(kind: str, lead_h: float | None) -> dict:
    """Downtime and cost of one developing fault, given how many hours of warning there were."""
    a = ASSUMPTIONS[kind]
    unplanned = a["unplanned_h"] * a["cost_per_h"] + a["secondary_damage"]
    if lead_h is not None and lead_h >= a["lead_needed_h"]:
        return {"planned": True, "downtime_h": a["planned_h"], "cost": a["planned_h"] * a["cost_per_h"]}
    return {"planned": False, "downtime_h": a["unplanned_h"], "cost": unplanned}


def compare(kind: str, lineguard_lead_h: float | None, static_lead_h: float | None) -> dict:
    ours, base = outcome(kind, lineguard_lead_h), outcome(kind, static_lead_h)
    return {
        "lineguard": ours, "static_alarm": base,
        "downtime_saved_h": round(base["downtime_h"] - ours["downtime_h"], 1),
        "cost_saved": round(base["cost"] - ours["cost"]),
    }
