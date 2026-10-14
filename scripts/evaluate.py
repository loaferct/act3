"""Measure LineGuard against today's fixed alarm limits on unseen random fault scenarios.

    python -m scripts.evaluate --runs 50
"""
import argparse
import json
import statistics as st

import numpy as np

from lineguard.detector import BASELINE_MIN, alert_episodes, fit
from lineguard.impact import compare
from lineguard.simulator import MIN_PER_DAY, random_plant, simulate_plant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()

    faults = detected = false_alerts = 0
    healthy_days = 0.0
    ours, static, saved_h, saved_cost = [], [], 0.0, 0
    rul_ratio = {48: [], 24: [], 12: []}   # predicted / actual hours to trip
    for r in range(args.runs):
        rng = np.random.default_rng(args.seed + r)
        for a in simulate_plant(seed=args.seed + r, plant=random_plant(rng)):
            m = fit(a)
            eps = alert_episodes(m)
            end = a.trip_time if a.trip_time is not None else len(a.values)
            onset = a.fault_onset if a.fault_onset is not None else end
            false_alerts += sum(1 for s, _ in eps if s < onset)
            healthy_days += max(0, min(onset, end) - BASELINE_MIN) / MIN_PER_DAY
            if a.trip_time is None:
                continue
            faults += 1
            hit = next((s for s, _ in eps if onset <= s < a.trip_time), None)
            lead = None if hit is None else (a.trip_time - hit) / 60
            slead = None if m.static_alarm is None else (a.trip_time - m.static_alarm) / 60
            for h in rul_ratio:
                t = a.trip_time - h * 60
                if t >= BASELINE_MIN and m.active[t]:
                    est = m.rul_hours(t)
                    if est:
                        rul_ratio[h].append(est / h)
            if lead is not None:
                detected += 1
                ours.append(lead)
            static.append(slead or 0.0)
            c = compare(a.spec.kind, lead, slead)
            saved_h += c["downtime_saved_h"]
            saved_cost += c["cost_saved"]

    res = {
        "scenarios": args.runs,
        "faults_reaching_trip": faults,
        "detected_before_trip": detected,
        "recall": round(detected / max(faults, 1), 3),
        "false_alerts_per_machine_month": round(false_alerts / max(healthy_days, 1e-9) * 30, 3),
        "median_warning_h_lineguard": round(st.median(ours), 1) if ours else None,
        "median_warning_h_fixed_alarm": round(st.median(static), 1) if static else None,
        "time_to_trip_estimate_vs_actual": {f"{h}h_before_trip": {"median": round(st.median(v), 2),
                                            "p10": round(float(np.percentile(v, 10)), 2),
                                            "p90": round(float(np.percentile(v, 90)), 2), "n": len(v)}
                                            for h, v in rul_ratio.items() if v},
        "downtime_saved_h_per_fault": round(saved_h / max(faults, 1), 1),
        "cost_saved_per_fault_eur": round(saved_cost / max(faults, 1)),
    }
    print(json.dumps(res, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
