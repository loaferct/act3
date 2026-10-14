"""Measure the copilot on your AMD endpoint: latency and generation speed for real alert analyses.

    python -m scripts.benchmark_llm --n 6
"""
import argparse
import json
import statistics as st

from lineguard.copilot import Copilot
from lineguard.detector import fit
from lineguard.knowledge import KnowledgeBase
from lineguard.llm import LLM
from lineguard.simulator import simulate_plant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    llm = LLM()
    if not llm.enabled:
        raise SystemExit("Set LLM_BASE_URL and LLM_MODEL first (see .env.example)")
    kb = KnowledgeBase("knowledge_base")
    cases = []
    for a in simulate_plant():
        if a.trip_time is None:
            continue
        m = fit(a)
        for hours_before in (36, 20, 6):
            t = a.trip_time - hours_before * 60
            s = m.state(t)
            if s["level"] not in ("OK", "LEARNING"):
                cases.append((a, s, t))
    lat, tps, ok = [], [], 0
    for a, s, t in cases[: args.n]:
        cop = Copilot(kb, llm)   # fresh cache
        info = {"id": a.spec.id, "name": a.spec.name, "kind_label": a.kind.label, "area": a.spec.area}
        r = cop.analyze(info, s, f"t={t}")
        m = r["meta"]
        if m.get("mode") == "llm":
            lat.append(m["latency_s"])
            tps.append(m["tokens_per_s"] or 0)
            ok += bool(r["analysis"].get("citations")) and "actions" in r["analysis"]
            print(f"{a.spec.id} {s['level']:8s} {m['latency_s']:5.1f}s {m['tokens_per_s']} tok/s  {r['analysis'].get('headline', '')[:90]}")
        else:
            print(a.spec.id, "failed:", m.get("reason"))
    if lat:
        print(json.dumps({"model": llm.model, "hardware": llm.hardware, "calls": len(lat),
                          "valid_grounded_plans": ok, "median_latency_s": st.median(lat),
                          "median_tokens_per_s": st.median(tps)}, indent=2))


if __name__ == "__main__":
    main()
