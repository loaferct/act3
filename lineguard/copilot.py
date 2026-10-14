"""Maintenance copilot: turns a detector alert into an explanation and a work order an
operator can act on, grounded in the plant's own manuals and work-order history.

The model (served on AMD GPUs) only writes the explanation and plan. Detection, the
remaining-life estimate and the cost numbers come from deterministic code, so the
model cannot invent them. Every claim must cite a source; uncited sources are dropped,
and a lockout/tagout step is enforced for any hands-on job.
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from .knowledge import KnowledgeBase
from .llm import LLM

SYSTEM = """You are LineGuard, a maintenance copilot for a manufacturing plant. You help shift operators
and maintenance planners decide what to do about early equipment warnings.

Rules:
- Use ONLY the sensor evidence and the numbered SOURCES given to you. Cite sources inline like [KB3].
- Never invent part numbers, limits, durations or procedures that are not in the sources.
- If the sources do not cover something, say so plainly.
- Write for a shift operator: short, concrete, no jargon without explanation.
- Physical work on a machine always starts with lockout/tagout per SOP-LOTO-01.
"""

ANALYZE = """ALERT EVIDENCE (computed by the detector, treat as facts):
{evidence}

SOURCES:
{sources}

Return ONLY a JSON object with these keys. Be brief: operators read this on a phone.
{{
 "headline": "one sentence (max 25 words): what is happening and how urgent",
 "explanation": "max 2 sentences: what the sensor pattern means physically, with citations",
 "likely_cause": "most likely root cause with citation (max 15 words)",
 "check_first": ["max 2 quick checks to confirm, each with citation"],
 "actions": [{{"step": "max 15 words", "who": "operator|mechanic|electrician|planner", "when": "now, this shift, or 'before ' + the repair_deadline from the evidence"}}],
 "parts": ["part numbers from the sources"],
 "safety": ["max 2 precautions with citation"],
 "job_duration_h": number,
 "priority": "P1|P2|P3",
 "if_ignored": "one sentence with citation"
}}
Use at most 4 actions. Write real values only: never copy angle-bracket placeholders or repeat a word."""

ASK = """CURRENT PLANT STATUS (from the monitoring system):
{status}

SOURCES:
{sources}

QUESTION from {role}: {question}

Answer in at most 150 words using only the status and sources. Cite sources like [KB3]. If the answer is not
in the sources, say what is missing and who to ask."""


def _fmt_sources(chunks) -> str:
    return "\n\n".join(f"[{c.id}] {c.doc} - {c.section}\n{c.text}" for c in chunks)


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


class Copilot:
    def __init__(self, kb: KnowledgeBase, llm: LLM):
        self.kb, self.llm = kb, llm
        self._cache: dict = {}
        self._pending: dict[tuple, Future] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2)   # shared endpoint: ~20 requests/min per team

    @staticmethod
    def _key(asset: dict, state: dict) -> tuple:
        return asset["id"], state["level"], (state.get("diagnosis") or {}).get("mode")

    def prefetch(self, asset: dict, state: dict, clock: str, deadline: str | None = None):
        """Start the analysis in the background when an alert opens or escalates, so it is ready on click."""
        if not self.llm.enabled:
            return
        key = self._key(asset, state)
        with self._lock:
            if key in self._cache or key in self._pending:
                return
            self._pending[key] = self._pool.submit(self._run, asset, state, clock, key, deadline)

    # ------------------------------------------------------------------ alert analysis
    def analyze(self, asset: dict, state: dict, clock: str, deadline: str | None = None) -> dict:
        key = self._key(asset, state)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            fut = self._pending.get(key)
        if fut is not None:
            return fut.result()
        return self._run(asset, state, clock, key, deadline)

    def _run(self, asset: dict, state: dict, clock: str, key: tuple, deadline: str | None = None) -> dict:
        try:
            out = self._analyze(asset, state, clock, deadline)
            with self._lock:
                self._cache[key] = out
            return out
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def _analyze(self, asset: dict, state: dict, clock: str, deadline: str | None = None) -> dict:
        diag = state.get("diagnosis") or {}

        evidence = self._evidence(asset, state, clock, deadline)
        query = f"{asset['kind_label']} {asset['id']} {diag.get('label', '')} " + " ".join(
            d["label"] for d in state["deviations"][:2]) + " procedure parts lockout"
        chunks = self.kb.search(query, k=5)
        history = [c for c in self.kb.search(f"{asset['id']} {diag.get('label', '')} unplanned downtime root cause", k=8)
                   if c.doc.startswith("Maintenance Work Order History") and asset["id"] in c.section][:2]
        loto = self.kb.search("lockout tagout steps isolate energy", k=1)
        chunks += [c for c in history + loto if c not in chunks]

        result = None
        meta = {"mode": "template", "reason": "LLM endpoint not configured"}
        if self.llm.enabled:
            try:
                text, meta = self.llm.chat([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": ANALYZE.format(evidence=json.dumps(evidence, indent=1),
                                                               sources=_fmt_sources(chunks))}],
                    max_tokens=700, json_mode=True)
                meta["mode"] = "llm"
                result = _parse_json(text)
                if result is None:
                    result = {"headline": text[:300], "explanation": text}
            except Exception as e:  # keep the operator view working if the endpoint is down
                meta = {"mode": "template", "reason": f"LLM call failed: {type(e).__name__}"}
        if result is None:
            result = self._template(asset, state, chunks)

        result = self._guard(result, chunks)
        out = {"analysis": result, "evidence": evidence, "meta": meta,
               "sources": [{"id": c.id, "doc": c.doc, "section": c.section, "text": c.text} for c in chunks
                           if c.id in result.get("citations", [])]}
        return out

    def _evidence(self, asset, state, clock, deadline=None):
        diag = state.get("diagnosis") or {}
        devs = [{k: d[k] for k in ("label", "unit", "normal", "now", "alarm", "trip")} | {"deviation_sigma": d["z"]}
                for d in state["deviations"] if abs(d["z"]) >= 2][:4]
        return {
            "plant_time": clock, "asset": f"{asset['id']} {asset['name']} ({asset['kind_label']})",
            "area": asset["area"], "alert_level": state["level"], "health_score": state["health"],
            "abnormal_sensors": devs,
            "pattern_match": {"failure_mode": diag.get("label"), "similarity": diag.get("match"),
                              "alternatives": [a["label"] for a in diag.get("alternatives", [])]},
            "estimated_hours_to_trip": state.get("rul_h"),
            "repair_deadline": deadline or "no trend toward a trip yet",
            "fixed_alarm_limit_reached": any((d["now"] - d["alarm"]) * d["direction"] >= 0 for d in state["deviations"]),
        }

    def _guard(self, r: dict, chunks) -> dict:
        valid = {c.id for c in chunks}
        cited = set(re.findall(r"KB\d+", json.dumps(r)))
        r["citations"] = sorted(cited & valid, key=lambda s: int(s[2:]))
        r["unverified_citations"] = sorted(cited - valid)
        safety = r.get("safety") or []
        if isinstance(safety, str):
            safety = [safety]
        loto = next((c.id for c in chunks if "LOTO" in c.doc), None)
        if not any(w in x.lower() for x in safety for w in ("lock", "loto", "isolat")):
            safety.insert(0, f"Isolate and lock out the machine per SOP-LOTO-01 before any hands-on work [{loto}]."
                          if loto else "Isolate and lock out the machine per SOP-LOTO-01 before any hands-on work.")
            if loto and loto not in r["citations"]:
                r["citations"].append(loto)
        r["safety"] = safety
        r.setdefault("actions", [])
        return r

    def _template(self, asset, state, chunks) -> dict:
        """Deterministic fallback so the product still works offline - clearly labelled in the UI."""
        diag = state.get("diagnosis") or {}
        top = state["deviations"][0]
        rul = state.get("rul_h")
        c0 = chunks[0].id if chunks else None
        urgent = rul is not None and rul < 8
        return {
            "headline": f"{asset['id']} {top['label'].lower()} is drifting away from normal"
                        + (f" - about {rul:.0f} h to trip at the current rate." if rul else "."),
            "explanation": f"{top['label']} is {top['now']} {top['unit']} vs {top['normal']} {top['unit']} normal "
                           f"for this time of shift ({top['z']:+.0f} sigma). The pattern matches "
                           f"'{diag.get('label', 'an unknown fault')}' ({diag.get('match', 0):.0%} similarity)."
                           + (f" See [{c0}]." if c0 else ""),
            "likely_cause": diag.get("label", "unknown"),
            "check_first": [f"Read {c.doc} - {c.section} [{c.id}]" for c in chunks[:2]],
            "actions": [{"step": "Confirm the reading on the local gauge / handheld meter", "who": "operator", "when": "now"},
                        {"step": "Raise a work order and reserve parts", "who": "planner", "when": "this shift"},
                        {"step": "Schedule the repair in the next planned stop", "who": "planner",
                         "when": "now" if urgent else "before the estimated trip time"}],
            "parts": [], "safety": [], "job_duration_h": None,
            "priority": "P1" if urgent else "P2" if state["level"] in ("WARNING", "CRITICAL") else "P3",
            "if_ignored": f"Unplanned trip in about {rul:.0f} h." if rul else "Unknown - keep monitoring.",
        }

    # ------------------------------------------------------------------ free questions
    def ask(self, question: str, status: list[dict], role: str = "operator") -> dict:
        chunks = self.kb.search(question, k=5)
        if not self.llm.enabled:
            return {"answer": "The AI model is not connected (set LLM_BASE_URL / LLM_MODEL). Most relevant manual sections:",
                    "sources": [{"id": c.id, "doc": c.doc, "section": c.section, "text": c.text} for c in chunks],
                    "meta": {"mode": "search-only"}}
        text, meta = self.llm.chat([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ASK.format(status=json.dumps(status, indent=1), sources=_fmt_sources(chunks),
                                                   role=role, question=question)}], max_tokens=400)
        meta["mode"] = "llm"
        used = set(re.findall(r"KB\d+", text))
        return {"answer": text, "meta": meta,
                "sources": [{"id": c.id, "doc": c.doc, "section": c.section, "text": c.text} for c in chunks if c.id in used]}
