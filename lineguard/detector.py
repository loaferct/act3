"""Early-warning detector.

Plants already alarm when a sensor crosses its fixed limit - usually hours before the trip,
too late to order parts or plan the stop. LineGuard learns each machine's normal behaviour
(per hour-of-shift) from a healthy baseline and flags small, persistent drifts days earlier.

Everything here is causal (state at minute t only uses data up to t) and explainable:
which sensor moved, how far from normal, which known failure mode it matches, and how
long until the trip limit at the current rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .simulator import MIN_PER_DAY, AssetData

BASELINE_MIN = 2 * MIN_PER_DAY     # learn "normal" from the first 2 days
ALPHA = 0.04                       # EWMA smoothing (~25 min memory)
WATCH_Z = 3.5                      # smoothed deviation (in noise units) that opens an alert
WARNING_Z = 8.0
CLEAR_Z = 2.0                      # alert closes once deviation falls back below this
PERSIST_MIN = 15                   # deviation must hold this long (filters glitches)
RUL_WINDOW = 2880                  # max minutes of drift history used for the time-to-trip fit
LEVELS = ["OK", "WATCH", "WARNING", "CRITICAL", "TRIPPED"]


def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    zi = (1 - alpha) * x[:1]
    out, _ = lfilter([alpha], [1, -(1 - alpha)], x, axis=0, zi=zi)
    return out


def _hysteresis(open_cond: np.ndarray, clear_cond: np.ndarray) -> np.ndarray:
    """Open after PERSIST_MIN minutes of open_cond; close after PERSIST_MIN minutes of clear_cond."""
    out = np.zeros(len(open_cond), dtype=bool)
    on, run = False, 0
    for i in range(len(open_cond)):
        if not on:
            run = run + 1 if open_cond[i] else 0
            if run >= PERSIST_MIN:
                on, run = True, 0
        else:
            run = run + 1 if clear_cond[i] else 0
            if run >= PERSIST_MIN:
                on, run = False, 0
        out[i] = on
    return out


def _run_length(mask: np.ndarray) -> np.ndarray:
    """Consecutive True count ending at each index."""
    out = np.zeros(len(mask), dtype=int)
    c = 0
    for i, m in enumerate(mask):
        c = c + 1 if m else 0
        out[i] = c
    return out


@dataclass
class AssetModel:
    data: AssetData
    profile: np.ndarray      # (24, S) normal value by hour of day
    scale: np.ndarray        # (S,) robust noise scale of residuals
    z: np.ndarray            # (T, S) smoothed deviation in noise units
    resid_s: np.ndarray      # (T, S) smoothed residual in engineering units
    active: np.ndarray       # (T,) bool: alert open (with hysteresis)
    static_alarm: int | None  # first minute a fixed DCS alarm limit fired (3-sample debounce)
    first_watch: int | None

    # ------------------------------------------------------------------ state at time t
    def state(self, t: int) -> dict:
        d = self.data
        sensors = d.sensors
        t = int(np.clip(t, 0, len(d.values) - 1))
        if d.trip_time is not None and t >= d.trip_time:
            return {"level": "TRIPPED", "health": 0, "tripped_at": d.trip_time,
                    "deviations": self._deviations(d.trip_time - 1), "diagnosis": self.diagnose(d.trip_time - 1),
                    "rul_h": 0.0, "alert_since": self.first_watch}
        if t < BASELINE_MIN:
            return {"level": "LEARNING", "health": 100, "deviations": self._deviations(t),
                    "diagnosis": None, "rul_h": None, "alert_since": None}

        m = float(np.max(np.abs(self.z[t])))
        watching = bool(self.active[t])
        level = "OK"
        rul_h = None
        diag = None
        if watching:
            level = "WATCH"
            diag = self.diagnose(t)
            rul_h = self.rul_hours(t)
            if m >= WARNING_Z or (rul_h is not None and rul_h < 48):
                level = "WARNING"
            raw = d.values[t]
            past_alarm = any((raw[j] - s.alarm) * s.direction >= 0 for j, s in enumerate(sensors))
            if past_alarm or (rul_h is not None and rul_h < 8):
                level = "CRITICAL"
        health = int(round(100 * np.exp(-max(0.0, m - 1.5) / 10)))
        since = self.first_watch if (self.first_watch is not None and self.first_watch <= t and watching) else None
        return {"level": level, "health": health, "deviations": self._deviations(t),
                "diagnosis": diag, "rul_h": rul_h, "alert_since": since}

    def _deviations(self, t: int) -> list[dict]:
        d = self.data
        hour = (t // 60) % 24
        out = []
        for j, s in enumerate(d.sensors):
            out.append({"sensor": s.key, "label": s.label, "unit": s.unit,
                        "normal": round(float(self.profile[hour, j]), 2),
                        "now": round(float(d.values[t, j]), 2),
                        "z": round(float(self.z[t, j]), 1),
                        "alarm": s.alarm, "trip": s.trip, "direction": s.direction})
        return sorted(out, key=lambda r: -abs(r["z"]))

    def diagnose(self, t: int) -> dict | None:
        """Match the deviation pattern against known failure-mode signatures (cosine similarity)."""
        d = self.data
        obs = np.clip(self.z[t], -30, 30)
        if np.linalg.norm(obs) < 1e-6:
            return None
        noise = np.array([s.noise for s in d.sensors])
        cands = []
        for mode in d.kind.faults.values():
            sig = np.zeros(len(d.sensors))
            for j, s in enumerate(d.sensors):
                if s.key in mode.effects:
                    mag, _ = mode.effects[s.key]
                    sig[j] = np.sign(mag) * min(abs(mag) / noise[j], 30)
            sim = float(obs @ sig / (np.linalg.norm(obs) * np.linalg.norm(sig)))
            cands.append({"mode": mode.key, "label": mode.label, "match": round(max(sim, 0.0), 2)})
        cands.sort(key=lambda c: -c["match"])
        best = cands[0]
        if best["match"] < 0.5:
            best = {"mode": "unknown", "label": "Unrecognised deviation pattern", "match": best["match"]}
        return {**best, "alternatives": cands[1:3]}

    def rul_hours(self, t: int) -> float | None:
        """Hours until the most-deviating sensor reaches its trip limit, extrapolating its recent trend.

        Degradation accelerates, so a straight line through recent data is optimistic (it said ~1.4x
        the real time left a day before a trip). A quadratic fitted from where this drift began follows
        the curve (24 h before a trip: median 0.95x, 10-90% 0.88-1.02x on the evaluation scenarios);
        a straight line over the last 3 h is the fallback when the curve never reaches the limit.
        """
        d = self.data
        best = None
        hour = (t // 60) % 24
        for j, s in enumerate(d.sensors):
            toward = self.z[:t + 1, j] * s.direction
            if toward[t] < WATCH_Z:                      # only sensors drifting toward their limit
                continue
            normal = np.nonzero(toward < 1.0)[0]         # last time this sensor was near normal
            lo = max(BASELINE_MIN, t - RUL_WINDOW, min(int(normal[-1]) if len(normal) else 0, t - 120))
            if t - lo < 30:
                continue
            xs = (np.arange(lo, t + 1) - t) / 60.0       # hours, 0 = now
            y = self.resid_s[lo:t + 1, j]
            target = s.trip - self.profile[hour, j]        # trip limit in residual units
            remaining = None
            if t - lo >= 120:
                a, b, c = np.polyfit(xs, y, 2)
                roots = [r.real for r in np.roots([a, b, c - target]) if abs(r.imag) < 1e-9 and r.real > 0]
                remaining = min(roots) if roots else None
            if remaining is None:
                k = min(len(y), 180)
                slope = np.polyfit(xs[-k:], y[-k:], 1)[0]
                if slope * s.direction > 0:
                    remaining = (target - y[-1]) / slope
            if remaining is not None and remaining > 0 and (best is None or remaining < best):
                best = remaining
        return None if best is None else round(float(min(best, 24 * 30)), 1)


def fit(data: AssetData) -> AssetModel:
    v = data.values
    T, S = v.shape
    hours = (np.arange(T) // 60) % 24
    base = v[:BASELINE_MIN]
    bh = hours[:BASELINE_MIN]
    profile = np.vstack([np.median(base[bh == h], axis=0) for h in range(24)])
    resid = v - profile[hours]
    mad = np.median(np.abs(resid[:BASELINE_MIN] - np.median(resid[:BASELINE_MIN], axis=0)), axis=0)
    floor = np.array([s.noise for s in data.sensors]) * 0.5
    scale = np.maximum(1.4826 * mad, floor)

    # Clip single-sample spikes before smoothing so glitches cannot open alerts
    zr = np.clip(resid / scale, -25, 25)
    z = _ewma(zr, ALPHA)
    resid_s = _ewma(resid, ALPHA)
    dirs = np.array([s.direction for s in data.sensors])
    toward_limit = (z * dirs)   # we only alert on drift towards a protection limit
    m = np.max(toward_limit, axis=1)
    cond = (m >= WATCH_Z) & (np.arange(T) >= BASELINE_MIN)
    if data.trip_time is not None:
        cond[data.trip_time:] = False
    active = _hysteresis(cond, m < CLEAR_Z)
    hits = np.nonzero(active)[0]
    first_watch = int(hits[0]) if len(hits) else None

    # Today's practice: fixed alarm limits with a 3-sample debounce
    alarm = np.array([s.alarm for s in data.sensors])
    over = np.any((v - alarm) * dirs >= 0, axis=1)
    if data.trip_time is not None:
        over[data.trip_time:] = False
    deb = _run_length(over)
    sa = np.nonzero(deb >= 3)[0]
    static_alarm = int(sa[0]) if len(sa) else None

    return AssetModel(data, profile, scale, z, resid_s, active, static_alarm, first_watch)


def alert_episodes(model: AssetModel) -> list[tuple[int, int]]:
    """(start, end) minute ranges where an alert was open."""
    a = model.active.astype(int)
    edges = np.diff(np.concatenate([[0], a, [0]]))
    return list(zip(np.nonzero(edges == 1)[0].tolist(), np.nonzero(edges == -1)[0].tolist()))
