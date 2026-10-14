"""Plant telemetry simulator.

Generates 1-minute sensor data for a small production line. Healthy machines follow a
shift-load cycle plus noise and the odd sensor glitch. Faulty machines degrade along a
physically-motivated signature (e.g. bearing wear: vibration first, temperature later)
until the protection limit trips the machine. Ground truth (fault onset, trip time) is
kept so detection quality and lead time can be measured, not just claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_PER_DAY = 1440


@dataclass(frozen=True)
class Sensor:
    key: str
    label: str
    unit: str
    nominal: float
    noise: float          # std of measurement noise
    daily_amp: float      # swing caused by the shift-load cycle
    alarm: float          # DCS alarm limit (what plants alert on today)
    trip: float           # protection trip limit -> unplanned stop
    direction: int        # +1: high limit, -1: low limit
    stopped: float        # value when the machine is stopped


@dataclass(frozen=True)
class FaultMode:
    key: str
    label: str
    effects: dict          # sensor key -> (magnitude at severity 1, exponent)
    trip_sensor: str


@dataclass(frozen=True)
class AssetKind:
    key: str
    label: str
    sensors: tuple
    faults: dict


def _s(*a):
    return Sensor(*a)


KINDS = {
    "pump": AssetKind("pump", "Centrifugal pump", (
        _s("vibration", "Bearing vibration", "mm/s", 2.2, 0.18, 0.15, 7.1, 11.0, +1, 0.2),
        _s("bearing_temp", "Bearing temperature", "°C", 55.0, 0.6, 2.0, 85.0, 95.0, +1, 25.0),
        _s("motor_current", "Motor current", "A", 42.0, 0.5, 2.5, 52.0, 56.0, +1, 0.0),
        _s("discharge_pressure", "Discharge pressure", "bar", 6.2, 0.05, 0.1, 4.6, 4.0, -1, 0.0),
    ), {
        "bearing_wear": FaultMode("bearing_wear", "Drive-end bearing wear", {
            "vibration": (10.0, 2.0), "bearing_temp": (45.0, 3.0), "motor_current": (2.5, 2.5)}, "vibration"),
        "cavitation": FaultMode("cavitation", "Cavitation / clogged suction strainer", {
            "discharge_pressure": (-2.6, 1.5), "vibration": (3.0, 1.3), "motor_current": (-4.0, 1.5)},
            "discharge_pressure"),
    }),
    "compressor": AssetKind("compressor", "Screw air compressor", (
        _s("vibration", "Airend vibration", "mm/s", 2.8, 0.2, 0.2, 7.1, 11.0, +1, 0.2),
        _s("discharge_temp", "Discharge temperature", "°C", 82.0, 0.7, 3.0, 105.0, 112.0, +1, 30.0),
        _s("motor_current", "Motor current", "A", 118.0, 1.2, 6.0, 140.0, 150.0, +1, 0.0),
        _s("outlet_pressure", "Outlet pressure", "bar", 7.5, 0.04, 0.08, 6.6, 6.2, -1, 0.0),
    ), {
        "cooler_fouling": FaultMode("cooler_fouling", "Oil cooler fouling", {
            "discharge_temp": (34.0, 1.6), "motor_current": (5.0, 1.5), "outlet_pressure": (-0.2, 1.5)},
            "discharge_temp"),
    }),
    "press": AssetKind("press", "Hydraulic press", (
        _s("vibration", "Pump vibration", "mm/s", 3.0, 0.2, 0.2, 7.1, 11.0, +1, 0.2),
        _s("oil_temp", "Hydraulic oil temperature", "°C", 48.0, 0.5, 2.0, 65.0, 70.0, +1, 25.0),
        _s("motor_current", "Motor current", "A", 64.0, 0.8, 3.0, 78.0, 84.0, +1, 0.0),
        _s("system_pressure", "System pressure", "bar", 210.0, 1.5, 3.0, 185.0, 175.0, -1, 0.0),
    ), {
        "pump_wear": FaultMode("pump_wear", "Hydraulic pump internal leakage", {
            "system_pressure": (-40.0, 1.7), "oil_temp": (14.0, 1.5), "vibration": (1.5, 2.0)},
            "system_pressure"),
    }),
    "conveyor": AssetKind("conveyor", "Belt conveyor drive", (
        _s("vibration", "Gearbox vibration", "mm/s", 1.8, 0.15, 0.1, 7.1, 11.0, +1, 0.2),
        _s("motor_temp", "Motor winding temperature", "°C", 62.0, 0.6, 2.5, 110.0, 125.0, +1, 25.0),
        _s("motor_current", "Motor current", "A", 21.0, 0.3, 1.5, 27.0, 30.0, +1, 0.0),
        _s("belt_speed", "Belt speed", "m/s", 1.6, 0.01, 0.0, 1.45, 1.35, -1, 0.0),
    ), {
        "belt_slip": FaultMode("belt_slip", "Belt slip / low belt tension", {
            "belt_speed": (-0.3, 1.4), "motor_temp": (25.0, 1.6), "vibration": (2.0, 1.5)}, "belt_speed"),
    }),
    "cnc": AssetKind("cnc", "CNC machining centre", (
        _s("spindle_vibration", "Spindle vibration", "mm/s", 1.2, 0.1, 0.15, 4.5, 7.0, +1, 0.1),
        _s("spindle_temp", "Spindle temperature", "°C", 38.0, 0.4, 2.5, 60.0, 68.0, +1, 22.0),
        _s("spindle_load", "Spindle load", "%", 55.0, 2.0, 8.0, 90.0, 100.0, +1, 0.0),
        _s("coolant_temp", "Coolant temperature", "°C", 24.0, 0.3, 1.0, 32.0, 35.0, +1, 22.0),
    ), {
        "spindle_bearing": FaultMode("spindle_bearing", "Spindle bearing preload loss", {
            "spindle_vibration": (6.5, 2.0), "spindle_temp": (32.0, 2.6)}, "spindle_vibration"),
    }),
}


@dataclass
class AssetSpec:
    id: str
    name: str
    kind: str
    area: str
    fault: str | None = None
    onset_day: float = 0.0
    duration_days: float = 1.0


@dataclass
class AssetData:
    spec: AssetSpec
    kind: AssetKind
    values: np.ndarray                 # (T, S)
    fault_onset: int | None = None     # minute index
    trip_time: int | None = None       # minute index where protection trips
    glitches: list = field(default_factory=list)

    @property
    def sensors(self):
        return self.kind.sensors


DEFAULT_PLANT = [
    AssetSpec("P-101", "Cooling-water pump", "pump", "Utilities", "bearing_wear", 3.0, 2.6),
    AssetSpec("P-102", "Coolant transfer pump", "pump", "Machining"),
    AssetSpec("C-201", "Main air compressor", "compressor", "Utilities", "cooler_fouling", 2.5, 3.4),
    AssetSpec("M-501", "800 t hydraulic press", "press", "Forming", "pump_wear", 4.0, 2.0),
    AssetSpec("CV-401", "Line 4 transfer conveyor", "conveyor", "Assembly"),
    AssetSpec("CNC-301", "5-axis machining centre", "cnc", "Machining"),
]


def simulate_asset(spec: AssetSpec, days: float, rng: np.random.Generator) -> AssetData:
    kind = KINDS[spec.kind]
    T = int(days * MIN_PER_DAY)
    t = np.arange(T)
    phase = rng.uniform(0, 2 * np.pi)
    load = np.sin(2 * np.pi * t / MIN_PER_DAY + phase)                 # shift-load cycle
    load += 0.3 * np.sin(2 * np.pi * t / (MIN_PER_DAY / 3) + phase)    # shift changes

    clean = np.empty((T, len(kind.sensors)))
    for j, s in enumerate(kind.sensors):
        clean[:, j] = s.nominal + s.daily_amp * load

    onset = trip = None
    if spec.fault:
        mode = kind.faults[spec.fault]
        onset = int(spec.onset_day * MIN_PER_DAY)
        dur = spec.duration_days * MIN_PER_DAY
        sev = np.clip((t - onset) / dur, 0, None)
        for j, s in enumerate(kind.sensors):
            if s.key in mode.effects:
                mag, p = mode.effects[s.key]
                clean[:, j] += mag * sev ** p
        k = [s.key for s in kind.sensors].index(mode.trip_sensor)
        ts = kind.sensors[k]
        crossed = np.nonzero((clean[:, k] - ts.trip) * ts.direction >= 0)[0]
        crossed = crossed[crossed >= onset]
        trip = int(crossed[0]) if len(crossed) else None

    noise = np.array([s.noise for s in kind.sensors])
    values = clean + rng.normal(size=clean.shape) * noise

    # A few single-sample sensor glitches (bad comms, EMI) on every machine
    glitches = []
    for _ in range(rng.integers(2, 6)):
        i, j = int(rng.integers(0, T)), int(rng.integers(0, len(kind.sensors)))
        s = kind.sensors[j]
        values[i, j] = s.nominal + s.direction * rng.uniform(8, 20) * s.noise
        glitches.append((i, j))

    if trip is not None:   # machine stopped by protection: values coast down to rest
        tail = T - trip
        decay = np.exp(-np.arange(tail) / 20.0)[:, None]
        stopped = np.array([s.stopped for s in kind.sensors])
        values[trip:] = stopped + (values[trip] - stopped) * decay + rng.normal(size=(tail, len(noise))) * noise * 0.2

    return AssetData(spec, kind, values, onset, trip, glitches)


def simulate_plant(seed: int = 7, days: float = 7.0, plant: list[AssetSpec] | None = None) -> list[AssetData]:
    rng = np.random.default_rng(seed)
    return [simulate_asset(spec, days, rng) for spec in (plant or DEFAULT_PLANT)]


def random_plant(rng: np.random.Generator, n_faulty: int = 3) -> list[AssetSpec]:
    """Same machines, random faults and timings - used to evaluate on unseen scenarios."""
    specs = [AssetSpec(a.id, a.name, a.kind, a.area) for a in DEFAULT_PLANT]
    for i in rng.choice(len(specs), size=n_faulty, replace=False):
        s = specs[i]
        s.fault = str(rng.choice(list(KINDS[s.kind].faults)))
        s.onset_day = float(rng.uniform(2.3, 4.5))
        s.duration_days = float(rng.uniform(1.0, 3.0))
    return specs
