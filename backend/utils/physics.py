"""Physics and risk computation utilities."""

import math
from typing import List, Dict, Optional

# ─── Speed range definitions ────────────────────────────────────────────────────

SPEED_RANGES = [
    (0,   30,  "0–30 km/h"),
    (30,  60,  "30–60 km/h"),
    (60,  90,  "60–90 km/h"),
    (90,  120, "90–120 km/h"),
    (120, 150, "120–150 km/h"),
    (150, 200, ">150 km/h"),
]

FRICTION_COEFFICIENT = 0.70   # dry tarmac
GRAVITY = 9.81                 # m/s²
REACTION_TIME_S = 1.5          # driver reaction time (seconds)


# ─── TTC & Risk ─────────────────────────────────────────────────────────────────

def compute_ttc(
    distance_px: float,
    relative_velocity_px_per_frame: float,
    fps: int = 5,
) -> float:
    """
    Compute Time-To-Collision (TTC) in seconds.

    Args:
        distance_px: Current distance between vehicles (pixels)
        relative_velocity_px_per_frame: Approach speed (px/frame), positive = closing
        fps: Frames per second of the source video

    Returns:
        TTC in seconds (capped at 999 if vehicles not closing)
    """
    if relative_velocity_px_per_frame <= 0:
        return 999.0  # Vehicles moving apart
    return distance_px / (relative_velocity_px_per_frame * fps)


def compute_risk_score(
    ttc: float,
    iou: float,
    trajectory_score: float = 0.0,
) -> float:
    """
    Calibrated risk score combining TTC, IoU, and trajectory convergence.

    Design intent:
      - Vehicles far apart, not closing → < 0.15
      - Vehicles closing quickly (TTC < 3s) → 0.40–0.65
      - Vehicles overlapping (IoU > 0) → 0.70+
      - Actual collision (IoU > 0.15, TTC near 0) → 0.85+

    Args:
        ttc: Time-To-Collision in seconds
        iou: Bounding box IoU between vehicles (0–1)
        trajectory_score: 0 = paths diverge, 1 = paths converge head-on

    Returns:
        Risk score 0.0–1.0
    """
    # TTC risk: only meaningful when vehicles are actually closing (TTC < 60s)
    # Cap contribution so distant vehicles don't inflate score
    if ttc >= 60.0:
        ttc_risk = 0.0
    elif ttc >= 10.0:
        ttc_risk = 0.3 * (60.0 - ttc) / 50.0   # gentle ramp 10–60s → 0..0.3
    else:
        ttc_risk = min(1.2, 2.0 / max(ttc, 0.5))  # rapid ramp below 10s, capped at 1.2

    # IoU risk: only meaningful when boxes actually overlap (IoU > 0.01)
    iou_risk = 4.0 * max(0.0, iou - 0.01)   # zero below 0.01, rises sharply above

    # Trajectory risk: scaled down — it's a soft signal
    traj_risk = 1.0 * trajectory_score

    raw = ttc_risk + iou_risk + traj_risk
    # Shift baseline high enough that non-closing, non-overlapping pairs stay < 0.12
    return _sigmoid(raw - 2.5)


def aggregate_probability(
    pair_risks: List[Dict],
) -> Dict:
    """
    Combine per-pair risk scores into an overall accident probability.

    Uses the PEAK risk across all frames/pairs as the primary signal,
    with a small bonus from mean risk. This avoids the complementary-product
    inflation bug where many low-risk frames accumulate to 100%.

    Args:
        pair_risks: List of {probability, ttc, pair_id}

    Returns:
        {probability, risk_level, min_ttc_seconds, dominant_pair_index}
    """
    if not pair_risks:
        return {
            "probability": 0.0,
            "risk_level": "Low",
            "min_ttc_seconds": 999.0,
            "dominant_pair_index": None,
        }

    probs = [p["probability"] for p in pair_risks]
    peak  = max(probs)
    mean  = sum(probs) / len(probs)

    # Weighted blend: 80% peak + 20% mean — preserves the worst-case signal
    # while preventing low-mean videos from scoring near peak
    final_prob = round(min(0.99, 0.80 * peak + 0.20 * mean), 4)

    min_ttc  = min(p["ttc"] for p in pair_risks)
    dominant = min(pair_risks, key=lambda x: x["ttc"])

    return {
        "probability": final_prob,
        "risk_level": _risk_level(final_prob),
        "min_ttc_seconds": round(min_ttc, 2),
        "dominant_pair_index": dominant.get("pair_id"),
    }


# ─── Stopping distance & probability table ─────────────────────────────────────

def compute_stopping_distance(speed_kmh: float, friction: float = FRICTION_COEFFICIENT) -> float:
    """
    Total stopping distance = reaction distance + braking distance.

    d_stop = v * t_r  +  v² / (2 * µ * g)
    """
    v_ms = speed_kmh / 3.6
    reaction_dist = v_ms * REACTION_TIME_S
    braking_dist  = (v_ms ** 2) / (2.0 * friction * GRAVITY)
    return reaction_dist + braking_dist


def compute_accident_probability_at_speed(distance_m: float, speed_kmh: float) -> float:
    """
    Probability of accident given inter-vehicle distance and vehicle speed.

    P = 1 - exp(-stopping_distance / distance_m)
    Capped at 0.99.
    """
    if distance_m <= 0:
        return 0.99
    stopping_d = compute_stopping_distance(speed_kmh)
    ratio = stopping_d / distance_m
    return min(0.99, round(1.0 - math.exp(-ratio), 4))


def generate_probability_table(distance_m: float) -> List[Dict]:
    """
    Generate accident probability for each speed range.

    Args:
        distance_m: Estimated real-world inter-vehicle distance in meters

    Returns:
        List of {speed_range, avg_speed_kmh, stopping_distance_m,
                 probability_pct, risk_level, safe}
    """
    table = []
    for low, high, label in SPEED_RANGES:
        avg_speed = (low + min(high, 200)) / 2.0
        stop_dist = compute_stopping_distance(avg_speed)
        prob = compute_accident_probability_at_speed(distance_m, avg_speed)

        table.append({
            "speed_range": label,
            "avg_speed_kmh": avg_speed,
            "stopping_distance_m": round(stop_dist, 1),
            "probability_pct": round(prob * 100, 1),
            "risk_level": _risk_level(prob),
            "safe": prob < 0.25,
        })

    return table


# ─── Trajectory intersection ────────────────────────────────────────────────────

def predict_trajectory_intersection(
    history_a: List[tuple],
    history_b: List[tuple],
) -> Optional[tuple]:
    """
    Extrapolate linear trajectories of two vehicles and find intersection.

    Args:
        history_a: [(x, y), ...] last N positions of vehicle A
        history_b: [(x, y), ...] last N positions of vehicle B

    Returns:
        (x, y) intersection point or None if parallel/diverging
    """
    if len(history_a) < 2 or len(history_b) < 2:
        return None

    # Direction vectors
    dx_a = history_a[-1][0] - history_a[-2][0]
    dy_a = history_a[-1][1] - history_a[-2][1]
    dx_b = history_b[-1][0] - history_b[-2][0]
    dy_b = history_b[-1][1] - history_b[-2][1]

    x1, y1 = history_a[-1]
    x2, y2 = history_b[-1]

    # Solve parametric line intersection
    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) < 1e-6:
        return None  # Parallel trajectories

    t = ((x2 - x1) * dy_b - (y2 - y1) * dx_b) / denom
    if t < 0:
        return None  # Intersection is in the past

    ix = x1 + t * dx_a
    iy = y1 + t * dy_a
    return (ix, iy)


def compute_trajectory_score(
    history_a: List[tuple],
    history_b: List[tuple],
) -> float:
    """
    Returns a 0.0–1.0 score indicating how much trajectories converge.
    1.0 = head-on collision course, 0.0 = diverging.
    """
    if len(history_a) < 2 or len(history_b) < 2:
        return 0.0

    # Relative velocity vector
    vel_a = (history_a[-1][0] - history_a[-2][0], history_a[-1][1] - history_a[-2][1])
    vel_b = (history_b[-1][0] - history_b[-2][0], history_b[-1][1] - history_b[-2][1])

    # Vector pointing from B → A
    rel = (history_a[-1][0] - history_b[-1][0], history_a[-1][1] - history_b[-1][1])
    rel_norm = math.sqrt(rel[0] ** 2 + rel[1] ** 2)

    if rel_norm < 1e-6:
        return 1.0  # Already at same position

    rel_unit = (rel[0] / rel_norm, rel[1] / rel_norm)

    # Component of relative velocity in the closing direction
    relative_speed = (vel_b[0] - vel_a[0], vel_b[1] - vel_a[1])
    closing = relative_speed[0] * rel_unit[0] + relative_speed[1] * rel_unit[1]

    speed_mag = math.sqrt(relative_speed[0] ** 2 + relative_speed[1] ** 2)
    if speed_mag < 1e-6:
        return 0.0

    return max(0.0, min(1.0, closing / speed_mag))


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _risk_level(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    elif probability < 0.50:
        return "Medium"
    elif probability < 0.75:
        return "High"
    else:
        return "Critical"
