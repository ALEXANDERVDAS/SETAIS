import numpy as np



# from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import csv
import math
import numpy as np

from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env, run_episode, record_video_episode
from search.base_search import ScenarioSearch
from search.hill_climbing import hill_climb

env_id = "highway-fast-v0"
policy = load_pretrained_policy("agents/model")
env, defaults = make_env(env_id)

seed = 0

rng = np.random.default_rng(seed)




def summarize_time_series(ts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract CSV-friendly summary features from one rollout time_series.

    Returns a flat dict with:
      - cars_passed
      - lanes_crossed
      - max_speed
      - avg_speed
      - closest_same_lane
      - closest_any
      - crashed  (1 if crashed else 0)

    Notes / assumptions:
      - 'others' entries do NOT have stable IDs, so we assume list ordering is stable enough
        across frames to approximate "passed cars" per index.
      - "closest" distances are approximated rectangle-to-rectangle clearance in ego frame
        using ego heading and both cars' length/width.
    """
    if not ts:
        return {
            "cars_passed": 0,
            "lanes_crossed": 0,
            "max_speed": float("nan"),
            "avg_speed": float("nan"),
            "closest_same_lane": float("inf"),
            "closest_any": float("inf"),
            "crashed": 0,
        }

    T = len(ts)

    # ---- crashed flag ----
    crashed = 1 if any(fr.get("crashed", False) for fr in ts) else 0

    # ---- speed stats ----
    speeds = [float(fr["ego"]["speed"]) for fr in ts if "ego" in fr and "speed" in fr["ego"]]
    max_speed = max(speeds) if speeds else float("nan")
    avg_speed = float(sum(speeds) / len(speeds)) if speeds else float("nan")

    # ---- lanes crossed: count lane_id changes over time ----
    lane_ids = [int(fr["ego"]["lane_id"]) for fr in ts if "ego" in fr and "lane_id" in fr["ego"]]
    lanes_crossed = 0
    for i in range(1, len(lane_ids)):
        if lane_ids[i] != lane_ids[i - 1]:
            lanes_crossed += 1

    # ---- helper: world->ego frame and rectangle clearance ----
    def clearance_ego_frame(fr: Dict[str, Any], other: Dict[str, Any]) -> float:
        """
        Approximate rectangle-to-rectangle clearance (meters) in ego-aligned frame.
        clearance >= 0: gap, clearance < 0: overlap-like.
        """
        ego = fr["ego"]
        pe = np.array(ego["pos"], dtype=float)
        po = np.array(other["pos"], dtype=float)

        h = float(ego["heading"])
        c, s = math.cos(h), math.sin(h)
        # rotation: world -> ego
        R = np.array([[ c, s],
                      [-s, c]], dtype=float)

        rel_local = R @ (po - pe)
        dx, dy = abs(rel_local[0]), abs(rel_local[1])

        heL = float(ego["length"]) / 2.0
        heW = float(ego["width"]) / 2.0
        hoL = float(other["length"]) / 2.0
        hoW = float(other["width"]) / 2.0

        clear_x = dx - (heL + hoL)
        clear_y = dy - (heW + hoW)

        # overlap-ish
        if clear_x <= 0 and clear_y <= 0:
            return max(clear_x, clear_y)  # negative

        # distance to AABB in ego frame
        return math.sqrt(max(clear_x, 0.0) ** 2 + max(clear_y, 0.0) ** 2)

    # ---- closest distances (same lane / any lane) ----
    closest_any = float("inf")
    closest_same_lane = float("inf")

    for fr in ts:
        ego_lane = int(fr["ego"]["lane_id"])
        for other in fr.get("others", []):
            sep = clearance_ego_frame(fr, other)
            if sep < closest_any:
                closest_any = sep
            if int(other.get("lane_id", -9999)) == ego_lane and sep < closest_same_lane:
                closest_same_lane = sep

    # If no same-lane vehicles ever existed, keep it as inf (CSV-friendly alternative below)
    # You can swap inf -> NaN if you prefer:
    if closest_same_lane == float("inf"):
        closest_same_lane = float("nan")
    if closest_any == float("inf"):
        closest_any = float("nan")

    # ---- cars passed (approx) ----
    # Idea: in ego frame, a car is "ahead" if its local x > 0, "behind" if local x < 0.
    # Count how many distinct others transition from ahead->behind over the rollout.
    passed_indices = set()

    # Track sign of each other vehicle's longitudinal position over time, by index
    prev_sign: Dict[int, int] = {}

    for fr in ts:
        ego = fr["ego"]
        pe = np.array(ego["pos"], dtype=float)
        h = float(ego["heading"])
        c, s = math.cos(h), math.sin(h)
        R = np.array([[ c, s],
                      [-s, c]], dtype=float)

        others = fr.get("others", [])
        for j, other in enumerate(others):
            po = np.array(other["pos"], dtype=float)
            rel_local = R @ (po - pe)
            x_local = float(rel_local[0])

            # sign with a small deadzone around 0 to avoid flicker
            dead = 0.5  # meters
            if x_local > dead:
                sign = 1
            elif x_local < -dead:
                sign = -1
            else:
                sign = 0

            if j not in prev_sign:
                prev_sign[j] = sign
                continue

            # "passed" if it was clearly ahead and later clearly behind
            if prev_sign[j] == 1 and sign == -1:
                passed_indices.add(j)

            # update only when sign is non-zero (reduces noise around 0)
            if sign != 0:
                prev_sign[j] = sign

    cars_passed = len(passed_indices)

    return {
        "cars_passed": cars_passed,
        "lanes_crossed": lanes_crossed,
        "max_speed": max_speed,
        "avg_speed": avg_speed,
        "closest_same_lane": closest_same_lane,
        "closest_any": closest_any,
        "crashed": crashed,
    }


def append_summary_to_csv(
    csv_path: str,
    summary: Dict[str, Any],
    *,
    fieldnames: Optional[List[str]] = None
) -> None:
    """
    Appends one summary row to a CSV. Creates the file + header if it doesn't exist.
    """
    if fieldnames is None:
        fieldnames = [
            "cars_passed",
            "lanes_crossed",
            "max_speed",
            "avg_speed",
            "closest_same_lane",
            "closest_any",
            "crashed",
        ]

    # Create file if needed and write header once
    try:
        with open(csv_path, "r", newline="") as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: summary.get(k) for k in fieldnames})


# ----------------------------
# Example usage
# ----------------------------

search = ScenarioSearch(env_id, base_cfg, param_spec, policy, defaults)

for i in range(1000):
    cfg = ScenarioSearch.sample_random_config(search, rng)
    s = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, cfg, policy, defaults, s)
    summary = summarize_time_series(ts)
    if crashed:
        summary["crashed"] = 1
    else:
        summary["crashed"] = 0

    append_summary_to_csv("rollouts.csv", summary)

# If you have many rollouts:
# for ts in list_of_time_series:
#     summary = summarize_time_series(ts, dt=0.1)
#     append_summary_to_csv("rollouts.csv", summary)

