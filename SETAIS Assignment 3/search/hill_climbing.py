"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from envs.highway_env_utils import run_episode, record_video_episode
from search.base_search import ScenarioSearch




# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    # TODO (students)
    lanes_changed = 0
    old_lane = time_series[0]["ego"]["lane_id"]
    has_crashed = 0
    smallest_distance_total = float('inf')
    t_crashed = None
    for frame in time_series:
        # print(frame['crashed'])

        ego = frame['ego']

        if ego['lane_id'] != old_lane:
            lanes_changed += 1
            old_lane = ego['lane_id']

        ego_position = ego['pos']
        smallest_distance_frame = float('inf')
        # print()
        for other in frame['others']:
            other_position = other['pos']

            # Time till collision. Probably wont work but maybe
            # if (other_position[0] - ego_position[0]) < 0:
            #     if ()
            # TTCX = ((np.abs(other_position[0] - ego_position[0]) - ego['length']/2 - other['length']/2) / (ego['speed'] - 20))
            # TTCY = (np.abs(other_position[1] - ego_position[1]) - ego['width']/2 - other['width']/2) / (ego["heading"] * 4)


            #Eucladian
            # distance = ((np.abs(other_position[0] - ego_position[0]) - ego['length']/2 - other['length']/2)**2 +
            #             (np.abs(other_position[1] - ego_position[1]) - ego['width']/2 - other['width']/2)**2)**0.5
            if ego['lane_id'] == other['lane_id']:
                distance = (np.abs(other_position[0] - ego_position[0]) - ego['length']/2 - other['length']/2) # For some reason gives negative output sometimes... Should of crashed
                # distance = (np.abs(other_position[0] - ego_position[0]))
            else:
                distance = float('inf')
            if distance < smallest_distance_frame:
                smallest_distance_frame = distance

        if has_crashed == 0 and frame['crashed']: # For some reason frame['crashed'] is always false even when I see it crash on the video...
            t_crashed = frame['t']
            has_crashed = 1

        if smallest_distance_frame < smallest_distance_total:
            smallest_distance_total = smallest_distance_frame

    dictionary = dict(
        {
            "crash_count": has_crashed,
            "min_distance": smallest_distance_total,
            "t_crashed": t_crashed,
            "lanes_changed": lanes_changed
        }
    )
    print(dictionary)
    return dictionary




    # raise NotImplementedError


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """

    crashed = objectives['crash_count']
    if crashed == 1:
        return - (150/(objectives['t_crashed']+1))
    else:
        # return 50 - objectives["lanes_changed"]
        return objectives['min_distance']
    # TODO (students)
    # raise NotImplementedError


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    # print(cfg)
    new_cfg = copy.deepcopy(cfg)  # copy

    # Pick one parameter to mutate
    keys = list(param_spec.keys())
    # amount_of_mutations = rng.integers(1, 5)
    amount_of_mutations = 5
    # for i in range(amount_of_mutations):
    #     k = rng.choice(keys)
    for k in keys:

        spec = param_spec[k]
        lo, hi = spec["min"], spec["max"]
        t = spec["type"]

        max_jump = 0.4 * (hi - lo)
        # rng.integers(-max_jump, max_jump)
        jump = rng.normal(0.0, max_jump)
        if t == "int":
            if rng.random() < 0.4:
                delta = int(rng.choice([-int(jump), int(jump)]))
            else:
                delta = int(rng.choice([-2, -1, 1, 2]))
            # delta = int(round(jump))
            old = int(cfg[k])
            mutated = old + delta
            mutated = int(np.clip(mutated, lo, hi))
            new_cfg[k] = mutated
        elif t == "float":
            if rng.random() < 0.2:
                delta = float(jump)
            else:
                delta = float(rng.choice([-0.2, -0.1, 0.1, 0.2]))
            # delta = float(jump)
            old = float(cfg[k])
            mutated = old + delta
            mutated = float(np.clip(mutated, lo, hi))
            new_cfg[k] = mutated

        print("Variable: "+str(k)+" chasnged with: "+str(delta))

    # --- Mutation rules ---
    # if t == "int":
    #     # Small local step most of the time, occasional bigger hop
    #     # (good for hill climbing: mostly local, sometimes escape plateaus)
    #     if rng.random() < 0.66:
    #         delta = int(rng.choice([-2, -1, 1, 2]))
    #     else:
    #         # bigger jump up to 30% of range
    #         max_jump = max(1, int(round(0.3 * (hi - lo))))
    #         delta = int(rng.integers(-max_jump, max_jump + 1))
    #         if delta == 0:
    #             delta = 1
    #
    #     cur = int(new_cfg.get(k, lo))
    #     mutated = cur + delta
    #     mutated = int(np.clip(mutated, lo, hi))
    #     new_cfg[k] = mutated
    #
    # elif t == "float":
    #     # Gaussian step scaled to range (local search)
    #     cur = float(new_cfg.get(k, lo))
    #     span = float(hi - lo)
    #
    #     # mostly small steps, occasionally larger
    #     if rng.random() < 0.66:
    #         sigma = 0.05 * span   # 5% of range
    #     else:
    #         sigma = 0.30 * span   # 20% of range
    #
    #     mutated = cur + rng.normal(0.0, sigma)
    #     mutated = float(np.clip(mutated, lo, hi))
    #     new_cfg[k] = mutated

    # else:
    #     raise ValueError(f"Unknown type {t} for parameter {k}")

    # --- Constraint handling: lane-dependent initial_lane_id ---
    # If lanes_count changed OR even if not, enforce validity just in case.

    print(new_cfg)
    lanes = int(new_cfg["lanes_count"])

    lane_id = int(new_cfg["initial_lane_id"])
    lane_id = int(np.clip(lane_id, 0, lanes - 1))
    new_cfg["initial_lane_id"] = lane_id

    return new_cfg

    # raise NotImplementedError


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    # seed = 1
    rng = np.random.default_rng(seed)

    # TODO (students): choose initialization (base_cfg or random scenario)
    search = ScenarioSearch(env_id, base_cfg, param_spec, policy, defaults)
    # current_cfg = dict(base_cfg)
    current_cfg = ScenarioSearch.sample_random_config(search, rng)
    print("Current config: "+str(current_cfg))

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)
    best_ts = ts
    history = [None] * iterations

    for i in range(iterations):
        best_mcfg = current_cfg
        print("Iteration: "+str(i)+" starting with config: "+str(best_mcfg))
        for mi in range(neighbors_per_iter):
            mcfg = mutate_config(current_cfg, param_spec, rng)
            crashed, ts = run_episode(env_id, mcfg, policy, defaults, seed_base)

            objectives = compute_objectives_from_time_series(ts)
            if crashed:
                print("CRASH CRASH CRASH")
                objectives["crash_count"] = 1
                objectives["t_crashed"] = len(ts)
            fitness = compute_fitness(objectives)
            if fitness < cur_fit:
                best_ts = ts
                obj = objectives
                cur_fit = fitness
                best_mcfg = mcfg
                print("Better (lower) fitness found: "+str(cur_fit))
                record_video_episode(env_id, best_mcfg, policy, defaults, seed_base, out_dir="videos")

        history[i] = cur_fit
        current_cfg = best_mcfg

        if cur_fit < 0:
            print("Fitness under 0, fitness is: "+str(cur_fit))
            print("Could break here")
            break

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base

    # history = [best_fit]
    return {
        "best_cfg": best_cfg,
        "best_obj": best_obj,
        "best_fit": best_fit,
        "history": history,
        "best_seed_base": best_seed_base,
        "best_ts": best_ts
    }

    # TODO (students): implement HC loop
    # - generate neighbors
    # - evaluate
    # - pick best
    # - accept if improved
    # - early stop on crash (optional)

    # raise NotImplementedError