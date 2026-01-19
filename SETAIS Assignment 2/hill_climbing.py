"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:
    - compute_fitness
    - mutate_seed
    - select_best
    - hill_climb

DO NOT change function signatures.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from keras.applications import vgg16
# from torchvision.models import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array
import torch
import os
import time
import csv
from datetime import datetime

# ============================================================
# GLOBAL LOGGING HELPERS
# ============================================================
QUERY_COUNT = 0

def reset_query_count():
    global QUERY_COUNT
    QUERY_COUNT = 0

def get_query_count() -> int:
    return int(QUERY_COUNT)

def compute_perturbation_metrics(original: np.ndarray, adv: np.ndarray) -> dict:
    """Compute common perturbation metrics for the report."""
    orig_f = original.astype(np.float32)
    adv_f = adv.astype(np.float32)
    diff = adv_f - orig_f

    linf = float(np.max(np.abs(diff)))
    l2 = float(np.sqrt(np.sum(diff ** 2)))

    changed = int(np.sum(np.abs(diff) > 1e-6))
    total = int(np.prod(diff.shape))

    return {
        "linf": linf,
        "l2": l2,
        "changed_values": changed,
        "changed_ratio": float(changed) / float(total) if total > 0 else 0.0,
    }

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, rows: list, fieldnames: list):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def top1_prediction(model, image_array: np.ndarray) -> tuple:
    """Return (label_str, prob_float, class_id_str) for top-1."""
    preds = model.predict(np.expand_dims(image_array, axis=0))
    top1 = decode_predictions(preds, top=1)[0][0]
    return top1[1], float(top1[2]), top1[0]

def top5_predictions(model, image_array: np.ndarray) -> list:
    preds = model.predict(np.expand_dims(image_array, axis=0))
    return [(cl[1], float(cl[2]), cl[0]) for cl in decode_predictions(preds, top=5)[0]]


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================
#TODO: Alexander
def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Compute fitness of an image for hill climbing.

    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)
    """
    global QUERY_COUNT
    QUERY_COUNT += 1

    # Get prediction probabilities
    preds = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]

    # Find top-2 indices
    idxsort = np.argsort(preds)
    top1idx = idxsort[-1]
    top2idx = idxsort[-2]

    # Decode top1 label string
    top1_label = decode_predictions(np.expand_dims(preds, axis=0), top=1)[0][0][1]

    top1_prob = float(preds[top1idx])
    top2_prob = float(preds[top2idx])

    if top1_label == target_label:
        # Still correct — we want to reduce confidence margin
        fitness = top1_prob - top2_prob
    else:
        # Misclassified — success (negative fitness)
        fitness = -top1_prob

    return float(fitness)



# ============================================================
# 2. MUTATION FUNCTION
# ============================================================
def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ANY NUMBER of mutated neighbors.

    Students may implement ANY mutation strategy:
        - modify 1 pixel
        - modify multiple pixels
        - patch-based mutation
        - channel-based mutation
        - gaussian noise (clipped)
        - etc.

    BUT EVERY neighbor must satisfy the L∞ constraint:

        For all pixels i,j,c:
            |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Requirements:
        ✓ Return a list of neighbors: [neighbor1, neighbor2, ..., neighborK]
        ✓ K can be ANY size ≥ 1
        ✓ Neighbors must be deep copies of seed
        ✓ Pixel values must remain in [0, 255]
        ✓ Must obey the L∞ bound exactly

    Args:
        seed (np.ndarray): input image
        epsilon (float): allowed perturbation budget

    Returns:
        List[np.ndarray]: mutated neighbors
    """

    seed_f = seed.astype(np.float32)
    l_budget = 255.0 * float(epsilon)

    # bounds relative to seed
    lower = np.clip(seed_f - l_budget, 0.0, 255.0)
    upper = np.clip(seed_f + l_budget, 0.0, 255.0)

    K = 20

    patch_sizes = [5, 16, 32]
    delta = max(1.0, 0.25 * l_budget)

    H, W, C = seed_f.shape
    neighbors: List[np.ndarray] = []

    for _ in range(K):
        neighbor = seed_f.copy()

        # random patch size and position
        p = int(np.random.choice(patch_sizes))
        top = np.random.randint(0, max(1, H - p + 1))
        left = np.random.randint(0, max(1, W - p + 1))

        # random noise in the patch
        noise = np.random.uniform(-delta, delta, size=(p, p, C)).astype(np.float32)
        neighbor[top:top + p, left:left + p, :] += noise

        # l bound relative to seed and pixel range
        neighbor = np.clip(neighbor, lower, upper)
        neighbor = np.clip(neighbor, 0.0, 255.0)

        neighbors.append(neighbor.astype(seed.dtype))

    return neighbors


# ============================================================
# 3. SELECT BEST CANDIDATE
# ============================================================

#TODO: Alexander
def select_best(
    candidates: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.

    Args:
        candidates (List[np.ndarray])
        model: classifier
        target_label (str)

    Returns:
        (best_image, best_fitness)
    """
    best_fitness = float('inf')
    best_candidate = candidates[0]
    for candidate in candidates:
        curr_fitness = compute_fitness(candidate, model, target_label)
        if curr_fitness < best_fitness:
            best_fitness = curr_fitness
            best_candidate = candidate

    return best_candidate, best_fitness


# ============================================================
# 4. HILL-CLIMBING ALGORITHM
# ============================================================

def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Main hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbors using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    Returns:
        (final_image, final_fitness)
    """
    trace = []
    current_seed = initial_seed.copy()
    current_fitness = compute_fitness(current_seed, model, target_label)

    l_budget = epsilon * 255
    initial_f = initial_seed.astype(np.float32)
    l_lower = np.clip(initial_f - l_budget, 0, 255)
    l_upper = np.clip(initial_f + l_budget, 0, 255)

    for iteration in range(iterations):
        neighbours = mutate_seed(seed=current_seed, epsilon=epsilon)
        neighbours_l_clipped = [np.clip(n, l_lower, l_upper) for n in neighbours]
        candidates = neighbours_l_clipped + [current_seed]
        best_seed, best_fitness = select_best(candidates=candidates, model=model, target_label=target_label)
        if best_fitness < current_fitness:
            current_seed = best_seed
            current_fitness = best_fitness
            trace.append({
                "iteration": int(iteration),
                "fitness": float(current_fitness),
            })
        if current_fitness < 0:
            break

    hill_climb.last_trace = trace
    return current_seed, current_fitness

# ============================================================
# 5. PROGRAM ENTRY POINT FOR RUNNING A SINGLE ATTACK
# ============================================================

if __name__ == "__main__":
    # -----------------------------
    # CONFIG
    # -----------------------------
    EPSILON = 0.30
    ITERATIONS = 300
    RESULTS_DIR = "hc_results"
    SAVE_ONLY_SUCCESS = False  # True => only save fooled images

    ensure_dir(RESULTS_DIR)

    # Load classifier
    model = vgg16.VGG16(weights="imagenet")

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    print(f"Loaded {len(image_list)} images from data/image_labels.json")
    print(f"Running Hill Climb with epsilon={EPSILON} for max {ITERATIONS} iterations")

    summary_rows = []
    detailed_rows = []

    start_all = time.time()

    for idx, item in enumerate(image_list):
        image_name = item["image"]
        target_label = item["label"]
        image_path = os.path.join("images", image_name)

        print("\n" + "=" * 60)
        print(f"[{idx+1}/{len(image_list)}] Image: {image_path} | Target: {target_label}")

        # Load image
        img = load_img(image_path)
        seed = img_to_array(img)

        # Clean prediction
        clean_top1_label, clean_top1_prob, _ = top1_prediction(model, seed)
        clean_correct = (clean_top1_label == target_label)

        # reset counters
        reset_query_count()

        # run attack
        t0 = time.time()
        final_img, final_fitness = hill_climb(
            initial_seed=seed.copy(),
            model=model,
            target_label=target_label,
            epsilon=EPSILON,
            iterations=ITERATIONS
        )
        runtime_s = time.time() - t0

        # adv prediction
        adv_top1_label, adv_top1_prob, _ = top1_prediction(model, final_img)
        success = (adv_top1_label != target_label)

        # perturbation metrics
        pert = compute_perturbation_metrics(seed, final_img)

        # query count
        queries = get_query_count()

        # Save images
        if success or not SAVE_ONLY_SUCCESS:
            base = os.path.splitext(os.path.basename(image_name))[0]
            clean_out = os.path.join(RESULTS_DIR, f"{base}_clean.png")
            adv_out = os.path.join(RESULTS_DIR, f"{base}_adv.png")
            array_to_img(seed).save(clean_out)
            array_to_img(final_img).save(adv_out)

        # summary row
        row = {
            "image": image_name,
            "target_label": target_label,
            "epsilon": float(EPSILON),
            "iterations_max": int(ITERATIONS),
            "runtime_s": float(runtime_s),
            "model_queries": int(queries),

            "clean_top1": clean_top1_label,
            "clean_prob": float(clean_top1_prob),
            "clean_correct": bool(clean_correct),

            "adv_top1": adv_top1_label,
            "adv_prob": float(adv_top1_prob),
            "success": bool(success),

            "final_fitness": float(final_fitness),
            "linf": pert["linf"],
            "l2": pert["l2"],
            "changed_values": pert["changed_values"],
            "changed_ratio": pert["changed_ratio"],
        }
        summary_rows.append(row)

        # detailed row (trace + top5)
        trace = getattr(hill_climb, "last_trace", [])
        detailed_rows.append({
            "image": image_name,
            "target_label": target_label,
            "success": bool(success),
            "final_fitness": float(final_fitness),
            "trace": trace,
            "clean_top5": top5_predictions(model, seed),
            "adv_top5": top5_predictions(model, final_img),
        })

        print(f"Clean top1: {clean_top1_label:20s} prob={clean_top1_prob:.5f} | correct={clean_correct}")
        print(f"Adv   top1: {adv_top1_label:20s} prob={adv_top1_prob:.5f} | success={success}")
        print(f"fitness={final_fitness:.6f} | queries={queries} | runtime={runtime_s:.2f}s | linf={pert['linf']:.2f}")

    total_runtime = time.time() - start_all

    # Write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(RESULTS_DIR, f"results_summary_{timestamp}.csv")
    detailed_path = os.path.join(RESULTS_DIR, f"results_detailed_{timestamp}.json")

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        write_csv(summary_path, summary_rows, fieldnames)

    with open(detailed_path, "w") as f:
        json.dump(detailed_rows, f, indent=2)

    # Aggregate stats
    total = len(summary_rows)
    successes = sum(1 for r in summary_rows if r["success"])
    clean_incorrect = sum(1 for r in summary_rows if not r["clean_correct"])
    success_rate = successes / total if total else 0

    mean_runtime = float(np.mean([r["runtime_s"] for r in summary_rows]))
    mean_queries = float(np.mean([r["model_queries"] for r in summary_rows]))
    mean_linf = float(np.mean([r["linf"] for r in summary_rows]))

    print("\n" + "=" * 60)
    print("HILL CLIMB SUMMARY")
    print(f"Total images: {total}")
    print(f"Clean misclassified (top1 != target): {clean_incorrect}")
    print(f"Attack success rate: {success_rate:.3f} ({successes}/{total})")
    print(f"Mean runtime (s): {mean_runtime:.2f}")
    print(f"Mean model queries: {mean_queries:.1f}")
    print(f"Mean L∞: {mean_linf:.2f}")
    print(f"Total runtime (s): {total_runtime:.2f}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved detailed JSON: {detailed_path}")