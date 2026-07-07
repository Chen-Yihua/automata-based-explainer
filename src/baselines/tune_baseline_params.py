#!/usr/bin/env python3
"""
Fair-flow hyperparameter tuning for SA / GA / PSO baselines.

Goal:
- Tune baseline search breadth parameters, then compare the best baseline result with Beam Search.
- Keep the comparison process fair by enforcing the same initial DFA, same samples,
  same operators, same agreement threshold, same max_evaluations, and one DFA edit per candidate.

Tuned parameters:
- SA: local candidate pool size only.
- GA: population size only.
- PSO: number of particles and local candidate pool size only.

Not tuned in the main fair flow:
- PSO max_ops_per_iteration is fixed to 1.
- SA temperature schedule is fixed.
- GA tournament size is fixed.

This script assumes a shared_init.pkl exists under a previous experiment folder.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pickle
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

# PROJECT_ROOT = Path(__file__).resolve().parent
# SRC_PATH = PROJECT_ROOT / "src"
# if str(SRC_PATH) not in sys.path:
#     sys.path.insert(0, str(SRC_PATH))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

for _p in [SRC_PATH, PROJECT_ROOT]:
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines.search_baselines import SharedInit, ga_dfa_search, pso_dfa_search, sa_dfa_search

# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------
AGREEMENT_THRESHOLD = 0.8
BATCH_SIZE = 500
MAX_EVALUATIONS = 3000

# ---------------------------------------------------------------------
# Fair-flow tuning grid
# ---------------------------------------------------------------------
# We tune only search breadth / population parameters.
# Everything that changes the refinement depth or gives a method a different
# candidate-generation process is fixed.
#
# Fair constraints used for every baseline run:
#   1. Same SharedInit: initial DFA + train/validation samples.
#   2. Same agreement threshold.
#   3. Same MAX_EVALUATIONS candidate budget.
#   4. Same DELETE / MERGE / DELTA operators.
#   5. One DFA edit per generated candidate.
#
# SA: tune how many one-step neighbors are considered per proposal.
#     steps is set to MAX_EVALUATIONS so the evaluation budget, not iteration
#     count, is the stopping control.
SA_POOL_SIZES = [3, 5, 10]
SA_FIXED = {
    "steps": MAX_EVALUATIONS,
    "T_max": 10.0,
    "T_min": 0.001,
}

# GA: tune population size. Mutation still applies one DFA refinement step.
GA_POPULATION_SIZES = [5, 10, 20]
GA_FIXED = {
    "tournament_size": 2,
}

# PSO: tune number of trajectories and local candidate pool size.
# pso_max_ops_per_iteration is fixed to 1 to match Beam's one-step expansion.
PSO_N_PARTICLES = [5, 10, 20]
PSO_POOL_SIZES = [3, 5, 10]
PSO_FIXED = {
    "pso_max_ops_per_iteration": 1,
}


def build_sa_grid() -> list[dict]:
    return [
        {**SA_FIXED, "sa_candidate_pool_size": pool_size}
        for pool_size in SA_POOL_SIZES
    ]


def build_ga_grid() -> list[int]:
    return list(GA_POPULATION_SIZES)


def build_pso_grid():
    return product(PSO_N_PARTICLES, PSO_POOL_SIZES, [PSO_FIXED["pso_max_ops_per_iteration"]])


def format_sa_config(cfg: dict) -> str:
    # Only pool size is treated as a tuned hyperparameter.
    # steps/T schedule are fixed fair-flow settings.
    return f"pool={cfg['sa_candidate_pool_size']}"


def format_pso_config(n_particles: int, pool_size: int, max_ops: int) -> str:
    return f"n_particles={n_particles},pool={pool_size},ops={max_ops}"


def extract_result_metrics(result: dict) -> dict:
    """Extract states and agreement using current and backward-compatible keys."""
    automata = result.get("automata")
    states = result.get("size", 0) or (len(automata.states) if automata is not None and hasattr(automata, "states") else 0)

    agreement = result.get("training_agreement", result.get("training_accuracy", 0.0))
    if isinstance(agreement, (list, tuple, np.ndarray)):
        agreement = float(np.asarray(agreement).reshape(-1)[-1]) if len(agreement) else 0.0
    else:
        agreement = float(agreement or 0.0)

    validation_agreement = result.get("validation_agreement", result.get("validation_accuracy", 0.0))
    if isinstance(validation_agreement, (list, tuple, np.ndarray)):
        validation_agreement = float(np.asarray(validation_agreement).reshape(-1)[-1]) if len(validation_agreement) else 0.0
    else:
        validation_agreement = float(validation_agreement or 0.0)

    return {
        "states": int(states) if states else 0,
        "agreement": agreement,
        "validation_agreement": validation_agreement,
        "evaluations_used": int(result.get("evaluations_used", 0) or 0),
        "max_evaluations": int(result.get("max_evaluations", MAX_EVALUATIONS) or MAX_EVALUATIONS),
        "operator_counts": result.get("operator_counts", {}),
    }


def load_shared_init_from_disk(output_dir: str | Path):
    output_dir = Path(output_dir)
    shared_init_path = output_dir / "shared" / "shared_init.pkl"
    if not shared_init_path.exists():
        print(f"  [SharedInit] 找不到: {shared_init_path}")
        return None

    try:
        with shared_init_path.open("rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, SharedInit):
            print(f"  ✓ 讀取 SharedInit: {len(obj.initial_dfa.states)} states")
            return obj

        if isinstance(obj, dict) and "initial_dfa" in obj:
            print(f"  ✓ 讀取 shared_init.pkl: {len(obj['initial_dfa'].states)} states")
            return obj

        if isinstance(obj, (tuple, list)) and len(obj) >= 4:
            print(f"  ✓ 讀取 shared_init.pkl 舊格式: {len(obj[0].states)} states")
            return {
                "initial_dfa": obj[0],
                "learner": obj[1],
                "validation_data": obj[2],
                "validation_labels": obj[3],
            }

        print("  ✗ 未知 shared_init.pkl format")
        return None
    except Exception as exc:
        print(f"  ✗ 讀取失敗: {exc}")
        return None


def _resolve_tune_root(tune_root: str | None) -> Path:
    """Resolve the folder used for tuning.

    Examples
    --------
    --tune_root regular_0.9_1000
        -> PROJECT_ROOT/test_result/regular_0.9_1000
    --tune_root test_result/regular_0.9_1000
        -> PROJECT_ROOT/test_result/regular_0.9_1000
    --tune_root /abs/path/to/result
        -> /abs/path/to/result
    """
    if not tune_root:
        return PROJECT_ROOT / "test_result"

    p = Path(tune_root)
    if p.is_absolute():
        return p

    # Allow both "regular_0.9_1000" and "test_result/regular_0.9_1000".
    if p.parts and p.parts[0] == "test_result":
        return PROJECT_ROOT / p
    return PROJECT_ROOT / "test_result" / p



def _resolve_tune_output_dir(output_name: str | None) -> Path:
    """Create a new result folder under test_result for tuning logs and tables."""
    base = PROJECT_ROOT / "test_result"
    if output_name:
        name = output_name
    else:
        name = "tune_fairflow_baselines_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / name
    out.mkdir(parents=True, exist_ok=False)
    return out


def _safe_name(value: str, max_len: int = 120) -> str:
    """Make a filesystem-safe compact name."""
    value = str(value).replace(str(PROJECT_ROOT), "")
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")
    return value[:max_len] or "run"


class Tee:
    """Write stdout to both terminal and a log file."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def write_results_tables(rows: list[dict], output_dir: Path) -> None:
    """Persist tuning results as CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tune_results.json"
    csv_path = output_dir / "tune_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "experiment", "algo", "config", "success", "states", "agreement",
        "validation_agreement", "time", "evaluations_used", "max_evaluations",
        "operator_counts", "meets_threshold", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_text(all_results: dict, experiment_names: list[str], output_dir: Path) -> None:
    """Write the same ranking summary to a text file."""
    rows = []
    for exp_name in experiment_names:
        for algo, configs in all_results[exp_name].items():
            for config_name, runs in configs.items():
                if not runs:
                    continue
                best_run = min(
                    runs,
                    key=lambda r: (
                        r.get("states", 10**9),
                        -r.get("agreement", 0.0),
                        r.get("time", 10**9),
                    ),
                )
                rows.append({
                    "experiment": exp_name,
                    "algo": algo.upper(),
                    "config": config_name,
                    "states": best_run.get("states", 0),
                    "agreement": best_run.get("agreement", 0.0),
                    "validation_agreement": best_run.get("validation_agreement", 0.0),
                    "time": best_run.get("time", 0.0),
                    "meets_threshold": best_run.get("agreement", 0.0) >= AGREEMENT_THRESHOLD,
                })

    ranked = sorted(rows, key=lambda r: (not r["meets_threshold"], r["states"], -r["agreement"], r["time"]))
    path = output_dir / "summary_top20.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Ranking rule: meets threshold > smaller states > larger agreement > shorter time\n")
        for i, row in enumerate(ranked[:20], start=1):
            f.write(
                f"{i:2d}. {row['experiment']:<50s} | {row['algo']:3s} | {row['config']:<35s} "
                f"| ok={int(row['meets_threshold'])} | states={row['states']:4d} "
                f"| agreement={row['agreement']:.4f} | val={row['validation_agreement']:.4f} "
                f"| time={row['time']:.1f}s\n"
            )


def write_best_by_algo_table(rows: list[dict], output_dir: Path) -> None:
    """Write the best tuned config for each experiment x algorithm."""
    successful = [r for r in rows if r.get("success")]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in successful:
        grouped[(row.get("experiment", ""), row.get("algo", ""))].append(row)

    best_rows = []
    for (experiment, algo), runs in sorted(grouped.items()):
        best = sorted(
            runs,
            key=lambda r: (
                not r.get("meets_threshold", False),
                r.get("states", 10**9),
                -r.get("agreement", 0.0),
                r.get("time", 10**9),
            ),
        )[0]
        best_rows.append(best)

    path = output_dir / "best_by_algo.csv"
    fieldnames = [
        "experiment", "algo", "config", "success", "states", "agreement",
        "validation_agreement", "time", "evaluations_used", "max_evaluations",
        "operator_counts", "meets_threshold", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in best_rows:
            writer.writerow(row)


def _parse_csv(value: str | None) -> set[str]:
    """Parse comma-separated dataset/instance filters."""
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def _matches_filters(exp_dir: Path, dataset_filter: set[str], instance_filter: set[str]) -> bool:
    """Return True if an experiment directory matches CLI filters."""
    parts = set(exp_dir.relative_to(PROJECT_ROOT).parts) if exp_dir.is_relative_to(PROJECT_ROOT) else set(exp_dir.parts)

    if dataset_filter and not (parts & dataset_filter):
        return False

    if instance_filter and not (parts & instance_filter):
        return False

    return True


def find_experiment_dirs(
    tune_root: str | None = None,
    datasets: str | None = None,
    instances: str | None = None,
) -> list[Path]:
    """Find experiment directories that contain shared/shared_init.pkl.

    Parameters
    ----------
    tune_root:
        Which result folder to search. For example: regular_0.9_1000 or realworld_0.8_500.
        If omitted, search the whole test_result folder.
    datasets:
        Optional comma-separated dataset names, e.g. SecureHandshake,mnist.
    instances:
        Optional comma-separated instance folders, e.g. instance_00,instance_01.
    """
    root = _resolve_tune_root(tune_root)
    dataset_filter = _parse_csv(datasets)
    instance_filter = _parse_csv(instances)

    if not root.exists():
        print(f"找不到 tune_root: {root}")
        return []

    candidates = sorted({p.parent.parent for p in root.rglob("shared/shared_init.pkl")})
    return [p for p in candidates if _matches_filters(p, dataset_filter, instance_filter)]


def parse_args():
    parser = argparse.ArgumentParser(description="Tune SA / GA / PSO baseline parameters on selected datasets.")
    parser.add_argument(
        "--tune_root",
        type=str,
        default=None,
        help="Result folder to tune on, e.g. regular_0.9_1000, realworld_0.8_500, or an absolute path.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names, e.g. SecureHandshake,DocumentReleaseWorkflow,mnist.",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Comma-separated instance folders, e.g. instance_00,instance_01. Omit to use all instances.",
    )
    parser.add_argument(
        "--tune_output_name",
        type=str,
        default=None,
        help="Folder name under test_result for tuning logs/results. Default: tune_baselines_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--keep_scratch",
        action="store_true",
        help="Keep temporary baseline output folders. By default only logs and summary tables are kept.",
    )
    return parser.parse_args()


def load_training_data(shared_data, batch_size: int = BATCH_SIZE):
    if isinstance(shared_data, SharedInit):
        if shared_data.training_data is not None and shared_data.training_labels is not None:
            return list(shared_data.training_data), np.asarray(shared_data.training_labels)
        return None

    if "training_data" in shared_data and "training_labels" in shared_data:
        training_data = list(shared_data["training_data"])
        training_labels = np.asarray(shared_data["training_labels"])
        print(f"  ✓ 從 shared_init.pkl 讀取訓練資料: {len(training_data)} samples")
        return training_data, training_labels

    validation_data = list(shared_data.get("validation_data", []))
    validation_labels = np.asarray(shared_data.get("validation_labels", []))
    if len(validation_data) > 0 and len(validation_labels) > 0:
        repeat_count = max(1, int(np.ceil(batch_size / len(validation_data))))
        training_data = (validation_data * repeat_count)[:batch_size]
        training_labels = np.tile(validation_labels, repeat_count)[:batch_size]
        print(f"  ⚠ shared_init 沒有 training_data，暫用 validation_data: {len(training_data)} samples")
        return training_data, training_labels

    return None


def make_shared_init(shared_data, training_data, training_labels) -> SharedInit:
    if isinstance(shared_data, SharedInit):
        return SharedInit(
            initial_dfa=shared_data.initial_dfa,
            learner=shared_data.learner,
            validation_data=list(shared_data.validation_data),
            validation_labels=np.asarray(shared_data.validation_labels),
            training_data=list(training_data),
            training_labels=np.asarray(training_labels),
        )

    return SharedInit(
        initial_dfa=shared_data["initial_dfa"],
        learner=shared_data["learner"],
        validation_data=list(shared_data.get("validation_data", [])),
        validation_labels=np.asarray(shared_data.get("validation_labels", [])),
        training_data=list(training_data),
        training_labels=np.asarray(training_labels),
    )


def run_sa_test(shared_init: SharedInit, cfg: dict, output_dir: Path) -> dict:
    cfg_name = format_sa_config(cfg)
    print(f"  [SA] {cfg_name}...", end="", flush=True)
    start = time.time()
    try:
        result = sa_dfa_search(
            data_type="Tabular",
            shared_init=shared_init,
            agreement_threshold=AGREEMENT_THRESHOLD,
            init_num_samples=100,
            batch_size=BATCH_SIZE,
            output_dir=str(output_dir),
            beam_size=1,
            steps=cfg["steps"],
            T_max=cfg["T_max"],
            T_min=cfg["T_min"],
            max_evaluations=MAX_EVALUATIONS,
            sa_candidate_pool_size=cfg["sa_candidate_pool_size"],
            instance=None,
        )
        elapsed = time.time() - start
        metrics = extract_result_metrics(result)
        eval_str = f", eval={metrics.get('evaluations_used', 0)}/{metrics.get('max_evaluations', MAX_EVALUATIONS)}"
        print(f" ✓ ({elapsed:.1f}s, states={metrics['states']}, agreement={metrics['agreement']:.3f}{eval_str})")
        return {"success": True, **metrics, "time": elapsed}
    except Exception as exc:
        elapsed = time.time() - start
        print(f" ✗ ({elapsed:.1f}s): {str(exc)[:80]}")
        return {"success": False, "time": elapsed, "error": str(exc)}


def run_ga_test(shared_init: SharedInit, population_size: int, output_dir: Path) -> dict:
    print(f"  [GA] pop={population_size}...", end="", flush=True)
    start = time.time()
    try:
        result = ga_dfa_search(
            data_type="Tabular",
            shared_init=shared_init,
            agreement_threshold=AGREEMENT_THRESHOLD,
            init_num_samples=100,
            batch_size=BATCH_SIZE,
            output_dir=str(output_dir),
            population_size=population_size,
            tournament_size=2,
            max_evaluations=MAX_EVALUATIONS,
            instance=None,
        )
        elapsed = time.time() - start
        metrics = extract_result_metrics(result)
        eval_str = f", eval={metrics.get('evaluations_used', 0)}/{metrics.get('max_evaluations', MAX_EVALUATIONS)}"
        print(f" ✓ ({elapsed:.1f}s, states={metrics['states']}, agreement={metrics['agreement']:.3f}{eval_str})")
        return {"success": True, **metrics, "time": elapsed}
    except Exception as exc:
        elapsed = time.time() - start
        print(f" ✗ ({elapsed:.1f}s): {str(exc)[:80]}")
        return {"success": False, "time": elapsed, "error": str(exc)}


def run_pso_test(shared_init: SharedInit, n_particles: int, pool_size: int, max_ops: int, output_dir: Path) -> dict:
    cfg_name = format_pso_config(n_particles, pool_size, max_ops)
    print(f"  [PSO] {cfg_name}...", end="", flush=True)
    start = time.time()
    try:
        result = pso_dfa_search(
            data_type="Tabular",
            shared_init=shared_init,
            agreement_threshold=AGREEMENT_THRESHOLD,
            init_num_samples=100,
            batch_size=BATCH_SIZE,
            output_dir=str(output_dir),
            n_particles=n_particles,
            beam_size=1,
            max_evaluations=MAX_EVALUATIONS,
            pso_candidate_pool_size=pool_size,
            pso_max_ops_per_iteration=max_ops,
            instance=None,
        )
        elapsed = time.time() - start
        metrics = extract_result_metrics(result)
        eval_str = f", eval={metrics.get('evaluations_used', 0)}/{metrics.get('max_evaluations', MAX_EVALUATIONS)}"
        print(f" ✓ ({elapsed:.1f}s, states={metrics['states']}, agreement={metrics['agreement']:.3f}{eval_str})")
        return {"success": True, **metrics, "time": elapsed}
    except Exception as exc:
        elapsed = time.time() - start
        print(f" ✗ ({elapsed:.1f}s): {str(exc)[:80]}")
        return {"success": False, "time": elapsed, "error": str(exc)}


def summarize(all_results: dict, experiment_names: list[str]) -> None:
    print("\n" + "=" * 100)
    print("跨資料集最佳參數組合")
    print("=" * 100)

    rows = []
    for exp_name in experiment_names:
        for algo, configs in all_results[exp_name].items():
            for config_name, runs in configs.items():
                if not runs:
                    continue
                best_run = min(runs, key=lambda r: (r.get("states", 10**9), -r.get("agreement", 0.0), r.get("time", 10**9)))
                rows.append({
                    "experiment": exp_name,
                    "algo": algo.upper(),
                    "config": config_name,
                    "states": best_run.get("states", 0),
                    "agreement": best_run.get("agreement", 0.0),
                    "time": best_run.get("time", 0.0),
                    "meets_threshold": best_run.get("agreement", 0.0) >= AGREEMENT_THRESHOLD,
                })

    if not rows:
        print("沒有可用結果。")
        return

    ranked = sorted(rows, key=lambda r: (not r["meets_threshold"], r["states"], -r["agreement"], r["time"]))
    print("排名規則: 達成 threshold > states 小 > agreement 大 > time 小")
    for i, row in enumerate(ranked[:20], start=1):
        print(
            f"{i:2d}. {row['experiment']:<35s} | {row['algo']:3s} | {row['config']:<35s} "
            f"| ok={int(row['meets_threshold'])} | states={row['states']:4d} "
            f"| agreement={row['agreement']:.4f} | time={row['time']:.1f}s"
        )


def main() -> None:
    args = parse_args()
    tune_output_dir = _resolve_tune_output_dir(args.tune_output_name)
    scratch_dir = tune_output_dir / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    log_path = tune_output_dir / "tune.log"
    all_rows: list[dict] = []

    with log_path.open("w", encoding="utf-8") as log_f, contextlib.redirect_stdout(Tee(sys.stdout, log_f)):
        print("=" * 100)
        print("SA / GA / PSO fair-flow hyperparameter tuning")
        print("=" * 100)
        print(f"Agreement threshold: {AGREEMENT_THRESHOLD}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Max evaluations: {MAX_EVALUATIONS}")
        print(f"Tune root: {_resolve_tune_root(args.tune_root)}")
        print(f"Tune output: {tune_output_dir}")
        print(f"Datasets: {args.datasets or 'ALL'}")
        print(f"Instances: {args.instances or 'ALL'}")
        print("Fair constraints:")
        print("  - same SharedInit, samples, threshold, and max_evaluations")
        print("  - same DELETE / MERGE / DELTA operators")
        print("  - one DFA edit per generated candidate")
        print(f"SA pool sizes: {SA_POOL_SIZES}")
        print(f"GA population sizes: {GA_POPULATION_SIZES}")
        print(f"PSO n_particles: {PSO_N_PARTICLES}")
        print(f"PSO candidate pool sizes: {PSO_POOL_SIZES}")
        print(f"PSO max_ops_per_iteration: {PSO_FIXED['pso_max_ops_per_iteration']} (fixed)")
        print()

        experiment_dirs = find_experiment_dirs(
            tune_root=args.tune_root,
            datasets=args.datasets,
            instances=args.instances,
        )
        if not experiment_dirs:
            print("找不到符合條件的 shared/shared_init.pkl。")
            print("請確認 --tune_root / --datasets / --instances 是否正確，或先執行一次 main experiment。")
            return

        print("找到可用 shared_init:")
        for d in experiment_dirs:
            print(f"  - {d.relative_to(PROJECT_ROOT)}")
        print()

        all_results = defaultdict(lambda: {"sa": defaultdict(list), "ga": defaultdict(list), "pso": defaultdict(list)})
        experiment_names = []

        for exp_dir in experiment_dirs:
            exp_name = str(exp_dir.relative_to(PROJECT_ROOT))
            safe_exp = _safe_name(exp_name)
            experiment_names.append(exp_name)
            print("\n" + "=" * 100)
            print(f"Experiment: {exp_name}")
            print("=" * 100)

            shared_data = load_shared_init_from_disk(exp_dir)
            if shared_data is None:
                continue

            training_result = load_training_data(shared_data, BATCH_SIZE)
            if not training_result:
                print("  ✗ 無法取得 training_data / training_labels")
                continue
            training_data, training_labels = training_result
            shared_init = make_shared_init(shared_data, training_data, training_labels)

            print("\n  SA 參數掃描")
            for cfg in build_sa_grid():
                cfg_name = format_sa_config(cfg)
                safe_cfg = _safe_name(cfg_name)
                out_dir = scratch_dir / safe_exp / f"sa_{safe_cfg}"
                result = run_sa_test(shared_init, cfg, out_dir)
                row = {
                    "experiment": exp_name,
                    "algo": "SA",
                    "config": cfg_name,
                    **result,
                    "meets_threshold": result.get("agreement", 0.0) >= AGREEMENT_THRESHOLD,
                }
                all_rows.append(row)
                if result["success"]:
                    all_results[exp_name]["sa"][cfg_name].append(result)

            print("\n  GA 參數掃描")
            for pop_size in build_ga_grid():
                cfg_name = f"pop={pop_size}"
                out_dir = scratch_dir / safe_exp / f"ga_pop{pop_size}"
                result = run_ga_test(shared_init, pop_size, out_dir)
                row = {
                    "experiment": exp_name,
                    "algo": "GA",
                    "config": cfg_name,
                    **result,
                    "meets_threshold": result.get("agreement", 0.0) >= AGREEMENT_THRESHOLD,
                }
                all_rows.append(row)
                if result["success"]:
                    all_results[exp_name]["ga"][cfg_name].append(result)

            print("\n  PSO 參數掃描")
            for n_particles, pool_size, max_ops in build_pso_grid():
                cfg_name = format_pso_config(n_particles, pool_size, max_ops)
                out_dir = scratch_dir / safe_exp / f"pso_n{n_particles}_pool{pool_size}_ops{max_ops}"
                result = run_pso_test(
                    shared_init,
                    n_particles=n_particles,
                    pool_size=pool_size,
                    max_ops=max_ops,
                    output_dir=out_dir,
                )
                row = {
                    "experiment": exp_name,
                    "algo": "PSO",
                    "config": cfg_name,
                    **result,
                    "meets_threshold": result.get("agreement", 0.0) >= AGREEMENT_THRESHOLD,
                }
                all_rows.append(row)
                if result["success"]:
                    all_results[exp_name]["pso"][cfg_name].append(result)

            # Persist after every experiment, so partial results survive interruption.
            write_results_tables(all_rows, tune_output_dir)
            write_best_by_algo_table(all_rows, tune_output_dir)
            write_summary_text(all_results, experiment_names, tune_output_dir)

        summarize(all_results, experiment_names)
        write_results_tables(all_rows, tune_output_dir)
        write_best_by_algo_table(all_rows, tune_output_dir)
        write_summary_text(all_results, experiment_names, tune_output_dir)

        if not args.keep_scratch and scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
            print(f"\n已刪除 temporary scratch folder: {scratch_dir}")

        print("\nTune results saved:")
        print(f"  log     : {log_path}")
        print(f"  csv     : {tune_output_dir / 'tune_results.csv'}")
        print(f"  json    : {tune_output_dir / 'tune_results.json'}")
        print(f"  best    : {tune_output_dir / 'best_by_algo.csv'}")
        print(f"  summary : {tune_output_dir / 'summary_top20.txt'}")


if __name__ == "__main__":
    main()
