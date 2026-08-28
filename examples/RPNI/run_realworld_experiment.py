"""
Real-world sequence experiment.

This script only loads neural teachers and selects test instances.  The shared
Beam / SA / GA / PSO logic lives in experiments.runner.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sklearn.metrics import accuracy_score

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, "external_modules")
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, "Explaining-FA")

for _p in [SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from experiments.runner import print_suite_summary, run_search_suite
from models.sequence_classifier import SequenceClassifier
from tee import Tee


DEFAULT_LANGUAGE_CONFIGS = {
    "mnist": dict(
        alphabet=["R", "U", "L", "D"],
        agreement_threshold=0.8,
        delta=0.01,
        tau=0.1,
        batch_size=1000,
        beam_size=1,
        init_num_samples=500,
        edit_distance=3,
        parallel=True,
        n_jobs=4,
        use_prediction_cache=True,
        prediction_cache_max_size=200000,
        num_test_instances=10,
        # test_instance=['R', 'R', 'R', 'R', 'D', 'D', 'L', 'D', 'D', 'L', 'D', 'D', 'D'],
        test_instance = ['R', 'R', 'R', 'R','D', 'D', 'L', 'D', 'D', 'L', 'D', 'D'],
        test_instances=None,
        max_length=20,
        embedding_dim=64,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
    ),
    "ECG": dict(
        alphabet=["VL", "L", "SL", "M", "SH", "H", "VH"],
        agreement_threshold=0.8,
        delta=0.01,
        tau=0.1,
        batch_size=1000,
        beam_size=1,
        init_num_samples=500,
        edit_distance=2,
        parallel=True,
        n_jobs=4,
        use_prediction_cache=True,
        prediction_cache_max_size=200000,
        num_test_instances=10,
        test_instance=['VL', 'M', 'M', 'M', 'H', 'H', 'SH', 'SL', 'SH', 'VL', 'SL'],
        test_instances=None,
        max_length=20,
        embedding_dim=64,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
    ),
    "wafer": dict(
        alphabet=["VL", "L", "SL", "M", "SH", "H", "VH"],
        agreement_threshold=0.8,
        delta=0.01,
        tau=0.1,
        batch_size=1000,
        beam_size=1,
        init_num_samples=500,
        edit_distance=2,
        parallel=True,
        n_jobs=4,
        use_prediction_cache=True,
        prediction_cache_max_size=200000,
        num_test_instances=10,
        test_instance=['VL', 'VH', 'VH', 'SL', 'M', 'SH', 'SH', 'SH', 'SH', 'SH', 'SH', 'SH', 'SL', 'L', 'L', 'L', 'L'],
        test_instances=None,
        max_length=20,
        embedding_dim=64,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
    ),
}


def get_languages_config(overrides=None):
    """Return dataset configurations with optional CLI overrides."""
    import copy

    configs = copy.deepcopy(DEFAULT_LANGUAGE_CONFIGS)
    if overrides:
        for cfg in configs.values():
            for key, value in overrides.items():
                if value is not None:
                    cfg[key] = value
    return configs


def _normalize_sequence(seq):
    if isinstance(seq, np.ndarray):
        return seq.tolist()
    return list(seq)


def get_test_instances(X_train, cfg):
    """
    Choose real-world test instances.

    Priority:
    1. cfg['test_instances']
    2. cfg['test_instance']
    3. first cfg['num_test_instances'] training sequences
    """
    if cfg.get("test_instances") is not None:
        return [_normalize_sequence(seq) for seq in cfg["test_instances"]]
    if cfg.get("test_instance") is not None:
        return [_normalize_sequence(cfg["test_instance"])]
    n = cfg.get("num_test_instances", 10)
    return [_normalize_sequence(seq) for seq in X_train[:n]]


def run_one_language(lang_code: str, cfg: dict, output_root: str) -> dict | None:
    """Load one neural teacher and run all selected local instances."""
    print(f"\n{'=' * 70}")
    print(f"  LANGUAGE: {lang_code}")
    print(f"{'=' * 70}")

    model_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_classifier_trained.pth")
    split_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_train_test_split.pkl")

    if not (os.path.exists(model_path) and os.path.exists(split_path)):
        print(f"  [SKIP] Pre-trained model or split file not found for {lang_code}.")
        return None

    with open(split_path, "rb") as f:
        split = pickle.load(f)

    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    alphabet = sorted(set(tok for seq in X_train for tok in seq))

    clf = SequenceClassifier(
        max_len=cfg["max_length"],
        embedding_dim=cfg["embedding_dim"],
        device="cpu",
    )
    clf.load(model_path)
    predict_fn = lambda seqs: clf.predict(seqs)

    clf_train_acc = accuracy_score(y_train, predict_fn(X_train))
    clf_test_acc = accuracy_score(y_test, predict_fn(X_test))
    print(f"  Neural Network train={clf_train_acc:.4f}  test={clf_test_acc:.4f}")

    test_instances = get_test_instances(X_train, cfg)
    print(f"  Selected test instances: {len(test_instances)}")
    test_instance_labels = predict_fn(test_instances) if test_instances else np.array([])
    for i, (inst, label) in enumerate(zip(test_instances, test_instance_labels)):
        print(f"    [{i:02d}] len={len(inst)} label={label} seq={inst}")

    all_instance_results = {}
    def _run_instance(instance_idx: int, test_instance):
        result_key = f"{lang_code}_instance_{instance_idx:02d}"
        try:
            result = run_search_suite(
                predict_fn=predict_fn,
                alphabet=alphabet,
                test_instance=test_instance,
                cfg=cfg,
                output_dir=os.path.join(output_root, lang_code, f"instance_{instance_idx:02d}"),
                methods=("beam", "sa", "ga", "pso"),
                metadata={
                    "dataset": lang_code,
                    "instance_idx": instance_idx,
                    "teacher_type": "neural_classifier",
                    "clf_train_acc": float(clf_train_acc),
                    "clf_test_acc": float(clf_test_acc),
                    "agreement_threshold": cfg.get("agreement_threshold"),
                },
            )
            return result_key, result
        except Exception as exc:
            print(f"\n[ERROR] {result_key}: {exc}")
            print("[TRACEBACK]")
            traceback.print_exc(file=sys.stdout)
            return result_key, None

    if len(test_instances) > 1:
        max_workers = min(len(test_instances), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_instance, instance_idx, test_instance) for instance_idx, test_instance in enumerate(test_instances)]
            for fut in futures:
                result_key, result = fut.result()
                all_instance_results[result_key] = result
    else:
        for instance_idx, test_instance in enumerate(test_instances):
            result_key, result = _run_instance(instance_idx, test_instance)
            all_instance_results[result_key] = result

    return all_instance_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-world DFA search experiment")
    parser.add_argument("--languages", type=str, default="mnist,ECG,wafer")
    parser.add_argument("--agreement_threshold", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--init_num_samples", type=int, default=None)
    parser.add_argument("--edit_distance", type=int, default=None)
    parser.add_argument("--max_evaluations", type=int, default=None)
    parser.add_argument("--num_test_instances", type=int, default=None)
    parser.add_argument("--parallel", dest="parallel", action="store_true", default=None, help="Enable parallel KL-LUCB sampling/agreement evaluation.")
    parser.add_argument("--no_parallel", dest="parallel", action="store_false", help="Disable parallel KL-LUCB sampling/agreement evaluation.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of worker threads for KL-LUCB sampling/agreement evaluation.")
    parser.add_argument("--no_prediction_cache", dest="use_prediction_cache", action="store_false", default=None, help="Disable teacher prediction cache.")
    parser.add_argument("--prediction_cache_max_size", type=int, default=None, help="Maximum cached teacher predictions. Use 0 for unlimited.")
    parser.add_argument("--profile_time", dest="profile_time", action="store_true", default=None, help="Print a one-line profiling summary (time per search phase) after BeamSearch.")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "Appended to the auto-derived output folder name "
            "(test_result/realworld_{threshold}_{batch_size}{output_suffix}). "
            "Use this to avoid colliding with an existing run when overriding "
            "a parameter (e.g. --beam_size) that isn't part of the folder name."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "agreement_threshold": args.agreement_threshold,
        "delta": args.delta,
        "tau": args.tau,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "init_num_samples": args.init_num_samples,
        "edit_distance": args.edit_distance,
        "max_evaluations": args.max_evaluations,
        "num_test_instances": args.num_test_instances,
        "parallel": args.parallel,
        "n_jobs": args.n_jobs,
        "use_prediction_cache": args.use_prediction_cache,
        "prediction_cache_max_size": (None if args.prediction_cache_max_size == 0 else args.prediction_cache_max_size),
        "profile_time": args.profile_time,
    }

    languages = get_languages_config(overrides=overrides)
    selected_names = [name.strip() for name in args.languages.split(",") if name.strip()]
    unknown_names = [name for name in selected_names if name not in languages]
    if unknown_names:
        raise ValueError(
            f"Unknown dataset name(s): {unknown_names}\n"
            f"Available datasets: {list(languages.keys())}"
        )
    languages = {name: languages[name] for name in selected_names}

    first_cfg = next(iter(languages.values()))
    agreement_threshold = first_cfg["agreement_threshold"]
    batch_size = first_cfg["batch_size"]
    output_root = os.path.join(
        PROJECT_ROOT, "test_result", f"realworld_{agreement_threshold}_{batch_size}{args.output_suffix}"
    )
    if os.path.exists(output_root):
        raise FileExistsError(
            f"Output root already exists: {output_root}\n"
            "Refusing to run into an existing results folder (would overwrite "
            "prior results). Pass --output_suffix to pick a different folder, "
            "or remove/move the existing one first if you intend to replace it."
        )
    os.makedirs(output_root, exist_ok=True)
    log_path = os.path.join(output_root, "experiment_log.txt")

    print(f"\n{'=' * 70}")
    print("  Real-World DFA Search Experiment")
    print(f"  Selected datasets: {', '.join(languages.keys())}")
    print(f"  Agreement threshold: {agreement_threshold}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output directory: {output_root}")
    print(f"{'=' * 70}\n")

    original_stdout = sys.stdout
    tee = Tee(log_path)
    try:
        all_results: dict = {}
        for lang_code, cfg in languages.items():
            try:
                language_results = run_one_language(lang_code, cfg, output_root)
                if language_results:
                    all_results.update(language_results)
                else:
                    all_results[lang_code] = None
            except Exception as exc:
                print(f"\n[ERROR] {lang_code}: {exc}")
                print("[TRACEBACK]")
                traceback.print_exc(file=sys.stdout)
                all_results[lang_code] = None

        print_suite_summary(all_results)

        exclude_keys = {"test_instance", "test_instances"}
        print(f"\n{'=' * 70}")
        print("  Experiment Parameters")
        print(f"{'=' * 70}")
        print(f"  Selected datasets: {', '.join(languages.keys())}")
        for lang_code, cfg in languages.items():
            print(f"\n  [{lang_code}]")
            for key, value in sorted(cfg.items()):
                if key not in exclude_keys:
                    print(f"    {key}: {value}")
        print(f"{'=' * 70}")
    finally:
        tee.close()
        sys.stdout = original_stdout

    print(f"\nFull log → {log_path}")


if __name__ == "__main__":
    main()
