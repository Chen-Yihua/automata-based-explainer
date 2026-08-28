"""
One-off wafer ablation: vary a single hyperparameter (edit_distance or
init_num_samples) in isolation, everything else identical to wafer's config
in run_realworld_experiment.py. Beam search only -- no SA/GA/PSO baselines.

Writes to test_result/wafer_ablation_<output_tag>/, refusing to run if that
folder already exists so it can never overwrite another experiment's results.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score

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
from run_realworld_experiment import get_languages_config  # reuse wafer's exact base config
from tee import Tee


def _normalize_sequence(seq):
    if isinstance(seq, np.ndarray):
        return seq.tolist()
    return list(seq)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-off wafer hyperparameter ablation, beam search only")
    parser.add_argument("--edit_distance", type=int, default=None)
    parser.add_argument("--init_num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--init_state_max",
        type=int,
        default=None,
        help=(
            "Upper bound of the initial-DFA state-count range (default 65, "
            "see automata_beam.py). Only affects this ablation script; the "
            "shared default is untouched for every other experiment."
        ),
    )
    parser.add_argument("--output_tag", type=str, required=True, help="Suffix for test_result/wafer_ablation_<output_tag>/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "edit_distance": args.edit_distance,
        "init_num_samples": args.init_num_samples,
        "batch_size": args.batch_size,
    }
    cfg = get_languages_config(overrides=overrides)["wafer"]
    if args.init_state_max is not None:
        cfg["init_state_range"] = (25, args.init_state_max)

    output_root = os.path.join(PROJECT_ROOT, "test_result", f"wafer_ablation_{args.output_tag}")
    if os.path.exists(output_root):
        raise FileExistsError(
            f"Output root already exists: {output_root}\n"
            "Refusing to run into an existing results folder. Pick a different --output_tag."
        )
    os.makedirs(output_root)
    log_path = os.path.join(output_root, "experiment_log.txt")

    model_path = os.path.join(PROJECT_ROOT, "models", "wafer_classifier_trained.pth")
    split_path = os.path.join(PROJECT_ROOT, "models", "wafer_train_test_split.pkl")
    with open(split_path, "rb") as f:
        split = pickle.load(f)
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    alphabet = sorted(set(tok for seq in X_train for tok in seq))

    clf = SequenceClassifier(max_len=cfg["max_length"], embedding_dim=cfg["embedding_dim"], device="cpu")
    clf.load(model_path)
    predict_fn = lambda seqs: clf.predict(seqs)

    clf_train_acc = accuracy_score(y_train, predict_fn(X_train))
    clf_test_acc = accuracy_score(y_test, predict_fn(X_test))

    test_instance = _normalize_sequence(cfg["test_instance"])

    original_stdout = sys.stdout
    tee = Tee(log_path)
    try:
        print(f"{'=' * 70}\n  WAFER ABLATION (beam search only)\n{'=' * 70}")
        print(f"  edit_distance={cfg['edit_distance']}  init_num_samples={cfg['init_num_samples']}")
        print(f"  Neural network: train_acc={clf_train_acc:.4f}  test_acc={clf_test_acc:.4f}")
        print(f"  Test instance: len={len(test_instance)}  seq={test_instance}")

        result = run_search_suite(
            predict_fn=predict_fn,
            alphabet=alphabet,
            test_instance=test_instance,
            cfg=cfg,
            output_dir=os.path.join(output_root, "wafer", "instance_00"),
            methods=("beam",),
            metadata={
                "dataset": "wafer",
                "instance_idx": 0,
                "teacher_type": "neural_classifier",
                "clf_train_acc": float(clf_train_acc),
                "clf_test_acc": float(clf_test_acc),
                "agreement_threshold": cfg.get("agreement_threshold"),
            },
        )
        print_suite_summary({"wafer_instance_00": result})
    finally:
        tee.close()
        sys.stdout = original_stdout

    print(f"\nFull log → {log_path}")


if __name__ == "__main__":
    main()
