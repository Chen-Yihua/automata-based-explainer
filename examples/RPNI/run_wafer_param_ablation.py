"""
One-off wafer ablation: for one init_num_samples value, run beam search across
several batch_size values (default 500/1000/2000), everything else identical
to wafer's config in run_realworld_experiment.py. Beam search only -- no
SA/GA/PSO baselines.

Writes to test_result/wafer_ablation_<output_tag>/init<init_num_samples>/, so
multiple invocations (one per init_num_samples value) can share the same
--output_tag parent folder while writing to separate init<N> subfolders.
Refuses to run if that specific init<N> subfolder already exists, so it can
never overwrite another run's results.

Each batch_size variant's full log/artifacts go under
init<N>/batch<B>/experiment_log.txt; a consolidated summary across all
batch_size values for this init_num_samples is written to
init<N>/all_batch_sizes.txt.
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
    parser.add_argument("--init_num_samples", type=int, required=True)
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="500,1000,2000",
        help="Comma-separated batch_size values to run for this init_num_samples.",
    )
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
    parser.add_argument(
        "--output_tag",
        type=str,
        required=True,
        help=(
            "Shared parent folder: test_result/wafer_ablation_<output_tag>/. "
            "Multiple invocations (one per init_num_samples) can share the "
            "same tag; each writes to its own init<N> subfolder."
        ),
    )
    return parser.parse_args()


def run_one_batch_size(
    batch_size: int,
    init_dir: str,
    edit_distance: int | None,
    init_num_samples: int,
    init_state_max: int | None,
    predict_fn,
    alphabet,
    test_instance,
    clf_train_acc: float,
    clf_test_acc: float,
) -> dict:
    """Run beam search for one batch_size value; returns the beam result dict."""
    overrides = {
        "edit_distance": edit_distance,
        "init_num_samples": init_num_samples,
        "batch_size": batch_size,
    }
    cfg = get_languages_config(overrides=overrides)["wafer"]
    if init_state_max is not None:
        cfg["init_state_range"] = (25, init_state_max)

    variant_dir = os.path.join(init_dir, f"batch{batch_size}")
    os.makedirs(variant_dir, exist_ok=True)
    log_path = os.path.join(variant_dir, "experiment_log.txt")

    original_stdout = sys.stdout
    tee = Tee(log_path)
    try:
        print(f"{'=' * 70}\n  WAFER ABLATION (beam search only)\n{'=' * 70}")
        print(f"  edit_distance={cfg['edit_distance']}  init_num_samples={cfg['init_num_samples']}  batch_size={cfg['batch_size']}")
        print(f"  Neural network: train_acc={clf_train_acc:.4f}  test_acc={clf_test_acc:.4f}")
        print(f"  Test instance: len={len(test_instance)}  seq={test_instance}")

        result = run_search_suite(
            predict_fn=predict_fn,
            alphabet=alphabet,
            test_instance=test_instance,
            cfg=cfg,
            output_dir=os.path.join(variant_dir, "wafer", "instance_00"),
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

    print(f"  batch_size={batch_size} done -> {log_path}")
    return result["beam"]


def write_combined_summary(
    summary_path: str,
    init_num_samples: int,
    edit_distance: int,
    agreement_threshold: float,
    batch_results: list[tuple[int, dict]],
) -> None:
    """Write one file consolidating every batch_size variant for this init_num_samples, as a table."""
    header = ["batch_size", "Init States", "Train (Init->Final)", "Status", "Validation (Init->Final)", "Final States", "Time (s)"]
    rows = []
    for batch_size, beam in batch_results:
        rows.append([
            str(batch_size),
            str(beam.get("initial_states")),
            f"{beam.get('initial_train_agreement'):.4f}->{beam.get('train_agreement'):.4f}",
            "OK" if beam.get("success") else "FAIL",
            f"{beam.get('initial_validation_agreement'):.4f}->{beam.get('validation_agreement'):.4f}",
            str(beam.get("states")),
            f"{beam.get('time'):.1f}",
        ])
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def _fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"

    with open(summary_path, "w") as f:
        f.write(f"Wafer ablation: init_num_samples={init_num_samples}, batch_size in {{{', '.join(str(b) for b, _ in batch_results)}}}\n")
        f.write(f"edit_distance={edit_distance}, agreement_threshold={agreement_threshold}, methods=(beam,)\n\n")
        f.write(_fmt_row(header) + "\n")
        f.write(sep + "\n")
        for row in rows:
            f.write(_fmt_row(row) + "\n")
    print(f"\nCombined summary -> {summary_path}")


def main() -> None:
    args = parse_args()
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]

    output_root = os.path.join(PROJECT_ROOT, "test_result", f"wafer_ablation_{args.output_tag}")
    init_dir = os.path.join(output_root, f"init{args.init_num_samples}")
    if os.path.exists(init_dir):
        raise FileExistsError(
            f"Output folder already exists: {init_dir}\n"
            "Refusing to run into an existing results folder. Pick a different "
            "--output_tag, or remove/move the existing one first if you intend to replace it."
        )
    os.makedirs(init_dir)

    model_path = os.path.join(PROJECT_ROOT, "models", "wafer_classifier_trained.pth")
    split_path = os.path.join(PROJECT_ROOT, "models", "wafer_train_test_split.pkl")
    with open(split_path, "rb") as f:
        split = pickle.load(f)
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    alphabet = sorted(set(tok for seq in X_train for tok in seq))

    base_cfg = get_languages_config()["wafer"]
    clf = SequenceClassifier(max_len=base_cfg["max_length"], embedding_dim=base_cfg["embedding_dim"], device="cpu")
    clf.load(model_path)
    predict_fn = lambda seqs: clf.predict(seqs)

    clf_train_acc = accuracy_score(y_train, predict_fn(X_train))
    clf_test_acc = accuracy_score(y_test, predict_fn(X_test))
    test_instance = _normalize_sequence(base_cfg["test_instance"])

    print(f"init_num_samples={args.init_num_samples}: running batch_sizes={batch_sizes} -> {init_dir}")

    batch_results: list[tuple[int, dict]] = []
    for batch_size in batch_sizes:
        beam = run_one_batch_size(
            batch_size=batch_size,
            init_dir=init_dir,
            edit_distance=args.edit_distance,
            init_num_samples=args.init_num_samples,
            init_state_max=args.init_state_max,
            predict_fn=predict_fn,
            alphabet=alphabet,
            test_instance=test_instance,
            clf_train_acc=clf_train_acc,
            clf_test_acc=clf_test_acc,
        )
        batch_results.append((batch_size, beam))

    write_combined_summary(
        summary_path=os.path.join(init_dir, "all_batch_sizes.txt"),
        init_num_samples=args.init_num_samples,
        edit_distance=args.edit_distance or base_cfg["edit_distance"],
        agreement_threshold=base_cfg["agreement_threshold"],
        batch_results=batch_results,
    )


if __name__ == "__main__":
    main()
