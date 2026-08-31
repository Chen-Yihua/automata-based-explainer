"""
Regular automata experiment.

This script only loads DFA teachers and selects test instances.  The shared
Beam / SA / GA / PSO logic lives in experiments.runner.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import traceback

import numpy as np
import torch

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

from automaton.dfa_utils import get_alphabet
from automaton.load_dfa import create_automata_dfa_predictor, load_dfa_from_dot
from experiments.runner import run_search_suite, print_suite_summary
from tee import Tee


DEFAULT_LANGUAGE_CONFIGS = {
    "SecureHandshake": dict(
        automata_name="SecureHandshake",
        filename="secure_handshake.dot",
        agreement_threshold=0.9,
        delta=0.01,
        tau=0.1,
        batch_size=1000,
        beam_size=1,
        init_num_samples=1000,
        edit_distance=7,
        parallel=True,
        n_jobs=4,
        use_prediction_cache=True,
        prediction_cache_max_size=200000,
        num_test_instances=5,
        # test_instance=['hello','cert','verify','key','ack','key','ack','retry','hello','key','ack','retry','hello','key','ack'],
        # test_instance = ['hello', 'cert', 'verify', 'key', 'ack', 'key', 'ack', 'cert', 'verify', 'key', 'ack', 'cert', 'verify', 'key', 'ack', 'cert'],
        test_instance = ['hello', 'cert', 'verify', 'key', 'ack', 'key', 'ack', 'cert', 'verify', 'key', 'ack'],
        test_instances=None,
    ),
    "DocumentReleaseWorkflow": dict(
        automata_name="DocumentReleaseWorkflow",
        filename="document_release_workflow.dot",
        agreement_threshold=0.9,
        delta=0.01,
        tau=0.1,
        batch_size=1000,
        beam_size=1,
        init_num_samples=1000,
        edit_distance=5,
        parallel=True,
        n_jobs=4,
        use_prediction_cache=True,
        prediction_cache_max_size=200000,
        num_test_instances=10,
        test_instance=['draft', 'review', 'review', 'review', 'approve', 'comment', 'comment', 'approve', 'comment', 'publish'],
        test_instances=None,
    ),
    "MultiObligationOrder": dict(
        automata_name="MultiObligationOrder",
        filename="multi_obligation_color_order.dot",
        agreement_threshold=0.9,
        delta=0.01,
        tau=0.1,
        batch_size=1000,
        beam_size=1,
        init_num_samples=1000,
        edit_distance=5,
        parallel=True,
        n_jobs=4,
        use_prediction_cache=True,
        prediction_cache_max_size=200000,
        num_test_instances=10,
        test_instance=['pick', 'blue', 'green', 'move', 'drop', 'dock', 'yellow'],
        test_instances=None,
    ),
}


def get_languages_config(overrides=None):
    """Return automata configurations with optional CLI overrides."""
    import copy

    configs = copy.deepcopy(DEFAULT_LANGUAGE_CONFIGS)
    if overrides:
        for cfg in configs.values():
            for key, value in overrides.items():
                if value is not None:
                    cfg[key] = value
    return configs


def random_walk_from_dfa(dfa, alphabet, max_length=10, min_length=1):
    """Generate one sequence by random walk from the teacher DFA."""
    if dfa.initial_state is None:
        raise ValueError("DFA has no initial state")

    curr_state = dfa.initial_state
    length = random.randint(min_length, max_length)
    seq = []
    for _ in range(length):
        transitions = getattr(curr_state, "transitions", {})
        if transitions:
            symbol = random.choice(list(transitions.keys()))
            curr_state = transitions[symbol]
        else:
            symbol = random.choice(alphabet)
        seq.append(symbol)
    return seq


def get_test_instances(
    teacher,
    alphabet,
    predict_fn,
    n=10,
    max_length=10,
    min_length=1,
    seed=42,
    desired_label=1,
    max_attempts=None,
):
    """
    Generate regular test instances from the teacher DFA.

    By default, only accepting examples are selected.  The actual explanation
    labels are still local agreement labels inside the sampler/runner.
    """
    random.seed(seed)
    max_attempts = max_attempts or n * 1000

    test_instances = []
    seen = set()
    attempts = 0
    while len(test_instances) < n and attempts < max_attempts:
        attempts += 1
        seq = random_walk_from_dfa(teacher, alphabet, max_length=max_length, min_length=min_length)
        key = tuple(seq)
        if key in seen:
            continue
        label = int(predict_fn([seq])[0])
        if desired_label is not None and label != desired_label:
            continue
        seen.add(key)
        test_instances.append(seq)

    if len(test_instances) < n:
        print(
            f"  [WARNING] Only generated {len(test_instances)} "
            f"test instances with desired_label={desired_label} after {attempts} attempts."
        )
    return test_instances


def _ensure_teacher_initial_state(teacher) -> None:
    """Repair missing initial_state metadata for older DOT files."""
    if teacher.initial_state is not None and hasattr(teacher.initial_state, "state_id"):
        return
    for state_name in ["S0", "Start_A0C0H0", "P00_0", "B0", "q0"]:
        candidate = next((s for s in teacher.states if s.state_id == state_name), None)
        if candidate:
            teacher.initial_state = candidate
            print(f"  [INFO] Set initial state to: {state_name}")
            return
    if getattr(teacher, "states", None):
        teacher.initial_state = teacher.states[0]
        print(f"  [INFO] Set initial state to first state: {teacher.states[0].state_id}")


def run_one_automata(automata_code: str, cfg: dict, output_root: str) -> dict | None:
    print(f"\n{'=' * 70}")
    print(f"  AUTOMATA: {automata_code}")
    print(f"{'=' * 70}")

    try:
        print(f"  [TEACHER] Loading automata DFA: {cfg['filename']}")
        teacher = load_dfa_from_dot(cfg["filename"])
        _ensure_teacher_initial_state(teacher)
        alphabet = get_alphabet(teacher)
        predict_fn = create_automata_dfa_predictor(teacher)
        print(f"  DFA: {len(teacher.states)} states, alphabet={alphabet}")
    except Exception as exc:
        print(f"  [ERROR] Failed to load automata DFA: {exc}")
        traceback.print_exc(file=sys.stdout)
        return None

    if cfg.get("test_instances") is not None:
        test_instances = [list(seq) for seq in cfg["test_instances"]]
    elif cfg.get("test_instance") is not None:
        test_instances = [list(cfg["test_instance"])]
    else:
        test_instances = get_test_instances(
            teacher=teacher,
            alphabet=alphabet,
            predict_fn=predict_fn,
            n=cfg.get("num_test_instances", 10),
            max_length=cfg.get("max_length", 20),
            min_length=cfg.get("min_test_length", 10),
            seed=cfg.get("seed", 42),
            desired_label=cfg.get("desired_label", 1),
        )

    print(f"  Selected test instances: {len(test_instances)}")
    for i, inst in enumerate(test_instances):
        print(f"    [{i:02d}] len={len(inst)} label={predict_fn([inst])[0]} seq={inst}")

    all_instance_results = {}
    for instance_idx, test_instance in enumerate(test_instances):
        result_key = f"{automata_code}_instance_{instance_idx:02d}"
        try:
            all_instance_results[result_key] = run_search_suite(
                predict_fn=predict_fn,
                alphabet=alphabet,
                test_instance=test_instance,
                cfg=cfg,
                output_dir=os.path.join(output_root, automata_code, f"instance_{instance_idx:02d}"),
                methods=("beam", "sa", "ga", "pso"),
                metadata={
                    "dataset": automata_code,
                    "instance_idx": instance_idx,
                    "teacher_type": "dfa",
                    "teacher_states": len(getattr(teacher, "states", [])),
                    "agreement_threshold": cfg.get("agreement_threshold"),
                },
            )
        except Exception as exc:
            print(f"\n[ERROR] {result_key}: {exc}")
            print("[TRACEBACK]")
            traceback.print_exc(file=sys.stdout)
            all_instance_results[result_key] = None

    return all_instance_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run regular automata DFA search experiment")
    parser.add_argument("--languages", type=str, default="SecureHandshake,DocumentReleaseWorkflow,MultiObligationOrder")
    parser.add_argument("--agreement_threshold", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--init_num_samples", type=int, default=None)
    parser.add_argument("--edit_distance", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_evaluations", type=int, default=None)
    parser.add_argument("--num_test_instances", type=int, default=None)
    parser.add_argument("--parallel", dest="parallel", action="store_true", default=None, help="Enable parallel KL-LUCB sampling/agreement evaluation.")
    parser.add_argument("--no_parallel", dest="parallel", action="store_false", help="Disable parallel KL-LUCB sampling/agreement evaluation.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of worker threads for KL-LUCB sampling/agreement evaluation.")
    parser.add_argument("--no_prediction_cache", dest="use_prediction_cache", action="store_false", default=None, help="Disable teacher prediction cache.")
    parser.add_argument("--prediction_cache_max_size", type=int, default=None, help="Maximum cached teacher predictions. Use 0 for unlimited.")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "Appended to the auto-derived output folder name "
            "(test_result/regular_{threshold}_{batch_size}{output_suffix}). "
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
        "max_length": args.max_length,
        "max_evaluations": args.max_evaluations,
        "num_test_instances": args.num_test_instances,
        "parallel": args.parallel,
        "n_jobs": args.n_jobs,
        "use_prediction_cache": args.use_prediction_cache,
        "prediction_cache_max_size": (None if args.prediction_cache_max_size == 0 else args.prediction_cache_max_size),
    }

    automata = get_languages_config(overrides=overrides)
    selected_names = [name.strip() for name in args.languages.split(",") if name.strip()]
    unknown_names = [name for name in selected_names if name not in automata]
    if unknown_names:
        raise ValueError(
            f"Unknown automata name(s): {unknown_names}\n"
            f"Available automata: {list(automata.keys())}"
        )
    automata = {name: automata[name] for name in selected_names}

    first_cfg = next(iter(automata.values()))
    agreement_threshold = first_cfg["agreement_threshold"]
    batch_size = first_cfg["batch_size"]
    output_root = os.path.join(
        PROJECT_ROOT, "test_result", f"regular_{agreement_threshold}_{batch_size}{args.output_suffix}"
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
    print("  Regular Automata DFA Search Experiment")
    print(f"  Selected automata: {', '.join(automata.keys())}")
    print(f"  Agreement threshold: {agreement_threshold}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output directory: {output_root}")
    print(f"{'=' * 70}\n")

    original_stdout = sys.stdout
    sys.stdout = Tee(log_path)
    try:
        all_results: dict = {}
        for automata_code, cfg in automata.items():
            try:
                automata_results = run_one_automata(automata_code, cfg, output_root)
                if automata_results:
                    all_results.update(automata_results)
                else:
                    all_results[automata_code] = None
            except Exception as exc:
                print(f"\n[ERROR] {automata_code}: {exc}")
                print("[TRACEBACK]")
                traceback.print_exc(file=sys.stdout)
                all_results[automata_code] = None

        print_suite_summary(all_results)

        exclude_keys = {"automata_name", "filename", "test_instance", "test_instances"}
        print(f"\n{'=' * 70}")
        print("  Experiment Parameters")
        print(f"{'=' * 70}")
        print(f"  Selected automata: {', '.join(automata.keys())}")
        for automata_code, cfg in automata.items():
            print(f"\n  [{automata_code}]")
            for key, value in sorted(cfg.items()):
                if key not in exclude_keys:
                    print(f"    {key}: {value}")
        print(f"{'=' * 70}")
    finally:
        sys.stdout = original_stdout

    print(f"\nFull log → {log_path}")


if __name__ == "__main__":
    main()

