"""
KL-LUCB ablation for local DFA explanations.

This script compares Beam Search WITH KL-LUCB and WITHOUT KL-LUCB under
the same initial DFA, same validation samples, and same explained instance.

Regular automata and real-world datasets are unified as black-box teachers:
the search only sees `predict_fn(sequences) -> labels`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import random
import sys
import time
import traceback
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score

# ── Path setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, "external_modules")
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, "Explaining-FA")

for _p in [SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from automaton.dfa_utils import get_alphabet
from automaton.load_dfa import (
    create_automata_dfa_predictor,
    load_dfa_from_dot,
)
from explainer.automata_beam import AutomataBeamSearch
from learner.dfa_learner import DFALearner, DFASampler
from models.sequence_classifier import SequenceClassifier
from tee import Tee


# ======================================================================
# Experiment configurations
# ======================================================================

# Kept identical to DEFAULT_LANGUAGE_CONFIGS in run_regular_experiment.py and
# run_realworld_experiment.py, so the with/without KL-LUCB ablation runs under
# the exact same task setup (same test instance, thresholds, batch size, edit
# distance, etc.) as the main experiments -- update all three together.
DEFAULT_LANGUAGE_CONFIGS = {
    # ────── Real-world datasets (from run_realworld_experiment.py) ──────
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
        test_instance=['R', 'R', 'R', 'R', 'D', 'D', 'L', 'D', 'D', 'L', 'D', 'D'],
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
    # ────── Regular automata teachers (from run_regular_experiment.py) ──────
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
        test_instance=['hello', 'cert', 'verify', 'key', 'ack', 'key', 'ack', 'cert', 'verify', 'key', 'ack'],
        test_instances=None,
        max_length=20,
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
        max_length=20,
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
        max_length=20,
    ),
}


def get_languages_config(overrides: Optional[dict] = None) -> dict:
    """Return deep-copied configs with optional CLI overrides."""
    import copy

    configs = copy.deepcopy(DEFAULT_LANGUAGE_CONFIGS)
    if overrides:
        for cfg in configs.values():
            for key, value in overrides.items():
                if value is not None:
                    cfg[key] = value
    return configs


# ======================================================================
# Generic helpers
# ======================================================================

PrebuiltInit = namedtuple(
    "PrebuiltInit",
    ["learner", "initial_dfa", "validation_data", "validation_labels"],
)


def is_regular_config(cfg: dict) -> bool:
    """Regular automata configs have a DOT filename."""
    return "filename" in cfg


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_scalar(value: Any) -> Any:
    """Convert [x], np.array([x]), and numpy scalar values into Python scalars."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        if len(value) == 1:
            return to_scalar(value[0])
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = to_scalar(value)
        return default if scalar is None else float(scalar)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        scalar = to_scalar(value)
        return default if scalar is None else int(scalar)
    except Exception:
        return default


def maybe_len_states(automaton: Any) -> int:
    if automaton is None:
        return 0
    if hasattr(automaton, "size"):
        return int(automaton.size)
    if hasattr(automaton, "states"):
        return int(len(automaton.states))
    return 0


def normalize_sequence(seq: Sequence[Any]) -> list:
    if isinstance(seq, np.ndarray):
        return seq.tolist()
    return list(seq)


def behavior_signature_from_preds(preds: Sequence[int]) -> str:
    arr = np.asarray(preds, dtype=np.uint8)
    return hashlib.md5(arr.tobytes()).hexdigest()


# ======================================================================
# DFA evaluation helpers
# ======================================================================

def dfa_accepts_sequence(dfa: Any, sequence: Sequence[Any]) -> bool:
    """Evaluate an AALpy-like DFA on one sequence without mutating current_state."""
    state = getattr(dfa, "initial_state", None)
    if state is None:
        return False

    for sym in sequence:
        transitions = getattr(state, "transitions", None)
        if transitions is None or sym not in transitions:
            return False
        state = transitions[sym]

    return bool(getattr(state, "is_accepting", False))


def dfa_predict_batch(dfa: Any, sequences: Sequence[Sequence[Any]]) -> np.ndarray:
    return np.asarray([1 if dfa_accepts_sequence(dfa, seq) else 0 for seq in sequences], dtype=int)


def extract_automata_pair(exp_obj: Any) -> Optional[Tuple[Any, Any]]:
    """
    Recover [initial_dfa, final_dfa] from an AutomataBeamSearch.automata_beam()
    result dict (its "automata" key holds exactly this pair).
    """
    if isinstance(exp_obj, dict):
        automata = exp_obj.get("automata")
        if isinstance(automata, (list, tuple)) and len(automata) == 2:
            return automata[0], automata[1]

    return None


def evaluate_final_dfa(
    exp_obj: Any,
    holdout_X: Sequence[Sequence[Any]],
    holdout_y: np.ndarray,
    fallback_validation_agreement: float,
) -> Dict[str, Any]:
    """
    Evaluate the final DFA on an external hold-out set.
    If the final DFA object is unavailable, fall back to final_validation_agreement.
    """
    pair = extract_automata_pair(exp_obj)
    if pair is None:
        return {
            "external_holdout_agreement": fallback_validation_agreement,
            "behavior_signature": None,
            "used_fallback_validation_agreement": True,
            "final_dfa_exposed": False,
        }

    _, final_dfa = pair
    preds = dfa_predict_batch(final_dfa, holdout_X)
    acc = float(np.mean(preds == holdout_y)) if len(holdout_y) > 0 else 0.0
    return {
        "external_holdout_agreement": acc,
        "behavior_signature": behavior_signature_from_preds(preds),
        "used_fallback_validation_agreement": False,
        "final_dfa_exposed": True,
    }


# ======================================================================
# Teacher/data construction
# ======================================================================

def random_sequences(alphabet: Sequence[Any], n: int, min_len: int, max_len: int) -> list[list[Any]]:
    """Generate random sequences from an alphabet."""
    max_len = max(min_len, max_len)
    sequences = []
    for _ in range(n):
        seq_len = random.randint(min_len, max_len)
        sequences.append([random.choice(list(alphabet)) for _ in range(seq_len)])
    return sequences


def choose_regular_instance(
    predict_fn: Callable,
    alphabet: Sequence[Any],
    cfg: dict,
    seed: int,
    desired_label: Optional[int] = 1,
) -> list:
    """Choose or generate a regular automata instance for local explanation."""
    if cfg.get("test_instance") is not None:
        return normalize_sequence(cfg["test_instance"])

    if cfg.get("test_instances"):
        instances = [normalize_sequence(seq) for seq in cfg["test_instances"]]
        return instances[seed % len(instances)]

    random.seed(seed)
    max_attempts = cfg.get("instance_max_attempts", 2000)
    min_len = cfg.get("min_test_length", 2)
    max_len = cfg.get("max_length", 20)

    fallback = None
    for _ in range(max_attempts):
        seq = random_sequences(alphabet, n=1, min_len=min_len, max_len=max_len)[0]
        fallback = seq
        label = int(np.asarray(predict_fn([seq]))[0])
        if desired_label is None or label == desired_label:
            return seq

    print(
        f"  [WARN] Could not find desired_label={desired_label} after {max_attempts} attempts; "
        "using last generated sequence."
    )
    return fallback or []


def build_regular_context(lang_code: str, cfg: dict, seed: int) -> Optional[dict]:
    """Build black-box teacher context from a DOT DFA."""
    try:
        teacher = load_dfa_from_dot(cfg["filename"])
        if teacher.initial_state is None or not hasattr(teacher.initial_state, "state_id"):
            teacher.initial_state = next(iter(teacher.states), None)

        alphabet = get_alphabet(teacher)
        predict_fn = create_automata_dfa_predictor(teacher)
        test_instance = choose_regular_instance(predict_fn, alphabet, cfg, seed)

        train_size = cfg.get("synthetic_train_size", 100)
        test_size = cfg.get("synthetic_test_size", 50)
        max_len = cfg.get("max_length", 20)
        X_train = random_sequences(alphabet, train_size, min_len=2, max_len=max_len)
        X_test = random_sequences(alphabet, test_size, min_len=2, max_len=max_len)
        y_train = predict_fn(X_train)
        y_test = predict_fn(X_test)

        print(f"  Teacher DFA: {lang_code}")
        print(f"    states={len(teacher.states)}, alphabet={alphabet}")
        print(f"    train acceptance rate={float(np.mean(y_train)):.4f}, test acceptance rate={float(np.mean(y_test)):.4f}")

        return {
            "dataset": lang_code,
            "teacher_type": "dfa",
            "teacher_states": len(teacher.states),
            "teacher_train_acc": 1.0,
            "teacher_test_acc": 1.0,
            "predict_fn": predict_fn,
            "alphabet": alphabet,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_instance": test_instance,
            "feature_names": ["seq_token"],
            "d_train_data": None,
        }
    except Exception as exc:
        print(f"  [ERROR] Failed to build regular context for {lang_code}: {exc}")
        traceback.print_exc()
        return None


def choose_realworld_instance(X_train: Sequence[Sequence[Any]], y_train: Sequence[int], cfg: dict, lang_code: str, seed: int) -> list:
    """Choose a real-world instance for local explanation."""
    if cfg.get("test_instance") is not None:
        return normalize_sequence(cfg["test_instance"])

    if cfg.get("test_instances"):
        instances = [normalize_sequence(seq) for seq in cfg["test_instances"]]
        return instances[seed % len(instances)]

    # Preserve the previous script's stable defaults.
    if lang_code == "mnist" and len(X_train) > 23:
        return normalize_sequence(X_train[23])
    if lang_code in {"ECG", "wafer"} and len(X_train) > 0:
        return normalize_sequence(X_train[0])

    positive_indices = [i for i, label in enumerate(y_train) if int(label) == 1]
    if len(positive_indices) > 2:
        return normalize_sequence(X_train[positive_indices[2]])
    if positive_indices:
        return normalize_sequence(X_train[positive_indices[0]])
    return normalize_sequence(X_train[0])


def build_realworld_context(lang_code: str, cfg: dict, seed: int) -> Optional[dict]:
    """Build black-box teacher context from a trained sequence classifier."""
    model_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_classifier_trained.pth")
    split_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_train_test_split.pkl")
    if not (os.path.exists(model_path) and os.path.exists(split_path)):
        print(f"  [SKIP] Pre-trained model or data split not found for {lang_code}.")
        return None

    with open(split_path, "rb") as f:
        split = pickle.load(f)

    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]

    clf = SequenceClassifier(
        max_len=cfg["max_length"],
        embedding_dim=cfg["embedding_dim"],
        device="cpu",
    )
    clf.load(model_path)
    predict_fn = lambda seqs: clf.predict(seqs)

    clf_train_acc = accuracy_score(y_train, predict_fn(X_train))
    clf_test_acc = accuracy_score(y_test, predict_fn(X_test))
    alphabet = cfg.get("alphabet") or sorted(set(tok for seq in X_train for tok in seq))
    test_instance = choose_realworld_instance(X_train, y_train, cfg, lang_code, seed)
    max_feature_len = max(len(s) for s in X_train) if len(X_train) > 0 else len(test_instance)

    print(f"  Sequence classifier: {lang_code}")
    print(f"    train_acc={clf_train_acc:.4f}, test_acc={clf_test_acc:.4f}")
    print(f"    alphabet={alphabet}")

    return {
        "dataset": lang_code,
        "teacher_type": "neural_classifier",
        "teacher_states": None,
        "teacher_train_acc": clf_train_acc,
        "teacher_test_acc": clf_test_acc,
        "predict_fn": predict_fn,
        "alphabet": alphabet,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_instance": test_instance,
        "feature_names": [f"pos_{i}" for i in range(max_feature_len)],
        "d_train_data": X_train,
    }


def build_teacher_context(lang_code: str, cfg: dict, seed: int) -> Optional[dict]:
    """
    Build a unified teacher context.
    Regular and real-world datasets differ only in how predict_fn is created.
    """
    if is_regular_config(cfg):
        return build_regular_context(lang_code, cfg, seed)
    return build_realworld_context(lang_code, cfg, seed)


# ======================================================================
# Explainer / beam execution
# ======================================================================

def create_explainer(
    context: dict,
    cfg: dict,
    seed: int,
) -> tuple[Any, Callable]:
    """
    Create and bind a DFASampler for the explained instance.

    Same construction as experiments.runner.make_sampler() (used by
    run_regular_experiment.py / run_realworld_experiment.py), inlined here so
    the per-run `seed` can be threaded through -- make_sampler() only reads
    seed from cfg. The sampler doubles as the "explainer" handle: it is both
    what AutomataBeamSearch samples from and the callable used to draw the
    external hold-out set below.
    """
    sampler = DFASampler(
        predictor=context["predict_fn"],
        alphabet=list(context["alphabet"]),
        seed=seed,
        edit_distance=cfg["edit_distance"],
        use_prediction_cache=cfg.get("use_prediction_cache", True),
        prediction_cache_max_size=cfg.get("prediction_cache_max_size", 200000),
    )
    sampler.set_instance_label(list(context["test_instance"]))
    sampler.set_n_covered(cfg.get("n_covered", 10))

    return sampler, sampler


def run_beam_once(
    explainer: Any,
    context: dict,
    cfg: dict,
    out_dir: str,
    use_kllucb: bool,
    prebuilt_init: Optional[PrebuiltInit],
) -> dict:
    """
    Run one beam-search variant and collect metrics.

    Same direct-call pattern as experiments.runner.run_beam() (used by
    run_regular_experiment.py / run_realworld_experiment.py): construct
    AutomataBeamSearch around the sampler built by create_explainer() and
    call .automata_beam() -- no AnchorTabular/AnchorBaseBeam involved (that
    generic tabular-anchor path never actually routes through
    AutomataBeamSearch and cannot see automaton_type/alphabet/edit_distance
    kwargs at all; the "explainer.explain(...)" call this replaced was
    silently broken).
    """
    sampler = explainer  # create_explainer() returns the DFASampler itself
    variant_name = "with_kllucb" if use_kllucb else "no_kllucb"
    print(f"\n  Beam Search {'WITH' if use_kllucb else 'WITHOUT'} KL-LUCB")

    search = AutomataBeamSearch(
        samplers=[sampler],
        predictor=sampler.predictor,
        sample_cache_size=cfg.get("sample_cache_size", 1000),
        parallel=cfg.get("parallel", False),
        n_jobs=cfg.get("n_jobs", 4),
    )

    t0 = time.time()
    result = search.automata_beam(
        data_type="Tabular",
        automaton_type="DFA",
        alphabet=list(context["alphabet"]),
        agreement_threshold=cfg["agreement_threshold"],
        delta=cfg["delta"],
        epsilon=cfg["tau"],
        epsilon_stop=cfg.get("epsilon_stop", 0.05),
        beam_size=cfg["beam_size"],
        batch_size=cfg["batch_size"],
        init_num_samples=cfg["init_num_samples"],
        max_evaluations=cfg.get("max_evaluations"),
        verbose=False,
        output_dir=os.path.join(out_dir, variant_name),
        use_kllucb=use_kllucb,
        prebuilt_init=prebuilt_init,
        save_graphs=cfg.get("save_graphs", True),
        save_plots=cfg.get("save_plots", True),
        collect_error_examples=cfg.get("collect_error_examples", False),
    )
    time_total = time.time() - t0

    init_time = safe_float(result.get("init_automaton_time", 0.0), 0.0)
    beam_time = max(0.0, time_total - init_time)

    return {
        "exp": result,
        "initial_state": safe_int(result.get("initial_state", 0), 0),
        "initial_training_agreement": safe_float(result.get("initial_training_agreement", 0.0), 0.0),
        "initial_validation_agreement": safe_float(result.get("initial_validation_agreement", 0.0), 0.0),
        "final_state": safe_int(result.get("final_state", 0), 0),
        "final_training_agreement": safe_float(result.get("final_training_agreement", 0.0), 0.0),
        "final_validation_agreement": safe_float(result.get("final_validation_agreement", 0.0), 0.0),
        "time_total": time_total,
        "init_time": init_time,
        "beam_time": beam_time,
        "success": bool(result.get("success", False)),
        "budget_used": result.get("budget_used", None),
        "automata": result.get("automata"),
        "validation_data": result.get("validation_data"),
        "validation_labels": result.get("validation_labels"),
    }


def extract_prebuilt_init(with_raw: dict) -> Optional[PrebuiltInit]:
    """Extract the initial DFA and validation set from a completed WITH-KL run."""
    automata_pair = with_raw.get("automata")
    if not automata_pair:
        print("  [ERROR] Could not extract initial DFA from beam search result")
        return None

    initial_dfa = automata_pair[0].copy()
    validation_data = list(with_raw.get("validation_data") or [])
    validation_labels = np.asarray(with_raw.get("validation_labels", []))

    print("  Extracting shared initialization...")
    print(
        f"  Extracted {len(validation_data)} validation samples, "
        f"DFA with {maybe_len_states(initial_dfa)} states"
    )

    return PrebuiltInit(
        learner=DFALearner(),
        initial_dfa=initial_dfa,
        validation_data=validation_data,
        validation_labels=validation_labels,
    )


# ======================================================================
# Per-seed and multi-seed experiment
# ======================================================================

def run_one_language_seed(
    lang_code: str,
    cfg: dict,
    output_root: str,
    seed: int,
    holdout_size: int,
) -> Optional[dict]:
    """Run one seed for one dataset/automaton."""
    set_all_seeds(seed)

    print(f"\n{'─' * 80}")
    print(f"  DATASET={lang_code} | SEED={seed}")
    print(f"{'─' * 80}")

    out_dir = os.path.join(output_root, lang_code, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    context = build_teacher_context(lang_code, cfg, seed)
    if context is None:
        return None

    label = int(np.asarray(context["predict_fn"]([context["test_instance"]]))[0])
    print(f"  Test instance: len={len(context['test_instance'])}, label={label}, seq={context['test_instance']}")

    # Create hold-out samples once; use the same hold-out set for both variants.
    holdout_explainer, holdout_sampler = create_explainer(context, cfg, seed)
    holdout_X, holdout_y = holdout_sampler(num_samples=holdout_size, compute_labels=True)
    holdout_X = list(holdout_X)
    holdout_y = np.asarray(holdout_y, dtype=int)

    results: Dict[str, Any] = {
        "dataset": lang_code,
        "teacher_type": context["teacher_type"],
        "teacher_states": context["teacher_states"],
        "teacher_train_acc": context["teacher_train_acc"],
        "teacher_test_acc": context["teacher_test_acc"],
        "seed": seed,
        "batch_size": int(cfg["batch_size"]),
        "holdout_size": holdout_size,
    }

    # WITH KL-LUCB
    with_explainer, _ = create_explainer(context, cfg, seed)
    try:
        with_raw = run_beam_once(
            explainer=with_explainer,
            context=context,
            cfg=cfg,
            out_dir=out_dir,
            use_kllucb=True,
            prebuilt_init=None,
        )
    except Exception as exc:
        print(f"  [ERROR] WITH KL-LUCB failed: {exc}")
        traceback.print_exc()
        return None

    prebuilt = extract_prebuilt_init(with_raw)
    if prebuilt is None:
        return None

    eval_with = evaluate_final_dfa(
        with_raw["exp"],
        holdout_X,
        holdout_y,
        fallback_validation_agreement=with_raw["final_validation_agreement"],
    )

    results["with_kllucb"] = {
        key: value
        for key, value in with_raw.items()
        if key != "exp"
    }
    results["with_kllucb"].update(
        {
            "external_holdout_agreement": eval_with["external_holdout_agreement"],
            "behavior_signature": eval_with["behavior_signature"],
            "used_fallback_validation_agreement": eval_with["used_fallback_validation_agreement"],
            "final_dfa_exposed": eval_with["final_dfa_exposed"],
            "shared_init_used": True,
        }
    )

    print(
        f"    WITH KL-LUCB: train={with_raw['final_training_agreement']:.4f}, "
        f"val={with_raw['final_validation_agreement']:.4f}, "
        f"holdout={eval_with['external_holdout_agreement']:.4f}, "
        f"states={with_raw['final_state']}, beam_time={with_raw['beam_time']:.1f}s"
    )

    # WITHOUT KL-LUCB, using same prebuilt initialization from WITH run.
    no_explainer, _ = create_explainer(context, cfg, seed)
    try:
        no_raw = run_beam_once(
            explainer=no_explainer,
            context=context,
            cfg=cfg,
            out_dir=out_dir,
            use_kllucb=False,
            prebuilt_init=prebuilt,
        )
    except Exception as exc:
        print(f"  [ERROR] WITHOUT KL-LUCB failed: {exc}")
        traceback.print_exc()
        return None

    eval_no = evaluate_final_dfa(
        no_raw["exp"],
        holdout_X,
        holdout_y,
        fallback_validation_agreement=no_raw["final_validation_agreement"],
    )

    results["no_kllucb"] = {
        key: value
        for key, value in no_raw.items()
        if key != "exp"
    }
    results["no_kllucb"].update(
        {
            "external_holdout_agreement": eval_no["external_holdout_agreement"],
            "behavior_signature": eval_no["behavior_signature"],
            "used_fallback_validation_agreement": eval_no["used_fallback_validation_agreement"],
            "final_dfa_exposed": eval_no["final_dfa_exposed"],
            "shared_init_used": True,
        }
    )

    print(
        f"    NO KL-LUCB:   train={no_raw['final_training_agreement']:.4f}, "
        f"val={no_raw['final_validation_agreement']:.4f}, "
        f"holdout={eval_no['external_holdout_agreement']:.4f}, "
        f"states={no_raw['final_state']}, beam_time={no_raw['beam_time']:.1f}s"
    )

    if not eval_with["final_dfa_exposed"] or not eval_no["final_dfa_exposed"]:
        print(
            "    [WARN] Final DFA object not exposed; holdout_agreement uses "
            "final_validation_agreement fallback for at least one variant."
        )

    return results


def summarize_language_results(lang_code: str, cfg: dict, seed_results: list[dict], holdout_size: int) -> dict:
    """Aggregate multiple seed-level results into one language summary."""
    with_accs, no_accs = [], []
    with_states, no_states = [], []
    with_success_count, no_success_count = 0, 0
    with_win_count, no_win_count, tie_count = 0, 0, 0
    with_sigs, no_sigs = [], []
    stats_arrays = {
        "with_kllucb": {
            "init_train": [],
            "init_val": [],
            "final_train": [],
            "final_val": [],
            "beam_time": [],
        },
        "no_kllucb": {
            "init_train": [],
            "init_val": [],
            "final_train": [],
            "final_val": [],
            "beam_time": [],
        },
    }

    threshold = cfg.get("agreement_threshold", 0.8)

    for res in seed_results:
        with_r = res["with_kllucb"]
        no_r = res["no_kllucb"]

        with_acc = safe_float(with_r.get("external_holdout_agreement", 0.0), 0.0)
        no_acc = safe_float(no_r.get("external_holdout_agreement", 0.0), 0.0)
        with_state = safe_int(with_r.get("final_state", 0), 0)
        no_state = safe_int(no_r.get("final_state", 0), 0)

        with_accs.append(with_acc)
        no_accs.append(no_acc)
        with_states.append(with_state)
        no_states.append(no_state)

        stats_arrays["with_kllucb"]["init_train"].append(safe_float(with_r.get("initial_training_agreement", 0.0)))
        stats_arrays["with_kllucb"]["init_val"].append(safe_float(with_r.get("initial_validation_agreement", 0.0)))
        stats_arrays["with_kllucb"]["final_train"].append(safe_float(with_r.get("final_training_agreement", 0.0)))
        stats_arrays["with_kllucb"]["final_val"].append(safe_float(with_r.get("final_validation_agreement", 0.0)))
        stats_arrays["with_kllucb"]["beam_time"].append(safe_float(with_r.get("beam_time", 0.0)))

        stats_arrays["no_kllucb"]["init_train"].append(safe_float(no_r.get("initial_training_agreement", 0.0)))
        stats_arrays["no_kllucb"]["init_val"].append(safe_float(no_r.get("initial_validation_agreement", 0.0)))
        stats_arrays["no_kllucb"]["final_train"].append(safe_float(no_r.get("final_training_agreement", 0.0)))
        stats_arrays["no_kllucb"]["final_val"].append(safe_float(no_r.get("final_validation_agreement", 0.0)))
        stats_arrays["no_kllucb"]["beam_time"].append(safe_float(no_r.get("beam_time", 0.0)))

        if with_acc >= threshold:
            with_success_count += 1
        if no_acc >= threshold:
            no_success_count += 1

        # Win = higher hold-out agreement. If tied, smaller DFA wins.
        if with_acc > no_acc + 1e-12:
            with_win_count += 1
        elif no_acc > with_acc + 1e-12:
            no_win_count += 1
        else:
            if with_state < no_state:
                with_win_count += 1
            elif no_state < with_state:
                no_win_count += 1
            else:
                tie_count += 1

        if with_r.get("behavior_signature") is not None:
            with_sigs.append(with_r["behavior_signature"])
        if no_r.get("behavior_signature") is not None:
            no_sigs.append(no_r["behavior_signature"])

    def mean_std(values: Sequence[float]) -> tuple[float, float]:
        return float(np.mean(values)), float(np.std(values))

    def build_method_summary(method: str, accs: list, states: list, success_count: int, sigs: list) -> dict:
        init_train_mean, init_train_std = mean_std(stats_arrays[method]["init_train"])
        init_val_mean, init_val_std = mean_std(stats_arrays[method]["init_val"])
        final_train_mean, final_train_std = mean_std(stats_arrays[method]["final_train"])
        final_val_mean, final_val_std = mean_std(stats_arrays[method]["final_val"])
        beam_time_mean, beam_time_std = mean_std(stats_arrays[method]["beam_time"])
        acc_mean, acc_std = mean_std(accs)
        state_mean, state_std = mean_std(states)

        return {
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "states_mean": state_mean,
            "states_std": state_std,
            "success_ratio": success_count / len(seed_results),
            "num_unique_signatures": len(set(sigs)) if sigs else None,
            "init_train_acc_mean": init_train_mean,
            "init_train_acc_std": init_train_std,
            "init_val_acc_mean": init_val_mean,
            "init_val_acc_std": init_val_std,
            "final_train_acc_mean": final_train_mean,
            "final_train_acc_std": final_train_std,
            "final_val_acc_mean": final_val_mean,
            "final_val_acc_std": final_val_std,
            "beam_time_mean": beam_time_mean,
            "beam_time_std": beam_time_std,
        }

    return {
        "language": lang_code,
        "batch_size": int(cfg["batch_size"]),
        "num_seeds": len(seed_results),
        "agreement_threshold": threshold,
        "holdout_size": holdout_size,
        "with_kllucb": build_method_summary("with_kllucb", with_accs, with_states, with_success_count, with_sigs),
        "no_kllucb": build_method_summary("no_kllucb", no_accs, no_states, no_success_count, no_sigs),
        "with_win_rate": with_win_count / len(seed_results),
        "no_win_rate": no_win_count / len(seed_results),
        "tie_rate": tie_count / len(seed_results),
        "all_seeds_results": seed_results,
    }


def run_one_language_multiseed(
    lang_code: str,
    cfg: dict,
    output_root: str,
    num_seeds: int,
    holdout_size: int,
    lang_log_path: Optional[str] = None,
) -> Optional[dict]:
    """Run all seeds for one dataset/automaton."""
    original_stdout = sys.stdout
    lang_tee = None

    try:
        if lang_log_path:
            lang_tee = Tee(lang_log_path)
            sys.stdout = lang_tee

        print(f"\n{'=' * 80}")
        print(f"  DATASET/AUTOMATA: {lang_code}")
        print(f"  Running {num_seeds} seeds")
        print(f"{'=' * 80}")

        seed_results: list[dict] = []
        for seed in range(num_seeds):
            try:
                result = run_one_language_seed(
                    lang_code=lang_code,
                    cfg=cfg,
                    output_root=output_root,
                    seed=seed,
                    holdout_size=holdout_size,
                )
                if result is not None:
                    seed_results.append(result)
                else:
                    print(f"  [SEED {seed}] FAILED (returned None)")
            except Exception as exc:
                print(f"  [SEED {seed}] FAILED with exception: {exc}")
                traceback.print_exc()

        if not seed_results:
            print(f"  [ERROR] All seeds failed for {lang_code}")
            return None

        summary = summarize_language_results(lang_code, cfg, seed_results, holdout_size)
        print(f"\n  [SUMMARY] {lang_code}: {summary['num_seeds']} seeds completed")
        return summary

    finally:
        if lang_tee is not None:
            sys.stdout = original_stdout
            lang_tee.close()
            print(f"  [LOG CLOSED] Language log saved: {lang_log_path}")


# ======================================================================
# Reporting helpers
# ======================================================================

def print_batch_summary_report(all_results: Dict[str, Optional[dict]], agreement_threshold: float) -> None:
    print(f"\n\n{'=' * 180}")
    print(f"  KL-LUCB SUMMARY (agreement_threshold={agreement_threshold})")
    print(f"{'=' * 180}\n")
    print(
        f"{'Dataset':<24} {'Method':<14} "
        f"{'Init Train':<14} {'Init Val':<14} {'Final Train':<14} {'Final Val':<14} "
        f"{'Holdout':<14} {'States':<14} {'Beam Time':<14} {'Target':<10} {'W/T/L':<14}"
    )
    print("─" * 180)

    for lang_code, stats in sorted(all_results.items()):
        if stats is None:
            print(f"{lang_code:<24} [NO DATA]")
            continue

        with_s = stats["with_kllucb"]
        no_s = stats["no_kllucb"]
        win_tie_loss = f"{stats['with_win_rate']:.1%}/{stats['tie_rate']:.1%}/{stats['no_win_rate']:.1%}"

        rows = [
            ("WITH KL-LUCB", with_s, win_tie_loss),
            ("NO KL-LUCB", no_s, ""),
        ]
        for method_name, method_stats, wtl in rows:
            print(
                f"{lang_code:<24} {method_name:<14} "
                f"{method_stats['init_train_acc_mean']:.3f}±{method_stats['init_train_acc_std']:.3f} "
                f"{method_stats['init_val_acc_mean']:.3f}±{method_stats['init_val_acc_std']:.3f} "
                f"{method_stats['final_train_acc_mean']:.3f}±{method_stats['final_train_acc_std']:.3f} "
                f"{method_stats['final_val_acc_mean']:.3f}±{method_stats['final_val_acc_std']:.3f} "
                f"{method_stats['acc_mean']:.3f}±{method_stats['acc_std']:.3f} "
                f"{method_stats['states_mean']:.1f}±{method_stats['states_std']:.1f} "
                f"{method_stats['beam_time_mean']:.1f}±{method_stats['beam_time_std']:.1f} "
                f"{method_stats['success_ratio']:.1%}     {wtl:<14}"
            )
        print("─" * 180)


def save_summary_csv(all_results: Dict[str, Optional[dict]], path: str) -> None:
    rows = []
    for lang_code, stats in sorted(all_results.items()):
        if stats is None:
            continue
        for method_key, method_name in [("with_kllucb", "WITH KL-LUCB"), ("no_kllucb", "NO KL-LUCB")]:
            s = stats[method_key]
            rows.append(
                {
                    "dataset": lang_code,
                    "method": method_name,
                    "batch_size": stats["batch_size"],
                    "num_seeds": stats["num_seeds"],
                    "agreement_threshold": stats["agreement_threshold"],
                    "holdout_size": stats["holdout_size"],
                    "holdout_agreement_mean": s["acc_mean"],
                    "holdout_agreement_std": s["acc_std"],
                    "states_mean": s["states_mean"],
                    "states_std": s["states_std"],
                    "success_ratio": s["success_ratio"],
                    "init_train_agreement_mean": s["init_train_acc_mean"],
                    "init_train_agreement_std": s["init_train_acc_std"],
                    "init_val_agreement_mean": s["init_val_acc_mean"],
                    "init_val_agreement_std": s["init_val_acc_std"],
                    "final_train_agreement_mean": s["final_train_acc_mean"],
                    "final_train_agreement_std": s["final_train_acc_std"],
                    "final_val_agreement_mean": s["final_val_acc_mean"],
                    "final_val_agreement_std": s["final_val_acc_std"],
                    "beam_time_mean": s["beam_time_mean"],
                    "beam_time_std": s["beam_time_std"],
                    "with_win_rate": stats["with_win_rate"],
                    "tie_rate": stats["tie_rate"],
                    "no_win_rate": stats["no_win_rate"],
                    "num_unique_signatures": s["num_unique_signatures"],
                }
            )

    if not rows:
        print("  [CSV] No rows to write.")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary CSV saved → {path}")


def save_seed_details_csv(all_results: Dict[str, Optional[dict]], path: str) -> None:
    rows = []
    for lang_code, stats in sorted(all_results.items()):
        if stats is None:
            continue
        for seed_res in stats.get("all_seeds_results", []):
            for method_key, method_name in [("with_kllucb", "WITH KL-LUCB"), ("no_kllucb", "NO KL-LUCB")]:
                r = seed_res[method_key]
                rows.append(
                    {
                        "dataset": lang_code,
                        "seed": seed_res["seed"],
                        "teacher_type": seed_res.get("teacher_type"),
                        "teacher_states": seed_res.get("teacher_states"),
                        "teacher_train_agreement": seed_res.get("teacher_train_agreement"),
                        "teacher_test_agreement": seed_res.get("teacher_test_agreement"),
                        "method": method_name,
                        "initial_state": r.get("initial_state"),
                        "initial_training_agreement": r.get("initial_training_agreement"),
                        "initial_validation_agreement": r.get("initial_validation_agreement"),
                        "final_state": r.get("final_state"),
                        "final_training_agreement": r.get("final_training_agreement"),
                        "final_validation_agreement": r.get("final_validation_agreement"),
                        "external_holdout_agreement": r.get("external_holdout_agreement"),
                        "time_total": r.get("time_total"),
                        "init_time": r.get("init_time"),
                        "beam_time": r.get("beam_time"),
                        "success": int(bool(r.get("success"))),
                        "budget_used": r.get("budget_used"),
                        "final_dfa_exposed": int(bool(r.get("final_dfa_exposed"))),
                        "used_fallback_validation_agreement": int(bool(r.get("used_fallback_validation_agreement"))),
                        "behavior_signature": r.get("behavior_signature"),
                    }
                )

    if not rows:
        print("  [CSV] No seed detail rows to write.")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Seed detail CSV saved → {path}")


# ======================================================================
# Entry point
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare beam search WITH and WITHOUT KL-LUCB")
    parser.add_argument(
        "--languages",
        type=str,
        default="mnist,ECG,wafer,SecureHandshake,DocumentReleaseWorkflow,MultiObligationOrder",
        help="Comma-separated list of datasets/automata to run",
    )
    parser.add_argument("--agreement_threshold", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--init_num_samples", type=int, default=None)
    parser.add_argument("--edit_distance", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--holdout_size", type=int, default=10000)
    parser.add_argument("--parallel", dest="parallel", action="store_true", default=None, help="Enable parallel KL-LUCB sampling/agreement evaluation.")
    parser.add_argument("--no_parallel", dest="parallel", action="store_false", help="Disable parallel KL-LUCB sampling/agreement evaluation (use for bit-for-bit reproducible runs).")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of worker threads for KL-LUCB sampling/agreement evaluation.")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "Appended to the auto-derived output folder name "
            "(test_result/kllucb_{threshold}_{batch_size}{output_suffix}). "
            "Use this to avoid colliding with an existing run when overriding "
            "--agreement_threshold to a value a task doesn't natively use "
            "(e.g. running real-world tasks at threshold=0.9 alongside their "
            "native 0.8 run)."
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
        "parallel": args.parallel,
        "n_jobs": args.n_jobs,
    }

    cfg_dict = get_languages_config(overrides=overrides)
    languages_to_run = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    unknown_names = [lang for lang in languages_to_run if lang not in cfg_dict]
    if unknown_names:
        raise ValueError(
            f"Unknown language/automata name(s): {unknown_names}\n"
            f"Available options: {list(cfg_dict.keys())}"
        )

    selected_cfg_dict = {lang: cfg_dict[lang] for lang in languages_to_run}
    first_cfg = next(iter(selected_cfg_dict.values()))
    agreement_threshold = first_cfg["agreement_threshold"]
    batch_size = first_cfg["batch_size"]
    threshold_tag = f"{agreement_threshold:g}"

    output_root = os.path.join(
        PROJECT_ROOT, "test_result", f"kllucb_{threshold_tag}_{batch_size}{args.output_suffix}"
    )
    if os.path.exists(output_root):
        raise FileExistsError(
            f"Output root already exists: {output_root}\n"
            "Refusing to run into an existing results folder (would overwrite "
            "prior results). Pass --output_suffix to pick a different folder, "
            "or remove/move the existing one first if you intend to replace it."
        )
    os.makedirs(output_root, exist_ok=True)

    log_path = os.path.join(output_root, "comparison.log")
    summary_csv_path = os.path.join(output_root, "summary.csv")
    seed_csv_path = os.path.join(output_root, "seed_details.csv")

    print(f"\n{'=' * 80}")
    print("  KL-LUCB Comparison Experiment")
    print(f"  Selected experiments: {', '.join(selected_cfg_dict.keys())}")
    print(f"  Num seeds: {args.num_seeds}")
    print(f"  Holdout size: {args.holdout_size}")
    print(f"  Output root: {output_root}")
    print(f"  Logging to: {log_path}")
    print(f"{'=' * 80}\n")

    original_stdout = sys.stdout
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        all_results: Dict[str, Optional[dict]] = {}
        lang_log_paths: Dict[str, str] = {}

        for lang_code, cfg in selected_cfg_dict.items():
            lang_log_file = os.path.join(output_root, f"{lang_code}_seed_log.txt")
            lang_log_paths[lang_code] = lang_log_file

            # Temporarily restore main tee after each language-specific run.
            sys.stdout = original_stdout
            print(
                f"\n[STARTING] {lang_code} "
                f"(batch_size={cfg['batch_size']}, num_seeds={args.num_seeds})"
            )
            print(f"  Language log file: {lang_log_file}")
            sys.stdout = tee

            result = run_one_language_multiseed(
                lang_code=lang_code,
                cfg=cfg,
                output_root=output_root,
                num_seeds=args.num_seeds,
                holdout_size=args.holdout_size,
                lang_log_path=lang_log_file,
            )
            all_results[lang_code] = result

            if result is not None:
                print(f"[COMPLETED] {lang_code}: {result['num_seeds']} seeds")
            else:
                print(f"[FAILED] {lang_code}")

        print_batch_summary_report(all_results, agreement_threshold)
        save_summary_csv(all_results, summary_csv_path)
        save_seed_details_csv(all_results, seed_csv_path)

        print("\n" + "=" * 100)
        print("LANGUAGE-SPECIFIC LOG FILES")
        print("=" * 100)
        for lang_key, log_file in sorted(lang_log_paths.items()):
            file_exists = "✓" if os.path.exists(log_file) else "✗"
            file_size = os.path.getsize(log_file) if os.path.exists(log_file) else 0
            print(f"  {file_exists} {lang_key:<25} -> {log_file} ({file_size} bytes)")

        print("\n" + "=" * 100)
        print(f"Main log saved to: {log_path}")
        print(f"Summary CSV: {summary_csv_path}")
        print(f"Seed details CSV: {seed_csv_path}")
        print(f"Output root: {output_root}")
        print("=" * 100)

    except Exception as exc:
        print(f"\n[FATAL ERROR] Unexpected exception: {exc}")
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        tee.close()
        print("[SCRIPT COMPLETED]")


if __name__ == "__main__":
    main()
