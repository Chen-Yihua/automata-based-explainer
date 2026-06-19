"""
Beam Search Comparison: WITH KL-LUCB vs WITHOUT KL-LUCB
=======================================================
Compares beam search WITH and WITHOUT KL-LUCB using multiple random seeds.
Includes external hold-out evaluation for the final DFA.

Usage (from project root):
    python examples/RPNI/run_kllucb_comparison.py --accuracy_threshold 0.85
    python examples/RPNI/run_kllucb_comparison.py --accuracy_threshold 0.85 --batch_size 1000 --num_seeds 5

Results will be saved to:
    test_result/kllucb_{threshold}_{batch_size}_{date}/
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import pickle
import random
import sys
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score

# ── Path setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, "external_modules")
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, "modified_modules")
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, "Explaining-FA")
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, "interpretera", "src")

for _p in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
from models.sequence_classifier import SequenceClassifier
from tee import Tee
from load_dfa import load_dfa_from_dot, get_automata_alphabet, create_automata_dfa_predictor


# ======================================================================
# Experiment configurations
# ======================================================================

DEFAULT_LANGUAGE_CONFIGS = {
    # ────── Real-World Datasets ──────
    "mnist": dict(
        alphabet        = ['R', 'U', 'L', 'D'],
        accuracy_threshold = 0.8,
        state_threshold = 5,
        delta           = 0.01,
        tau             = 0.05,
        batch_size      = 500,
        beam_size       = 1,
        init_num_samples= 500,
        edit_distance   = 3,
        num_test_instances = 10,
        test_instance = None, # ['R', 'R', 'R', 'R', 'D', 'D', 'L', 'D', 'D', 'L', 'D', 'D', 'D']
        test_instances = None,
        max_length      = 20, embedding_dim = 64, hidden_dim = 256,
        num_layers      = 2,  dropout = 0.3,
    ),
    "ECG": dict(
        alphabet        = ['VL', 'L', 'SL', 'M', 'SH', 'H', 'VH'],
        accuracy_threshold = 0.8,
        state_threshold = 5,
        delta           = 0.01,
        tau             = 0.05,
        batch_size      = 500,
        beam_size       = 1,
        init_num_samples= 500,
        edit_distance   = 2,
        num_test_instances = 10,
        test_instance = None, # ['VL', 'M', 'M', 'M', 'H', 'H', 'SH', 'SL', 'SH', 'VL', 'SL']
        test_instances = None,
        max_length      = 20, embedding_dim = 64, hidden_dim = 256,
        num_layers      = 2,  dropout = 0.3,
    ),
    "wafer": dict(
        alphabet        = ['VL', 'L', 'SL', 'M', 'SH', 'H', 'VH'],
        accuracy_threshold = 0.8,
        state_threshold = 5,
        delta           = 0.01,
        tau             = 0.05,
        batch_size      = 500,
        beam_size       = 1,
        init_num_samples= 500,
        edit_distance   = 3,
        num_test_instances = 10,
        test_instance = None, # ['VL', 'VH', 'VH', 'SL', 'M', 'SH', 'SH', 'SH', 'SH', 'SH', 'SH', 'SH', 'SL', 'L', 'L', 'L', 'L']
        test_instances = None,
        max_length      = 20, embedding_dim = 64, hidden_dim = 256,
        num_layers      = 2,  dropout = 0.3,
    ),
    # ────── Regular ──────
    "SecureHandshake": dict(
        automata_name   = "SecureHandshake",
        filename        = "secure_handshake.dot",
        accuracy_threshold = 0.9,
        delta           = 0.01,
        tau             = 0.1,
        batch_size      = 1000,
        beam_size       = 1,
        init_num_samples= 1000,
        edit_distance   = 5,
        num_test_instances = 10,
        test_instance      = None, # ['hello', 'cert', 'verify', 'verify', 'cert', 'verify', 'key', 'ack', 'ack']
        test_instances     = None,
    ),
    "DocumentReleaseWorkflow": dict(
        automata_name   = "DocumentReleaseWorkflow",
        filename        = "document_release_workflow.dot",
        accuracy_threshold = 0.9,
        delta           = 0.01,
        tau             = 0.05,
        batch_size      = 1000,
        beam_size       = 1,
        init_num_samples= 1000,
        edit_distance   = 5,
        num_test_instances = 10,
        test_instance      = None, # ['draft', 'review', 'review', 'review', 'approve', 'comment', 'comment', 'approve', 'comment', 'publish']
        test_instances     = None,
    ),
    "MultiObligationOrder": dict(
        automata_name   = "MultiObligationOrder",
        filename        = "multi_obligation_color_order.dot",
        accuracy_threshold = 0.9,
        delta           = 0.01,
        tau             = 0.05,
        batch_size      = 1000,
        beam_size       = 1,
        init_num_samples= 1000,
        edit_distance   = 5,
        num_test_instances = 10,
        test_instance      = None, # ['pick', 'blue', 'green', 'move', 'move', 'drop', 'dock', 'yellow']
        test_instances     = None,
    ),
}

def get_languages_config(overrides=None):
    """
    Return experiment configurations.

    Each dataset has its own default setting. Command-line arguments can
    temporarily override these defaults without modifying this file.
    """
    import copy
    configs = copy.deepcopy(DEFAULT_LANGUAGE_CONFIGS)

    if overrides:
        for cfg in configs.values():
            for key, value in overrides.items():
                if value is not None:
                    cfg[key] = value

    return configs


# ======================================================================
# Helper functions
# ======================================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_scalar(value: Any) -> Any:
    """Convert values like [x], np.array([x]), np.scalar to Python scalar."""
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


def maybe_len_states(automaton: Any) -> int:
    if automaton is None:
        return 0
    if hasattr(automaton, "size"):
        return int(automaton.size)
    if hasattr(automaton, "states"):
        return int(len(automaton.states))
    return 0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = to_scalar(value)
        if scalar is None:
            return default
        return float(scalar)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        scalar = to_scalar(value)
        if scalar is None:
            return default
        return int(scalar)
    except Exception:
        return default


def behavior_signature_from_preds(preds: Sequence[int]) -> str:
    """Compact behavior signature for comparing DFA behavior on the same hold-out set."""
    arr = np.asarray(preds, dtype=np.uint8)
    return hashlib.md5(arr.tobytes()).hexdigest()


PrebuiltInit = namedtuple("PrebuiltInit", ["learner", "initial_dfa", "validation_data", "validation_labels"])


# ======================================================================
# DFA evaluation helpers
# ======================================================================
def dfa_accepts_sequence(dfa: Any, sequence: Sequence[Any]) -> bool:
    """Evaluate an AALpy-like DFA on one sequence."""
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
    preds = [1 if dfa_accepts_sequence(dfa, seq) else 0 for seq in sequences]
    return np.asarray(preds, dtype=int)


def extract_automata_pair(exp_obj: Any) -> Optional[Tuple[Any, Any]]:
    """
    Try multiple places to recover [initial_dfa, final_dfa] from explanation output.

    Works best if AnchorTabular.explain() exposes one of:
      - exp_obj.raw_result['automata']
      - exp_obj.result['automata']
      - exp_obj.data['automata']
      - exp_obj.automata_pair
      - exp_obj.automata
    """
    candidate_containers: List[Any] = []

    for attr in ["raw_result", "result", "data"]:
        value = getattr(exp_obj, attr, None)
        if isinstance(value, dict):
            candidate_containers.append(value)

    for container in candidate_containers:
        automata = container.get("automata")
        if isinstance(automata, (list, tuple)) and len(automata) == 2:
            return automata[0], automata[1]

    for attr in ["automata_pair", "automata"]:
        value = getattr(exp_obj, attr, None)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return value[0], value[1]

    return None


def evaluate_final_dfa(
    exp_obj: Any,
    holdout_X: Sequence[Sequence[Any]],
    holdout_y: np.ndarray,
    fallback_validation_accuracy: float,
) -> Dict[str, Any]:
    """
    Evaluate the final DFA on an external hold-out set.
    If the final DFA object is unavailable, fall back to final_validation_accuracy.
    """
    pair = extract_automata_pair(exp_obj)
    if pair is None:
        return {
            "external_holdout_accuracy": fallback_validation_accuracy,
            "behavior_signature": None,
            "used_fallback_validation_accuracy": True,
            "final_dfa_exposed": False,
        }

    _, final_dfa = pair
    preds = dfa_predict_batch(final_dfa, holdout_X)
    acc = float(np.mean(preds == holdout_y)) if len(holdout_y) > 0 else 0.0
    signature = behavior_signature_from_preds(preds)
    return {
        "external_holdout_accuracy": acc,
        "behavior_signature": signature,
        "used_fallback_validation_accuracy": False,
        "final_dfa_exposed": True,
    }

# ======================================================================
# Experiment
# ======================================================================
def run_one_language_seed(
    lang_code: str,
    cfg: dict,
    output_root: str,
    seed: int,
    holdout_size: int,
) -> Optional[dict]:
    set_all_seeds(seed)

    print(f"\n{'─' * 80}")
    print(f"  SEED={seed}")
    print(f"{'─' * 80}")

    out_dir = os.path.join(output_root, lang_code, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    data_type = cfg.get("type", "realworld")
    teacher_dfa = None

    if data_type == "regular":
        try:
            teacher_dfa = load_dfa_from_dot(cfg["filename"])
            if teacher_dfa.initial_state is None or not hasattr(teacher_dfa.initial_state, "state_id"):
                teacher_dfa.initial_state = next(iter(teacher_dfa.states), None)

            alphabet = get_automata_alphabet(teacher_dfa)
            predict_fn = create_automata_dfa_predictor(teacher_dfa)

            print(f"  Automata DFA: {len(teacher_dfa.states)} states, alphabet={alphabet}")

            X_train = []
            for _ in range(100):
                seq_len = random.randint(2, 8)
                seq = [random.choice(alphabet) for _ in range(seq_len)]
                X_train.append(seq)
            X_train = np.array(X_train, dtype=object)
            y_train = predict_fn(X_train)
            clf_train_acc = np.mean(y_train)

            X_test = []
            for _ in range(50):
                seq_len = random.randint(2, 8)
                seq = [random.choice(alphabet) for _ in range(seq_len)]
                X_test.append(seq)
            X_test = np.array(X_test, dtype=object)
            y_test = predict_fn(X_test)
            clf_test_acc = np.mean(y_test)

            print(f"  Teacher DFA: {len(X_train)} training sequences, {clf_train_acc:.1%} acceptance rate")
            print(f"  Teacher DFA test acceptance rate: {clf_test_acc:.1%}")

            test_instance = cfg.get("test_instance")
        except Exception as e:
            print(f"  [ERROR] Failed to load automata DFA: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
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
        print(f"  Classifier  train={clf_train_acc:.4f}  test={clf_test_acc:.4f}")

        test_instance = cfg.get("test_instance")
        alphabet = cfg["alphabet"]
        if test_instance is None:
            if lang_code == "mnist":
                test_instance = X_train[23]
            elif lang_code in ["ECG", "wafer"]:
                test_instance = X_train[0]
            else:
                positive_indices = [i for i, label in enumerate(y_train) if label == 1]
                test_instance = X_train[positive_indices[2]] if positive_indices else X_train[0]

    explainer = AnchorTabular(
        predictor=predict_fn,
        feature_names=[f"pos_{i}" for i in range(max(len(s) for s in X_train))],
        categorical_names={},
        seed=seed,
    )

    if data_type == "regular":
        explainer.fit(
            automaton_type="DFA",
            train_data=X_train,
            alphabet=alphabet,
            disc_perc=None,
            data_type="automata",
            initial_dfa=teacher_dfa,
        )
    else:
        explainer.fit(
            automaton_type="DFA",
            train_data=X_train,
            alphabet=alphabet,
            disc_perc=None,
            data_type="real_world",
        )

    # Keep original behavior if this attribute exists.
    if hasattr(explainer.samplers[0], "d_train_data"):
        explainer.samplers[0].d_train_data = X_train

    for sampler in explainer.samplers:
        sampler.set_instance_label(test_instance)
        sampler.set_n_covered(10)
        sampler.edit_distance = cfg["edit_distance"]

    # External hold-out used only for post-hoc evaluation of the final DFA.
    holdout_sampler = explainer.samplers[0]
    holdout_X, holdout_y = holdout_sampler(num_samples=holdout_size, compute_labels=True)
    holdout_X = list(holdout_X)
    holdout_y = np.asarray(holdout_y, dtype=int)

    results: Dict[str, Any] = {
        "seed": seed,
        "batch_size": int(cfg["batch_size"]),
        "holdout_size": holdout_size,
    }

    # ─────────────────────────────────────────────────────────────────
    # METHOD 1: Beam Search WITH KL-LUCB
    # ─────────────────────────────────────────────────────────────────
    print("\n  Beam Search WITH KL-LUCB")
    t0 = time.time()
    try:
        beam_with_kllucb = explainer.explain(
            type='Tabular',
            automaton_type='DFA',
            alphabet=cfg['alphabet'],
            X=test_instance,
            edit_distance=cfg['edit_distance'],
            accuracy_threshold=cfg['accuracy_threshold'],
            delta=cfg['delta'],
            tau=cfg['tau'],
            beam_size=cfg['beam_size'],
            batch_size=cfg['batch_size'],
            init_num_samples=cfg['init_num_samples'],
            verbose=False,
            output_dir=os.path.join(out_dir, 'beam_with_kllucb'),
            use_kllucb=True,
            prebuilt_init=None,
        )
    except Exception as e:
        print(f"  [ERROR] Beam Search WITH KL-LUCB failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    time_with_kllucb = time.time() - t0
    init_time_with = safe_float(getattr(beam_with_kllucb, 'init_automaton_time', 0.0), 0.0)
    beam_time_with = max(0.0, time_with_kllucb - init_time_with)
    initial_state = safe_int(getattr(beam_with_kllucb, 'initial_state', 0), 0)
    initial_train = safe_float(getattr(beam_with_kllucb, 'initial_training_accuracy', 0.0), 0.0)
    initial_validation = safe_float(getattr(beam_with_kllucb, 'initial_validation_accuracy', 0.0), 0.0)
    final_train = safe_float(getattr(beam_with_kllucb, 'final_training_accuracy', 0.0), 0.0)
    final_validation = safe_float(getattr(beam_with_kllucb, 'final_validation_accuracy', 0.0), 0.0)
    final_state = safe_int(getattr(beam_with_kllucb, 'final_state', 0), 0)
    budget_used_with = getattr(beam_with_kllucb, 'budget_used', None)
    success_with = bool(getattr(beam_with_kllucb, 'success', False))

    mab = getattr(explainer, 'mab', None)
    automatas = getattr(mab, 'automatas', None) or []
    if not automatas:
        print("  [ERROR] Could not extract initial DFA from explainer.mab.automatas")
        return None

    initial_dfa = automatas[0].copy()
    validation_data = list(getattr(mab, 'validation_data', []))
    validation_labels = np.asarray(getattr(mab, 'validation_labels', []))

    print("  Extracting shared initialization...")
    print(f"  Extracted {len(validation_data)} initial samples, DFA with {maybe_len_states(initial_dfa)} states")

    from learner.dfa_learner import DFALearner
    prebuilt = PrebuiltInit(
        learner=DFALearner(),
        initial_dfa=initial_dfa,
        validation_data=validation_data,
        validation_labels=validation_labels,
    )

    eval_with = evaluate_final_dfa(
        beam_with_kllucb,
        holdout_X,
        holdout_y,
        fallback_validation_accuracy=final_validation,
    )

    results['with_kllucb'] = {
        'initial_state': initial_state,
        'initial_training_accuracy': initial_train,
        'initial_validation_accuracy': initial_validation,
        'final_state': final_state,
        'final_training_accuracy': final_train,
        'final_validation_accuracy': final_validation,
        'external_holdout_accuracy': eval_with['external_holdout_accuracy'],
        'behavior_signature': eval_with['behavior_signature'],
        'used_fallback_validation_accuracy': eval_with['used_fallback_validation_accuracy'],
        'final_dfa_exposed': eval_with['final_dfa_exposed'],
        'time_total': time_with_kllucb,
        'init_time': init_time_with,
        'beam_time': beam_time_with,
        'success': success_with,
        'budget_used': budget_used_with,
        'shared_init_used': True,
    }

    print(
        f"    WITH KL-LUCB:  train_acc={final_train:.4f}, "
        f"val_acc={final_validation:.4f}, holdout_acc={eval_with['external_holdout_accuracy']:.4f}, "
        f"states={final_state}, beam_time={beam_time_with:.1f}s"
    )

    # ─────────────────────────────────────────────────────────────────
    # METHOD 2: Beam Search WITHOUT KL-LUCB (same initial automaton)
    # ─────────────────────────────────────────────────────────────────
    print("\n  Beam Search WITHOUT KL-LUCB (using shared initial automaton)...")
    for sampler in explainer.samplers:
        sampler.set_instance_label(test_instance)
        sampler.set_n_covered(10)
        sampler.edit_distance = cfg['edit_distance']

    t0 = time.time()
    try:
        beam_no_kllucb = explainer.explain(
            type='Tabular',
            automaton_type='DFA',
            alphabet=cfg['alphabet'],
            X=test_instance,
            edit_distance=cfg['edit_distance'],
            accuracy_threshold=cfg['accuracy_threshold'],
            delta=cfg['delta'],
            tau=cfg['tau'],
            beam_size=cfg['beam_size'],
            batch_size=cfg['batch_size'],
            init_num_samples=cfg['init_num_samples'],
            verbose=False,
            output_dir=os.path.join(out_dir, 'beam_no_kllucb'),
            use_kllucb=False,
            prebuilt_init=prebuilt,
        )
    except Exception as e:
        print(f"  [ERROR] Beam Search WITHOUT KL-LUCB failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    time_no_kllucb = time.time() - t0
    init_time_no = safe_float(getattr(beam_no_kllucb, 'init_automaton_time', 0.0), 0.0)
    beam_time_no = max(0.0, time_no_kllucb - init_time_no)
    initial_state_no = safe_int(getattr(beam_no_kllucb, 'initial_state', 0), 0)
    initial_train_no = safe_float(getattr(beam_no_kllucb, 'initial_training_accuracy', 0.0), 0.0)
    initial_validation_no = safe_float(getattr(beam_no_kllucb, 'initial_validation_accuracy', 0.0), 0.0)
    final_train_no = safe_float(getattr(beam_no_kllucb, 'final_training_accuracy', 0.0), 0.0)
    final_validation_no = safe_float(getattr(beam_no_kllucb, 'final_validation_accuracy', 0.0), 0.0)
    final_state_no = safe_int(getattr(beam_no_kllucb, 'final_state', 0), 0)
    budget_used_no = getattr(beam_no_kllucb, 'budget_used', None)
    success_no = bool(getattr(beam_no_kllucb, 'success', False))

    eval_no = evaluate_final_dfa(
        beam_no_kllucb,
        holdout_X,
        holdout_y,
        fallback_validation_accuracy=final_validation_no,
    )

    results['no_kllucb'] = {
        'initial_state': initial_state_no,
        'initial_training_accuracy': initial_train_no,
        'initial_validation_accuracy': initial_validation_no,
        'final_state': final_state_no,
        'final_training_accuracy': final_train_no,
        'final_validation_accuracy': final_validation_no,
        'external_holdout_accuracy': eval_no['external_holdout_accuracy'],
        'behavior_signature': eval_no['behavior_signature'],
        'used_fallback_validation_accuracy': eval_no['used_fallback_validation_accuracy'],
        'final_dfa_exposed': eval_no['final_dfa_exposed'],
        'time_total': time_no_kllucb,
        'init_time': init_time_no,
        'beam_time': beam_time_no,
        'success': success_no,
        'budget_used': budget_used_no,
        'shared_init_used': True,
    }

    print(
        f"    NO KL-LUCB:    train_acc={final_train_no:.4f}, "
        f"val_acc={final_validation_no:.4f}, holdout_acc={eval_no['external_holdout_accuracy']:.4f}, "
        f"states={final_state_no}, beam_time={beam_time_no:.1f}s"
    )

    if not eval_with['final_dfa_exposed'] or not eval_no['final_dfa_exposed']:
        print("    [WARN] Final DFA object not exposed; holdout_acc is currently using final_validation_accuracy as fallback.")

    return results


# ======================================================================
# Multi-seed experiment for single language + single batch size
# ======================================================================
def run_one_language_multiset(
    lang_code: str,
    cfg: dict,
    output_root: str,
    num_seeds: int = 10,
    holdout_size: int = 10000,
    lang_log_path: Optional[str] = None,
) -> Optional[dict]:
    # Save the original stdout (which is already redirected by main Tee)
    original_stdout = sys.stdout
    lang_tee = None
    
    try:
        # Create language-specific logger if path provided
        if lang_log_path:
            lang_tee = Tee(lang_log_path)
        
        print(f"\n{'=' * 80}")
        print(f"  LANGUAGE/AUTOMATA: {lang_code}")
        print(f"  Running {num_seeds} seeds...")
        if lang_log_path:
            print(f"  [Language log: {lang_log_path}]")
        print(f"{'=' * 80}")

        all_seeds_results = []
        for seed_idx in range(num_seeds):
            try:
                result = run_one_language_seed(
                    lang_code,
                    cfg,
                    output_root,
                    seed_idx,
                    holdout_size=holdout_size,
                )
                if result is not None:
                    all_seeds_results.append(result)
                else:
                    print(f"  [SEED {seed_idx}] FAILED (returned None)")
            except Exception as e:
                print(f"  [SEED {seed_idx}] FAILED with exception: {e}")
                import traceback
                traceback.print_exc()

        if not all_seeds_results:
            print(f"  [ERROR] All seeds failed for {lang_code}")
            return None

        with_accs, no_accs = [], []
        with_states, no_states = [], []
        with_success_count, no_success_count = 0, 0
        with_win_count, no_win_count, tie_count = 0, 0, 0
        with_sigs, no_sigs = [], []
        with_init_train_accs, no_init_train_accs = [], []
        with_init_val_accs, no_init_val_accs = [], []
        with_final_train_accs, no_final_train_accs = [], []
        with_final_val_accs, no_final_val_accs = [], []
        with_times, no_times = [], []

        for res in all_seeds_results:
            with_r = res['with_kllucb']
            no_r = res['no_kllucb']

            with_acc = safe_float(with_r.get('external_holdout_accuracy', 0.0), 0.0)
            no_acc = safe_float(no_r.get('external_holdout_accuracy', 0.0), 0.0)
            with_state = safe_int(with_r.get('final_state', 0), 0)
            no_state = safe_int(no_r.get('final_state', 0), 0)

            with_accs.append(with_acc)
            no_accs.append(no_acc)
            with_states.append(with_state)
            no_states.append(no_state)

            # 收集初始/最終準確率和時間
            with_init_train_accs.append(safe_float(with_r.get('initial_training_accuracy', 0.0), 0.0))
            no_init_train_accs.append(safe_float(no_r.get('initial_training_accuracy', 0.0), 0.0))
            with_init_val_accs.append(safe_float(with_r.get('initial_validation_accuracy', 0.0), 0.0))
            no_init_val_accs.append(safe_float(no_r.get('initial_validation_accuracy', 0.0), 0.0))
            with_final_train_accs.append(safe_float(with_r.get('final_training_accuracy', 0.0), 0.0))
            no_final_train_accs.append(safe_float(no_r.get('final_training_accuracy', 0.0), 0.0))
            with_final_val_accs.append(safe_float(with_r.get('final_validation_accuracy', 0.0), 0.0))
            no_final_val_accs.append(safe_float(no_r.get('final_validation_accuracy', 0.0), 0.0))
            with_times.append(safe_float(with_r.get('beam_time', 0.0), 0.0))
            no_times.append(safe_float(no_r.get('beam_time', 0.0), 0.0))

            if with_acc >= cfg.get('accuracy_threshold', 0.8):
                with_success_count += 1
            if no_acc >= cfg.get('accuracy_threshold', 0.8):
                no_success_count += 1

            # Win = higher hold-out accuracy. If tied, smaller DFA wins.
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

            if with_r.get('behavior_signature') is not None:
                with_sigs.append(with_r['behavior_signature'])
            if no_r.get('behavior_signature') is not None:
                no_sigs.append(no_r['behavior_signature'])

        summary = {
            'language': lang_code,
            'batch_size': int(cfg['batch_size']),
            'num_seeds': len(all_seeds_results),
            'accuracy_threshold': cfg.get('accuracy_threshold', 0.8),
            'holdout_size': holdout_size,
            'with_kllucb': {
                'acc_mean': float(np.mean(with_accs)),
                'acc_std': float(np.std(with_accs)),
                'states_mean': float(np.mean(with_states)),
                'states_std': float(np.std(with_states)),
                'success_ratio': with_success_count / len(all_seeds_results),
                'num_unique_signatures': len(set(with_sigs)) if with_sigs else None,
                # initial/final accuracy statistics
                'init_train_acc_mean': float(np.mean(with_init_train_accs)),
                'init_train_acc_std': float(np.std(with_init_train_accs)),
                'init_val_acc_mean': float(np.mean(with_init_val_accs)),
                'init_val_acc_std': float(np.std(with_init_val_accs)),
                'final_train_acc_mean': float(np.mean(with_final_train_accs)),
                'final_train_acc_std': float(np.std(with_final_train_accs)),
                'final_val_acc_mean': float(np.mean(with_final_val_accs)),
                'final_val_acc_std': float(np.std(with_final_val_accs)),
                'beam_time_mean': float(np.mean(with_times)),
                'beam_time_std': float(np.std(with_times)),
            },
            'no_kllucb': {
                'acc_mean': float(np.mean(no_accs)),
                'acc_std': float(np.std(no_accs)),
                'states_mean': float(np.mean(no_states)),
                'states_std': float(np.std(no_states)),
                'success_ratio': no_success_count / len(all_seeds_results),
                'num_unique_signatures': len(set(no_sigs)) if no_sigs else None,
                # initial/final accuracy statistics
                'init_train_acc_mean': float(np.mean(no_init_train_accs)),
                'init_train_acc_std': float(np.std(no_init_train_accs)),
                'init_val_acc_mean': float(np.mean(no_init_val_accs)),
                'init_val_acc_std': float(np.std(no_init_val_accs)),
                'final_train_acc_mean': float(np.mean(no_final_train_accs)),
                'final_train_acc_std': float(np.std(no_final_train_accs)),
                'final_val_acc_mean': float(np.mean(no_final_val_accs)),
                'final_val_acc_std': float(np.std(no_final_val_accs)),
                'beam_time_mean': float(np.mean(no_times)),
                'beam_time_std': float(np.std(no_times)),
            },
            'with_win_rate': with_win_count / len(all_seeds_results),
            'no_win_rate': no_win_count / len(all_seeds_results),
            'tie_rate': tie_count / len(all_seeds_results),
            'all_seeds_results': all_seeds_results,
        }
        
        print(f"\n  [SUMMARY] Language={lang_code}, Seeds={len(all_seeds_results)} completed successfully")
        return summary
        
    finally:
        # Restore stdout and close language-specific logger
        if lang_tee is not None:
            lang_tee.close()
            sys.stdout = original_stdout
            print(f"  [LOG CLOSED] Language log saved: {lang_log_path}")


# ======================================================================
# Reporting helpers
# ======================================================================
def print_batch_summary_report(all_results: Dict[str, dict], accuracy_threshold: float) -> None:
    print(f"\n\n{'=' * 280}")
    print(f"  SUMMARY (accuracy_threshold={accuracy_threshold})")
    print(f"{'=' * 280}\n")

    print(
        f"{'Language':<22} {'Method':<14} "
        f"{'Init Train':<16} {'Init Val':<16} {'Final Train':<16} {'Final Val':<16} "
        f"{'Final States':<14} {'Beam Time(s)':<14} {'Reach Target':<12} {'Win/Tie/Loss':<16}"
    )
    print(
        f"{'':22} {'':14} "
        f"{'mean±std':<16} {'mean±std':<16} {'mean±std':<16} {'mean±std':<16} "
        f"{'mean±std':<14} {'mean±std':<14} {'ratio':<12} {'(WITH view)':<16}"
    )
    print("─" * 280)

    for lang_code, stats in sorted(all_results.items()):
        if stats is None:
            continue

        with_s = stats['with_kllucb']
        no_s = stats['no_kllucb']
        win_tie_loss = f"{stats['with_win_rate']:.1%}/{stats['tie_rate']:.1%}/{stats['no_win_rate']:.1%}"

        # Pre-format all values for WITH KL-LUCB
        with_init_train = f"{with_s['init_train_acc_mean']:.3f}±{with_s['init_train_acc_std']:.3f}"
        with_init_val = f"{with_s['init_val_acc_mean']:.3f}±{with_s['init_val_acc_std']:.3f}"
        with_final_train = f"{with_s['final_train_acc_mean']:.3f}±{with_s['final_train_acc_std']:.3f}"
        with_final_val = f"{with_s['final_val_acc_mean']:.3f}±{with_s['final_val_acc_std']:.3f}"
        with_states_str = f"{with_s['states_mean']:.1f}±{with_s['states_std']:.1f}"
        with_time_str = f"{with_s['beam_time_mean']:.1f}±{with_s['beam_time_std']:.1f}"
        with_success_str = f"{with_s['success_ratio']:.1%}"

        # Pre-format all values for NO KL-LUCB
        no_init_train = f"{no_s['init_train_acc_mean']:.3f}±{no_s['init_train_acc_std']:.3f}"
        no_init_val = f"{no_s['init_val_acc_mean']:.3f}±{no_s['init_val_acc_std']:.3f}"
        no_final_train = f"{no_s['final_train_acc_mean']:.3f}±{no_s['final_train_acc_std']:.3f}"
        no_final_val = f"{no_s['final_val_acc_mean']:.3f}±{no_s['final_val_acc_std']:.3f}"
        no_states_str = f"{no_s['states_mean']:.1f}±{no_s['states_std']:.1f}"
        no_time_str = f"{no_s['beam_time_mean']:.1f}±{no_s['beam_time_std']:.1f}"
        no_success_str = f"{no_s['success_ratio']:.1%}"

        print(
            f"{lang_code:<22} {'WITH KL-LUCB':<14} "
            f"{with_init_train:<16} {with_init_val:<16} {with_final_train:<16} {with_final_val:<16} "
            f"{with_states_str:<14} {with_time_str:<14} {with_success_str:<12} {win_tie_loss:<16}"
        )
        print(
            f"{'':<22} {'NO KL-LUCB':<14} "
            f"{no_init_train:<16} {no_init_val:<16} {no_final_train:<16} {no_final_val:<16} "
            f"{no_states_str:<14} {no_time_str:<14} {no_success_str:<12} {'':<16}"
        )
        print("─" * 280)





# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare beam search WITH and WITHOUT KL-LUCB"
    )

    parser.add_argument(
        "--languages",
        type=str,
        default="mnist,ECG,wafer,SecureHandshake,DocumentReleaseWorkflow,MultiObligationOrder",
        help="Comma-separated list of datasets/automata to run",
    )

    # Config overrides. Use None to preserve per-experiment default settings.
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=None,
        help="Override accuracy threshold for selected experiments",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Override delta parameter for KL-LUCB / beam search",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override tau parameter",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=None,
        help="Override beam size",
    )
    parser.add_argument(
        "--init_num_samples",
        type=int,
        default=None,
        help="Override init_num_samples",
    )
    parser.add_argument(
        "--edit_distance",
        type=int,
        default=None,
        help="Override edit distance",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Override maximum sequence length",
    )

    # Runner-level parameters. These are not per-dataset config values.
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="Number of random seeds per language",
    )
    parser.add_argument(
        "--holdout_size",
        type=int,
        default=10000,
        help="External hold-out size used to evaluate the final DFA",
    )

    args = parser.parse_args()

    overrides = {
        "accuracy_threshold": args.accuracy_threshold,
        "delta": args.delta,
        "tau": args.tau,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "init_num_samples": args.init_num_samples,
        "edit_distance": args.edit_distance,
        "max_length": args.max_length,
    }

    cfg_dict = get_languages_config(overrides=overrides)

    languages_to_run = [
        lang.strip()
        for lang in args.languages.split(",")
        if lang.strip()
    ]

    unknown_names = [
        lang for lang in languages_to_run
        if lang not in cfg_dict
    ]

    if unknown_names:
        raise ValueError(
            f"Unknown language/automata name(s): {unknown_names}\n"
            f"Available options: {list(cfg_dict.keys())}"
        )

    selected_cfg_dict = {
        lang: cfg_dict[lang]
        for lang in languages_to_run
    }

    first_cfg = next(iter(selected_cfg_dict.values()))
    accuracy_threshold = first_cfg["accuracy_threshold"]
    batch_size = first_cfg["batch_size"]

    threshold_tag = f"{accuracy_threshold:g}"
    output_root = os.path.join(
        PROJECT_ROOT,
        "test_result",
        f"kllucb_{threshold_tag}_{batch_size}"
    )
    os.makedirs(output_root, exist_ok=True)

    log_path = os.path.join(output_root, "comparison.log")

    print(f"\n{'=' * 80}")
    print("  KL-LUCB Comparison Experiment")
    print(f"  Selected experiments: {', '.join(selected_cfg_dict.keys())}")
    print(f"  Num seeds: {args.num_seeds}")
    print(f"  Holdout size: {args.holdout_size}")
    print(f"  Output root: {output_root}")
    print(f"  Logging to: {log_path}")
    print(f"{'=' * 80}\n")

    tee = Tee(log_path)
    lang_log_paths: Dict[str, str] = {}

    try:
        all_results: Dict[str, dict] = {}

        print(f"\n{'#' * 100}")
        print(f"[BATCH SIZE] {batch_size}")
        print(f"{'#' * 100}")

        for lang_code, cfg in selected_cfg_dict.items():
            lang_log_file = os.path.join(output_root, f"{lang_code}_seed_log.txt")
            lang_log_paths[lang_code] = lang_log_file

            print(
                f"\n[STARTING] Processing {lang_code} "
                f"with batch_size={cfg['batch_size']}, "
                f"num_seeds={args.num_seeds}..."
            )
            print(f"  [Language log file: {lang_log_file}]")
            sys.stdout.flush()

            try:
                result = run_one_language_multiset(
                    lang_code,
                    cfg,
                    output_root,
                    num_seeds=args.num_seeds,
                    holdout_size=args.holdout_size,
                    lang_log_path=lang_log_file,
                )
                all_results[lang_code] = result

                if result is not None:
                    print(
                        f"[COMPLETED] {lang_code} succeeded "
                        f"with {result['num_seeds']} seeds"
                    )
                    print(f"  Results saved to: {lang_log_file}")
                else:
                    print(f"[FAILED] {lang_code} returned None")

            except Exception as e:
                print(f"[ERROR] {lang_code} failed with exception: {e}")
                import traceback
                traceback.print_exc()
                all_results[lang_code] = None

            sys.stdout.flush()

        print_batch_summary_report(all_results, accuracy_threshold)
        sys.stdout.flush()

        print("\n" + "=" * 100)
        print("LANGUAGE-SPECIFIC LOG FILES SUMMARY")
        print("=" * 100)

        for lang_key, log_file in sorted(lang_log_paths.items()):
            file_exists = "✓" if os.path.exists(log_file) else "✗"
            file_size = os.path.getsize(log_file) if os.path.exists(log_file) else 0
            print(f"  {file_exists} {lang_key:<25} -> {log_file} ({file_size} bytes)")

        sys.stdout.flush()

        print("\n" + "=" * 100)
        print(f"Main log saved to: {log_path}")
        print(f"Output root: {output_root}")
        print("=" * 100)
        sys.stdout.flush()

    except Exception as e:
        print(f"\n[FATAL ERROR] Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

    finally:
        tee.close()
        print("[SCRIPT COMPLETED]")
