"""
Real-World Experiment: DFA Search on Real-World Datasets
==========================================================
Runs all four DFA search methods (Beam, SA, GA, PSO) on real-world datasets
(ECG5000, Wafer, MNIST stroke) with configurable accuracy threshold.

Usage (from project root):
    python examples/RPNI/run_realworld_experiment.py --accuracy_threshold 0.8
    python examples/RPNI/run_realworld_experiment.py --accuracy_threshold 0.9 --batch_size 500

Results will be saved to:
    test_result/baseline_experiment/accuracy_threshold_{threshold}/
"""

from __future__ import annotations
import argparse
import csv
import os
import pickle
import random
import sys
import time
import traceback
from typing import Optional
import numpy as np
import torch
from sklearn.metrics import accuracy_score

# ── Path setup (same as run_tomita.py) ─────────────────────────────────
PROJECT_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_PATH         = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, 'modified_modules')
EXPLAINING_FA    = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

for _p in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
from models.sequence_classifier import SequenceClassifier
from search_baselines import sa_dfa_search, ga_dfa_search, pso_dfa_search, SharedInit
from tee import Tee

# ======================================================================
# Experiment configurations
# ======================================================================

DEFAULT_LANGUAGE_CONFIGS = {
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
    # "character_trajectories": dict(
    #     alphabet=None,
    #     accuracy_threshold = 0.8,
    #     delta           = 0.01,
    #     tau             = 0.05,
    #     batch_size      = 500,
    #     beam_size       = 1,
    #     init_num_samples= 2000,
    #     edit_distance   = 3,
    #     num_test_instances = 10,
    #     test_instance = None, # ['R|F_LOW', 'R|F_LOW', 'DL|F_HIGH', 'DR|F_VHIGH', 'UR|F_VHIGH', 'R|F_VHIGH', 'DL|F_VHIGH', 'DR|F_MID', 'UR|F_MID', 'UR|F_HIGH', 'DL|F_MID', 'D|F_MID', 'UR|F_MID', 'UR|F_HIGH', 'DL|F_HIGH', 'R|F_VLOW', 'UR|F_VLOW', 'UR|F_VLOW', 'UR|F_VLOW', 'UR|F_LOW'],
    #     test_instances = None,
    #     max_length      = 20, embedding_dim = 64, hidden_dim = 256,
    #     num_layers      = 2,  dropout = 0.3,
    # ),
    # "adfa": dict(
    #     alphabet=None,
    #     accuracy_threshold = 0.8,
    #     delta           = 0.01,
    #     tau             = 0.05,
    #     batch_size      = 500,
    #     beam_size       = 1,
    #     init_num_samples= 1000,
    #     edit_distance   = 3,
    #     num_test_instances = 10,
    #     test_instance = None,
    #     test_instances = None,
    #     max_length      = 20, embedding_dim = 64, hidden_dim = 256,
    #     num_layers      = 2,  dropout = 0.3,
    # ),
    # "bpi2012": dict(
    #     alphabet=None,
    #     accuracy_threshold = 0.8,
    #     delta           = 0.01,
    #     tau             = 0.05,
    #     batch_size      = 500,
    #     beam_size       = 1,
    #     init_num_samples= 2000,
    #     edit_distance   = 3,
    #     num_test_instances = 10,
    #     test_instance = None,
    #     test_instances = None,
    #     max_length      = 20, embedding_dim = 64, hidden_dim = 256,
    #     num_layers      = 2,  dropout = 0.3,
    #     ),
    # "malware": dict(
    #     alphabet=None,
    #     accuracy_threshold = 0.8,
    #     delta           = 0.01,
    #     tau             = 0.05,
    #     batch_size      = 500,
    #     beam_size       = 1,
    #     init_num_samples= 2000,
    #     edit_distance   = 3,
    #     num_test_instances = 10,
    #     test_instance = None,
    #     test_instances = None,
    #     max_length      = 20, embedding_dim = 64, hidden_dim = 256,
    #     num_layers      = 2,  dropout = 0.3,
    # ),
    # "unix_user": dict(
    #     alphabet=None,
    #     accuracy_threshold = 0.8,
    #     delta           = 0.01,
    #     tau             = 0.05,
    #     batch_size      = 500,
    #     beam_size       = 1,
    #     init_num_samples= 2000,
    #     edit_distance   = 3,
    #     num_test_instances = 10,
    #     test_instance = None,
    #     test_instances = None,
    #     max_length      = 20, embedding_dim = 64, hidden_dim = 256,
    #     num_layers      = 2,  dropout = 0.3,
    # )
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

# Search strategy: Fair comparison with unified budget
# All methods (SA, GA, PSO) use same budget as Beam Search: measure actual candidates generated
SA_STEPS         = 500  # SA steps per start (full cooling schedule)
GA_POPULATION_SIZE = 10  # GA population size
PSO_PARTICLES    = 5   # PSO particle count
PSO_MAX_OPS_PER_ITERATION = 1  # PSO max operations per particle evaluation


# ==================================================================================
# Utility functions for loading/saving beam search results and shared initialization
# ==================================================================================
def _normalize_sequence(seq):
    """
    Convert one sequence to a normal Python list.
    This avoids issues when X_train stores sequences as numpy arrays.
    """
    if isinstance(seq, np.ndarray):
        return seq.tolist()
    return list(seq)

def get_test_instances(X_train, cfg):
    """
    Get test instances for real-world datasets.

    Priority:
    1. If cfg['test_instances'] is provided, use it.
    2. If cfg['test_instance'] is provided, use it as a single-item list.
    3. Otherwise, use the first N training sequences.
    """
    if cfg.get("test_instances") is not None:
        return [_normalize_sequence(seq) for seq in cfg["test_instances"]]

    if cfg.get("test_instance") is not None:
        return [_normalize_sequence(cfg["test_instance"])]

    n = cfg.get("num_test_instances", 10)
    return [_normalize_sequence(seq) for seq in X_train[:n]]

def _extract_accuracy(value) -> float:
    """
    Extract scalar accuracy from various formats returned by search methods.
    
    Handles:
    - Scalar float/int: return as float
    - List (from anchor_base.py): return first element or mean
    - None/zero: return 0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(np.mean(value)) if len(value) > 1 else float(value[0])
    return float(value)


def eval_validation_acc(dfa, val_data, val_labels, learner) -> float:
    """
    DFA accuracy on the validation set (initial DFA training samples).

    val_labels[i] = 1 if model predicts positive class
                    0 if model predicts negative class
    """
    if dfa is None or len(val_data) == 0:
        return 0.0
    accepts = np.array([learner.check_path_accepted(dfa, seq) for seq in val_data])
    lbl     = np.asarray(val_labels)
    correct = int(np.sum((lbl == 1) & accepts) + np.sum((lbl == 0) & ~accepts))
    return correct / len(val_data)


def load_beam_results_from_disk(output_dir: str) -> Optional[dict]:
    """
    Load beam search results from disk (previously saved).
    
    Looks for lang_code/shared/beam_results.pkl
    
    Returns
    -------
    dict with keys: 'initial_state', 'initial_training_accuracy', 'final_state', etc.
    OR None if file does not exist or load failed
    """
    if output_dir is None:
        return None

    shared_dir = os.path.join(output_dir, "shared")
    beam_results_path = os.path.join(shared_dir, "beam_results.pkl")

    if not os.path.exists(beam_results_path):
        print(f"  [BeamResults] No cached beam results found at {beam_results_path}.")
        return None
    try:
        with open(beam_results_path, 'rb') as f:
            beam_res = pickle.load(f)
        
        if not isinstance(beam_res, dict):
            print(f"  [BeamResults] WARNING: Loaded object is not dict type.")
            return None
        
        cached_train = _extract_accuracy(beam_res.get('final_training_accuracy', 0.0))
        cached_val = _extract_accuracy(beam_res.get('final_validation_accuracy', 0.0))
        print(f"  [BeamResults] Loaded from disk: train_acc={cached_train:.4f}, "
              f"validation_acc={cached_val:.4f}, states={beam_res['final_state']}")
        return beam_res
    except Exception as exc:
        print(f"  [BeamResults] Failed to load from disk: {exc}")
        return None

def load_shared_init_from_disk(output_dir: str) -> Optional[dict]:
    """
    Load shared initialization from disk (previously saved by beam search).
    
    Looks for lang_code/shared/shared_init.pkl
    Supports both new format (6 fields) and old format (4 fields).
    Returns a dict with initial_dfa, learner, validation_data, validation_labels.
    Training data will be generated in run_one_language().
    
    Returns
    -------
    dict with keys: 'initial_dfa', 'learner', 'validation_data', 'validation_labels'
    OR None if file does not exist or load failed
    """
    if output_dir is None:
        return None

    shared_dir = os.path.join(output_dir, "shared")
    shared_init_path = os.path.join(shared_dir, "shared_init.pkl")

    if not os.path.exists(shared_init_path):
        print(f"  [SharedInit] No cached shared init found at {shared_init_path}.")
        return None

    try:
        with open(shared_init_path, 'rb') as f:
            obj = pickle.load(f)
        
        # Handle dict format (new format)
        if isinstance(obj, dict) and 'initial_dfa' in obj:
            print(f"  [SharedInit] Loaded: {len(obj['initial_dfa'].states)} states, "
                  f"{len(obj['validation_data'])} validation samples")
            return obj
        
        # Handle raw tuple/list (old format, 4 fields)
        elif isinstance(obj, (tuple, list)) and len(obj) >= 4:
            initial_dfa, learner, validation_data, validation_labels = obj[:4]
            print(f"  [SharedInit] Loaded old format: {len(initial_dfa.states)} states, "
                  f"{len(validation_data)} validation samples")
            return {
                'initial_dfa': initial_dfa,
                'learner': learner,
                'validation_data': validation_data,
                'validation_labels': validation_labels,
            }
        else:
            return None
            
    except TypeError as e:
        # Old SharedInit NamedTuple deserialization error
        # This pickle file contains an old SharedInit object that can't be recreated
        # Return None - beam search will regenerate and overwrite it with new dict format
        print(f"  [SharedInit] Old format (incompatible): {e}")
        print(f"    Will regenerate from beam search on next run.")
        return None
        
    except Exception as exc:
        print(f"  [SharedInit] Failed to load from disk: {exc}")
        return None

def save_beam_results_to_disk(beam_results: dict, output_dir: str = None) -> bool:
    """
    Save beam search results to disk.
    
    Parameters
    ----------
    beam_results : dict with keys 'train_acc', 'validation_acc', 'states', 'time', 'success'
    output_dir   : language directory (lang_code/)
    
    Returns
    -------
    True if saved successfully, False otherwise
    """
    if output_dir is None or beam_results is None:
        return False

    shared_dir = os.path.join(output_dir, "shared")
    try:
        os.makedirs(shared_dir, exist_ok=True)
        beam_results_path = os.path.join(shared_dir, "beam_results.pkl")
        with open(beam_results_path, 'wb') as f:
            pickle.dump(beam_results, f)
        print(f"  [BeamResults] Saved to {beam_results_path}")
        return True
    except Exception as exc:
        print(f"  [WARNING] Could not save beam results: {exc}")
        return False

def build_shared_init_from_beam(explainer, sampler_fn, output_dir: str = None, batch_size: int = 1000, generate_training_data: bool = False) -> Optional[dict]:
    """
    Extract shared initialization data from beam search result and save to disk.
    [Plan A] Returns dict with: initial_dfa, learner, validation_data, validation_labels, training_data, training_labels
    If generate_training_data=True, generates and saves training_data using sampler_fn (shared across all methods)
    
    Parameters
    ----------
    explainer          : AnchorTabular object with mab (multi-armed bandit) set after explain()
    sampler_fn         : Callable – generates perturbation samples
    output_dir         : language directory (lang_code/) to save shared/ subfolder
    batch_size         : int – number of samples to generate for training_data (Plan A)
    generate_training_data : bool – if True, generate and save training_data to be shared across GA/SA/PSO
    
    Returns
    -------
    dict with keys: 'initial_dfa', 'learner', 'validation_data', 'validation_labels', 'training_data', 'training_labels'
    OR None if extraction failed
    """
    mab = getattr(explainer, 'mab', None)
    automatas = getattr(mab, 'automatas', None) or []
    initial_dfa = automatas[0].copy()
    validation_data = list(getattr(mab, 'validation_data'))
    validation_labels = np.asarray(getattr(mab, 'validation_labels'))

    # Get learner instance from mab
    from learner.dfa_learner import DFALearner
    learner = DFALearner()

    # Generate and save training_data to be shared across GA/SA/PSO
    training_data_raw, training_labels = sampler_fn(num_samples=batch_size, compute_labels=True)
    training_data = list(training_data_raw)
    training_labels = np.asarray(training_labels)

    # Create dict
    shared_data = {
        'initial_dfa': initial_dfa,
        'learner': learner,
        'validation_data': validation_data,
        'validation_labels': validation_labels,
        'training_data': training_data,
        'training_labels': training_labels,
    }

    # Save to disk under lang_code/shared/
    if output_dir is not None:
        shared_dir = os.path.join(output_dir, "shared")
        try:
            os.makedirs(shared_dir, exist_ok=True)
            # Save base shared init
            shared_init_path = os.path.join(shared_dir, "shared_init.pkl")
            with open(shared_init_path, 'wb') as f:
                pickle.dump(shared_data, f)
            print(f"  [SharedInit] Saved to {shared_init_path}")
            print(f"    ├─ initial_dfa: {len(initial_dfa.states)} states")
            print(f"    ├─ validation_data: {len(validation_data)} samples")
            print(f"    ├─ validation_labels: {len(validation_labels)} labels")
            if generate_training_data:
                print(f"    ├─ training_data: {len(training_data)} samples [SHARED across GA/SA/PSO]")
                print(f"    └─ training_labels: {len(training_labels)} labels")
        except Exception as exc:
            print(f"  [WARNING] Could not save shared init: {exc}")

    return shared_data


# ======================================================================
# Experiment
# ======================================================================
def run_one_language(lang_code: str, cfg: dict, output_root: str) -> dict | None:
    """
    Run one real-world dataset on multiple automatically selected test instances.

    For real-world datasets, the default test instances are the first
    cfg['num_test_instances'] sequences from X_train.
    """
    print(f"\n{'=' * 70}")
    print(f"  LANGUAGE: {lang_code}")
    print(f"{'=' * 70}")

    # Load model
    model_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_classifier_trained.pth")
    split_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_train_test_split.pkl")

    if not (os.path.exists(model_path) and os.path.exists(split_path)):
        print(f"  [SKIP] Pre-trained model or data split not found for {lang_code}.")
        return None

    with open(split_path, "rb") as f:
        split = pickle.load(f)

    X_train, y_train = split["X_train"], split["y_train"]
    X_test,  y_test  = split["X_test"],  split["y_test"]
    alphabet = sorted(set(tok for seq in X_train for tok in seq))

    clf = SequenceClassifier(
        max_len=cfg["max_length"],
        embedding_dim=cfg["embedding_dim"],
        device="cpu",
    )
    clf.load(model_path)
    predict_fn = lambda seqs: clf.predict(seqs)

    clf_train_acc = accuracy_score(y_train, predict_fn(X_train))
    clf_test_acc  = accuracy_score(y_test,  predict_fn(X_test))

    print(f"  Neural Network train={clf_train_acc:.4f}  test={clf_test_acc:.4f}")
    
    # Get test instances
    test_instances = get_test_instances(X_train, cfg)
    print(f"  Selected test instances: {len(test_instances)}")
    for i, inst in enumerate(test_instances):
        print(f"    [{i:02d}] len={len(inst)} label={predict_fn([inst])[0]} seq={inst}")

    #  Run each instance independently
    all_instance_results = {}

    for instance_idx, test_instance in enumerate(test_instances):
        result_key = f"{lang_code}_instance_{instance_idx:02d}"

        try:
            all_instance_results[result_key] = run_one_language_instance(
                lang_code=lang_code,
                cfg=cfg,
                output_root=output_root,
                instance_idx=instance_idx,
                test_instance=test_instance,
                X_train=X_train,
                clf_train_acc=clf_train_acc,
                clf_test_acc=clf_test_acc,
                predict_fn=predict_fn,
                alphabet=alphabet,
            )
        except Exception as exc:
            print(f"\n[ERROR] {result_key}: {exc}")
            print("[TRACEBACK]")
            traceback.print_exc()
            all_instance_results[result_key] = None

    return all_instance_results

def run_one_language_instance(
    lang_code: str,
    cfg: dict,
    output_root: str,
    instance_idx: int,
    test_instance: list,
    X_train,
    clf_train_acc: float,
    clf_test_acc: float,
    predict_fn,
    alphabet,
) -> dict | None:
    """
    Run Beam / SA / GA / PSO for one selected test instance.
    """

    print(f"\n{'-' * 70}")
    print(f"  LANGUAGE: {lang_code} | INSTANCE {instance_idx:02d}")
    print(f"{'-' * 70}")

    out_dir = os.path.join(output_root, lang_code, f"instance_{instance_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Test instance: {test_instance}")
    print(f"  Label: {predict_fn([test_instance])[0]}")

    # get Explainer and sampler
    explainer = AnchorTabular(
        predictor=predict_fn,
        feature_names=[f"pos_{i}" for i in range(max(len(s) for s in X_train))],
        categorical_names={},
        seed=42,
    )

    explainer.fit(
        automaton_type="DFA",
        train_data=None,
        alphabet=alphabet,
        disc_perc=None,
        data_type="real_world",
        initial_dfa=None,
    )
    explainer.samplers[0].d_train_data = X_train

    for sampler in explainer.samplers:
        sampler.set_instance_label(test_instance)
        sampler.set_n_covered(10)
        sampler.edit_distance = cfg["edit_distance"]

    sampler_fn = explainer.samplers[0]
    max_evaluations = cfg.get("max_evaluations", None)

    # ══════════════════════════════════════════════════════════════════
    # Beam Search
    # ══════════════════════════════════════════════════════════════════
    print("\n  ─── Beam Search (First - to create/load shared_init) ─────────────────────────")

    # Check if both shared_init and beam_results are cached
    t_shared = time.time()
    shared_data = load_shared_init_from_disk(out_dir)
    shared_build_time = time.time() - t_shared
    beam_cached = load_beam_results_from_disk(out_dir)

    if shared_data is not None and beam_cached is not None:
        # Perfect cache: both shared_data and beam results exist
        print("    [CACHED] Using previously saved shared_init and beam results.")
        initial_state = beam_cached['initial_state']
        initial_train = _extract_accuracy(beam_cached['initial_training_accuracy'])
        initial_validation = _extract_accuracy(beam_cached['initial_validation_accuracy'])
        final_train = _extract_accuracy(beam_cached['final_training_accuracy'])
        final_validation = _extract_accuracy(beam_cached['final_validation_accuracy'])
        final_state = beam_cached['final_state']
        beam_time = beam_cached['time']
        beam_success = beam_cached['success']
        budget_used = beam_cached['budget_used']
    else:
        # Need to run beam search:
        if beam_cached is not None:
            print("    [INFO] Beam results exist but shared_init is corrupted; regenerating shared_init...")
        elif shared_data is not None:
            print("    [INFO] Found shared_init but no beam results; running beam search.")
        else:
            print("    [INFO] No cached data found; running beam search to generate initial DFA.")

        t0 = time.time()
        beam_expl = explainer.explain(
            type               = 'Tabular',
            automaton_type     = 'DFA',
            alphabet           = alphabet,
            X                  = test_instance,
            edit_distance      = cfg['edit_distance'],
            accuracy_threshold = cfg['accuracy_threshold'],
            delta              = cfg['delta'],
            tau                = cfg['tau'],
            beam_size          = cfg['beam_size'],
            batch_size         = cfg['batch_size'],
            init_num_samples   = cfg['init_num_samples'],
            verbose            = False,
            output_dir         = os.path.join(out_dir, "beam"),
            max_evaluations    = max_evaluations,
            task_type = "realworld",
        )
        beam_time_total = time.time() - t0
        init_time = getattr(beam_expl, 'init_automaton_time', 0.0)
        beam_time = max(0.0, beam_time_total - init_time)

        initial_state = getattr(beam_expl, 'initial_state')
        initial_train = _extract_accuracy(getattr(beam_expl, 'initial_training_accuracy'))
        initial_validation = _extract_accuracy(getattr(beam_expl, 'initial_validation_accuracy'))
        final_train  = _extract_accuracy(getattr(beam_expl, 'final_training_accuracy'))
        final_validation = _extract_accuracy(getattr(beam_expl, 'final_validation_accuracy'))
        final_state = getattr(beam_expl, 'final_state')
        beam_success = getattr(beam_expl, 'success')
        budget_used = getattr(beam_expl, 'budget_used', None)
        
        # Ensure final_state is an integer
        if isinstance(final_state, list):
            final_state = final_state[-1] if final_state else 0
        final_state = int(final_state) if final_state else 0
        
        print(f"    train={final_train:.4f}  val={final_validation:.4f}  states={final_state}  time={beam_time:.1f}s"
              f"  {'✓' if beam_success else '✗'}")

        # Save shared_init from beam search
        t_shared = time.time()
        shared_data = build_shared_init_from_beam(explainer, sampler_fn, out_dir, batch_size=cfg['batch_size'], generate_training_data=True)
        shared_build_time = time.time() - t_shared

        # Save beam results to disk for future runs
        beam_results_dict = {
            'initial_state': initial_state,
            'initial_training_accuracy': initial_train,
            'initial_validation_accuracy': initial_validation,
            'final_training_accuracy': final_train,
            'final_validation_accuracy': final_validation,
            'final_state': final_state,
            'time': beam_time,
            'success': beam_success,
            'budget_used': budget_used,
        }
        print(f"  [BEAM] final_state: {final_state}")
        save_beam_results_to_disk(beam_results_dict, out_dir)

    if shared_data is None:
        print("  [SKIP] Could not extract shared_init from beam search.")
        return None
    
    # Reconstruct SharedInit with the SHARED training data
    shared = SharedInit(
        initial_dfa=shared_data['initial_dfa'],
        learner=shared_data['learner'],
        validation_data=shared_data['validation_data'],
        validation_labels=shared_data['validation_labels'],
        training_data=shared_data['training_data'],
        training_labels=shared_data['training_labels']
    )
    
    # Use the run-file budget for SA/GA/PSO (same fixed budget as Beam)
    print(f"  [Budget] Setting SA/GA/PSO max_evaluations={max_evaluations} (run-file budget)")
    
    results: dict = {
        "lang"              : lang_code,
        "clf_train_acc"     : clf_train_acc,
        "clf_test_acc"      : clf_test_acc,
        "initial_dfa_states": initial_state,
        "initial_train_acc" : initial_train,
        "initial_validation_acc": 1.0,  # Validation accuracy on validation_data
        "shared_build_time" : shared_build_time,
    }

    results['beam'] = dict(
        train_acc   = final_train,
        validation_acc = final_validation,
        states      = final_state,
        time        = beam_time,
        success     = beam_success,
    )

    print(f"    train={final_train:.4f}  validation={final_validation:.4f}  "
          f"states={final_state}  time={beam_time:.1f}s"
          f"  {'✓' if beam_success else '✗'}")

    # # ══════════════════════════════════════════════════════════════════════════════════════════════
    # # Method 2 – Simulated Annealing
    # # ══════════════════════════════════════════════════════════════════════════════════════════════
    # print("\n  ─── Simulated Annealing ───────────────────────────────")
    # t0 = time.time()
    # sa_res  = sa_dfa_search(
    #     data_type          = "Tabular",
    #     shared_init        = shared,
    #     accuracy_threshold = cfg['accuracy_threshold'],
    #     init_num_samples   = cfg['init_num_samples'],
    #     batch_size         = cfg['batch_size'],
    #     output_dir         = os.path.join(out_dir, "sa"),
    #     beam_size          = 1,  # Single neighbor per step
    #     steps              = SA_STEPS,
    #     T_max              = 10.0,
    #     T_min              = 0.001,
    #     max_evaluations    = max_evaluations,  # Budget limit: same as Beam Search
    #     instance           = test_instance,
    # )
    # sa_time   = time.time() - t0
    # sa_dfa    = sa_res.get('automata')
    # # training_accuracy: accuracy on perturbation samples from search_baselines
    # sa_train  = _extract_accuracy(sa_res.get('training_accuracy', 0))
    # # validation_accuracy: accuracy on validation samples from beam search (shared.validation_data/validation_labels)
    # sa_validation = eval_validation_acc(sa_dfa, shared.validation_data, shared.validation_labels, shared.learner)
    # sa_states = int(sa_res.get('size', 0) or
    #                 (len(sa_dfa.states) if sa_dfa else 0))
    # results['sa'] = dict(
    #     train_acc   = sa_train,
    #     validation_acc = sa_validation,
    #     states      = sa_states,
    #     time        = sa_time,
    #     success     = sa_res.get('success', False),
    # )
    # print(f"    train={sa_train:.4f}  validation={sa_validation:.4f}  "
    #       f"states={sa_states}  time={sa_time:.1f}s"
    #       f"  {'✓' if sa_res.get('success') else '✗'}")

    # # ══════════════════════════════════════════════════════════════════
    # # Method 3 – Genetic Algorithm
    # # ══════════════════════════════════════════════════════════════════
    # print("\n  ─── Genetic Algorithm ────────────────────────────────")
    # t0 = time.time()
    # ga_res  = ga_dfa_search(
    #     data_type          = "Tabular",
    #     shared_init        = shared,
    #     accuracy_threshold = cfg['accuracy_threshold'],
    #     init_num_samples   = cfg['init_num_samples'],
    #     batch_size         = cfg['batch_size'],
    #     output_dir         = os.path.join(out_dir, "ga"),
    #     population_size    = GA_POPULATION_SIZE,
    #     tournament_size    = 2,  # Tournament selection: pick best from 2 random
    #     max_evaluations    = max_evaluations,  # Budget limit: same as Beam Search
    #     instance           = test_instance,
    # )
    # ga_time   = time.time() - t0
    # ga_dfa    = ga_res.get('automata')
    # # training_accuracy: accuracy on perturbation samples from search_baselines
    # ga_train  = _extract_accuracy(ga_res.get('training_accuracy', 0))
    # # validation_accuracy: accuracy on validation samples from beam search (shared.validation_data/validation_labels)
    # ga_validation = eval_validation_acc(ga_dfa, shared.validation_data, shared.validation_labels, shared.learner)
    # ga_states = int(ga_res.get('size', 0) or
    #                 (len(ga_dfa.states) if ga_dfa else 0))
    # results['ga'] = dict(
    #     train_acc   = ga_train,
    #     validation_acc = ga_validation,
    #     states      = ga_states,
    #     time        = ga_time,
    #     success     = ga_res.get('success', False),
    # )
    # print(f"    train={ga_train:.4f}  validation={ga_validation:.4f}  "
    #       f"states={ga_states}  time={ga_time:.1f}s"
    #       f"  {'✓' if ga_res.get('success') else '✗'}")

    # # ══════════════════════════════════════════════════════════════════
    # # Method 4 – Particle Swarm Optimisation
    # # ══════════════════════════════════════════════════════════════════
    # print("\n  ─── Particle Swarm Optimisation ─────────────────────")
    # t0 = time.time()
    # pso_res  = pso_dfa_search(
    #     data_type          = "Tabular",
    #     shared_init        = shared,
    #     accuracy_threshold = cfg['accuracy_threshold'],
    #     init_num_samples   = cfg['init_num_samples'],
    #     batch_size         = cfg['batch_size'],
    #     output_dir         = os.path.join(out_dir, "pso"),
    #     n_particles        = PSO_PARTICLES,
    #     beam_size          = 1,  # 1 candidate per particle (swarm provides diversity)
    #     max_evaluations    = max_evaluations,  # Budget limit: same as Beam Search
    #     instance           = test_instance,
    #     pso_max_ops_per_iteration=PSO_MAX_OPS_PER_ITERATION,
    # )
    # pso_time   = time.time() - t0
    # pso_dfa    = pso_res.get('automata')
    # pso_train  = _extract_accuracy(pso_res.get('training_accuracy', 0))
    # pso_validation = eval_validation_acc(pso_dfa, shared.validation_data, shared.validation_labels, shared.learner)
    # pso_states = int(pso_res.get('size', 0) or (len(pso_dfa.states) if pso_dfa else 0))
    # results['pso'] = dict(
    #     train_acc   = pso_train,
    #     validation_acc = pso_validation,
    #     states      = pso_states,
    #     time        = pso_time,
    #     success     = pso_res.get('success', False),
    # )
    # print(f"    train={pso_train:.4f}  validation={pso_validation:.4f}  "
    #       f"states={pso_states}  time={pso_time:.1f}s"
    #       f"  {'✓' if pso_res.get('success') else '✗'}")

    # ── Cleanup resources after language experiment ─────────────────────
    print(f"\n  [CLEANUP] Cleaning up resources for {lang_code}...")
    import gc
    
    # Clean up DFA optimizer cache if available
    try:
        from dfa_optimization import _cxp_cache
        if _cxp_cache is not None and hasattr(_cxp_cache, 'clear'):
            _cxp_cache.clear()
            print(f"  [CLEANUP] Cleared CXP cache")
    except:
        pass
    
    # Delete global learner instance to free memory
    try:
        from search_baselines import _AUTO_INSTANCE
        _AUTO_INSTANCE = None
        print(f"  [CLEANUP] Cleared global AUTO_INSTANCE")
    except:
        pass
    
    # Clean up mata files left over from CXP analysis
    mata_files = ['dfa_explicit.mata', 'explanation.txt']
    for fname in mata_files:
        if os.path.exists(fname):
            try:
                os.remove(fname)
                print(f"  [CLEANUP] Removed {fname}")
            except Exception as e:
                print(f"  [CLEANUP] Failed to remove {fname}: {e}")
    
    # Force garbage collection
    gc.collect()
    print(f"  [CLEANUP] Garbage collection complete\n")

    return results


# ======================================================================
# Summary table + CSV export
# ======================================================================
METHODS = ['beam', 'sa', 'ga', 'pso']
METHOD_LABELS = {'beam': 'BeamSearch', 'sa': 'SA', 'ga': 'GA', 'pso': 'PSO'}

def print_summary(all_results: dict, accuracy_threshold: float = 0.9) -> None:
    print("\n\n" + "=" * 100)
    print(f"  EXPERIMENT SUMMARY (threshold={accuracy_threshold:.1f}) - WITH INITIAL & FINAL METRICS")
    print("=" * 100)

    # Per-language results
    for lang_code, res in sorted(all_results.items()):
        if res is None:
            print(f"\n  {lang_code}: [NO DATA]")
            continue

        print(f"\n  {lang_code}  "
              f"(clf_train={res['clf_train_acc']:.4f}  "
              f"clf_test={res['clf_test_acc']:.4f}  "
              f"initial_states={res['initial_dfa_states']})")
        
        # Header for initial/final accuracies
        init_train = res.get('initial_train_acc', 0)
        init_val   = res.get('initial_validation_acc', 0)
        
        print(f"\n  Initial (RPNI):  train={init_train:.4f}  validation={init_val:.4f}")
        print("  " + "─" * 96)
        print(f"  | {'Method':12s} | {'Train (Init→Final)':20s} | {'Validation (Init→Final)':24s} | "
              f"{'States':10s} | {'Time(s)':10s} |")
        print("  " + "─" * 96)
        
        for m in METHODS:
            r = res.get(m, {})
            ok = '✓' if r.get('success') else '✗'
            train_init = res.get('initial_train_acc', 0)
            train_final = r.get('train_acc', 0)
            val_init = res.get('initial_validation_acc', 0)
            val_final = r.get('validation_acc', 0)
            
            # Safely extract states count, handling both int and list formats
            states_val = r.get('states', 0)
            if isinstance(states_val, list):
                states_val = states_val[-1] if states_val else 0
            states_val = int(states_val) if states_val else 0
            
            time_val = r.get('time', 0)
            if isinstance(time_val, list):
                time_val = time_val[-1] if time_val else 0.0
            time_val = float(time_val) if time_val else 0.0
            
            print(f"  | {METHOD_LABELS[m]:12s} | "
                  f"{train_init:.4f}→{train_final:.4f} {ok:1s}       | "
                  f"{val_init:.4f}→{val_final:.4f}       | "
                  f"{states_val:10d} | "
                  f"{time_val:10.1f} |")
        print("  " + "─" * 96)

    # Paper suitability analysis (per-method: initial vs final)
    print("\n\n  PAPER SUITABILITY ANALYSIS")
    print("  " + "─" * 110)
    print(f"  {'Lang':6s}  {'Method':10s}  {'Δ_validation':>14s}  {'Δ_train':>10s}  {'Δ_state':>10s}  {'Init_Val':>10s}  {'Final_Val':>10s}  {'Init_Train':>10s}  {'Final_Train':>10s}  {'Init_States':>10s}  {'Final_States':>10s}")
    print("  " + "─" * 110)

    for lang_code, res in sorted(all_results.items()):
        if res is None:
            print(f"  {lang_code:6s}  [NO DATA]")
            continue
        init_val = res.get('initial_validation_acc', 0)
        init_train = res.get('initial_train_acc', 0)
        init_states = res.get('initial_dfa_states', 0)
        if isinstance(init_states, list):
            init_states = init_states[-1] if init_states else 0
        init_states = int(init_states) if init_states else 0

        for m in ['beam', 'sa', 'ga', 'pso']:
            r = res.get(m, {})
            val = r.get('validation_acc', 0)
            train = r.get('train_acc', 0)
            states = r.get('states', 0)
            if isinstance(states, list):
                states = states[-1] if states else 0
            states = int(states) if states else 0

            delta_val = val - init_val
            delta_train = train - init_train
            delta_states = states - init_states

            print(f"  {lang_code:6s}  {m:10s}  {delta_val:14.4f}  {delta_train:10.4f}  {delta_states:10d}  "
                  f"{init_val:10.4f}  {val:10.4f}  {init_train:10.4f}  {train:10.4f}  {init_states:10d}  {states:10d}")
    print()


def save_csv(all_results: dict, path: str) -> None:
    rows = []
    for lang, res in all_results.items():
        if res is None:
            continue
        for m in METHODS:
            r = res.get(m, {})
            rows.append({
                'language'              : lang,
                'method'                : m,
                'initial_train_acc'     : res.get('initial_train_acc', ''),
                'final_train_acc'       : r.get('train_acc', ''),
                'initial_validation_acc': res.get('initial_validation_acc', ''),
                'final_validation_acc'  : r.get('validation_acc', ''),
                'states'                : r.get('states', ''),
                'time_s'                : r.get('time', ''),
                'success'               : int(r.get('success', False)),
                'clf_train_acc'         : res.get('clf_train_acc', ''),
                'clf_test_acc'          : res.get('clf_test_acc', ''),
                'init_states'           : res.get('initial_dfa_states', ''),
            })
    if not rows:
        print("  [CSV] No rows to write.")
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV saved → {path}")


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run real-world DFA search experiment"
    )

    parser.add_argument(
        "--languages",
        type=str,
        default="mnist,ECG,wafer",
        help="Comma-separated dataset names to run, e.g., mnist or mnist,ECG,wafer"
    )

    # Use None as default so dataset-specific config values are preserved
    # unless the user explicitly overrides them.
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=None,
        help="Override accuracy threshold for selected datasets"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Override delta parameter for beam search"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override tau parameter for beam search"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=None,
        help="Override beam size"
    )
    parser.add_argument(
        "--init_num_samples",
        type=int,
        default=None,
        help="Override number of initial samples"
    )
    parser.add_argument(
        "--edit_distance",
        type=int,
        default=None,
        help="Override maximum edit distance for perturbations"
    )
    parser.add_argument(
        "--max_evaluations",
        type=int,
        default=None,
        help="Override fixed candidate evaluation budget for Beam/SA/GA/PSO"
    )
    parser.add_argument(
        "--num_test_instances",
        type=int,
        default=None,
        help="Override number of test instances"
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
        "max_evaluations": args.max_evaluations,
        "num_test_instances": args.num_test_instances,
    }

    LANGUAGES = get_languages_config(overrides=overrides)

    selected_names = [
        name.strip()
        for name in args.languages.split(",")
        if name.strip()
    ]

    unknown_names = [
        name for name in selected_names
        if name not in LANGUAGES
    ]

    if unknown_names:
        raise ValueError(
            f"Unknown dataset name(s): {unknown_names}\n"
            f"Available datasets: {list(LANGUAGES.keys())}"
        )

    LANGUAGES = {
        name: LANGUAGES[name]
        for name in selected_names
    }

    # Use the first selected dataset config to name the output folder.
    first_cfg = next(iter(LANGUAGES.values()))
    accuracy_threshold = first_cfg["accuracy_threshold"]
    batch_size = first_cfg["batch_size"]

    OUTPUT_ROOT = os.path.join(
        PROJECT_ROOT,
        "test_result",
        f"realworld_{accuracy_threshold}_{batch_size}"
    )
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    log_path = os.path.join(OUTPUT_ROOT, "experiment_log.txt")

    print(f"\n{'=' * 70}")
    print("  Real-World DFA Search Experiment")
    print(f"  Selected datasets: {', '.join(LANGUAGES.keys())}")
    print(f"  Accuracy threshold: {accuracy_threshold}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output directory: {OUTPUT_ROOT}")
    print(f"{'=' * 70}\n")

    _orig_stdout = sys.stdout
    sys.stdout = Tee(log_path)

    try:
        all_results: dict = {}

        for lang_code, cfg in LANGUAGES.items():
            try:
                language_results = run_one_language(
                    lang_code,
                    cfg,
                    OUTPUT_ROOT,
                )

                if language_results:
                    all_results.update(language_results)
                else:
                    all_results[lang_code] = None
            except Exception as exc:
                print(f"\n[ERROR] {lang_code}: {exc}")
                print("[TRACEBACK]")
                traceback.print_exc()
                all_results[lang_code] = None

        print_summary(all_results, accuracy_threshold)
        save_csv(all_results, os.path.join(OUTPUT_ROOT, "results.csv"))

    finally:
        sys.stdout = _orig_stdout

    print(f"\nFull log → {log_path}")