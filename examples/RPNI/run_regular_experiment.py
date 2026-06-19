"""
Regular Automata DFA Search Experiment
=======================================
Searches for DFA explanations for regular automata from DOT files (automata/ folder).
Uses perfect automata DFAs as ground truth teachers.

Usage (from project root):
    python examples/RPNI/run_regular_experiment.py --accuracy_threshold 0.8
    python examples/RPNI/run_regular_experiment.py --accuracy_threshold 0.9 --batch_size 500

Results will be saved to:
    test_result/regular_0.8_500/
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
from typing import Optional, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score

# ── Path setup ─────────────────────────────────
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

# Import local modules first before external dependencies
from search_baselines import sa_dfa_search, ga_dfa_search, pso_dfa_search, SharedInit
from tee import Tee
from load_dfa import load_dfa_from_dot, get_automata_alphabet, create_automata_dfa_predictor

# Import external alibi module last (has complex dependencies)
from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular

# ======================================================================
# Experiment configurations
# ======================================================================

DEFAULT_LANGUAGE_CONFIGS = {
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
    # "LexerTokenization": dict(
    #     automata_name   = "LexerTokenization",
    #     filename        = "lexer_tokenization.dot",
    #     accuracy_threshold = 0.9,
    #     delta           = 0.01,
    #     tau             = 0.05,
    #     batch_size      = 1000,
    #     beam_size       = 1,
    #     init_num_samples= 1000,
    #     edit_distance   = 2,
    #     num_test_instances = 10,
    #     test_instance      = None, # ['id', 'value', 'assign', 'value', 'semicolon', 'semicolon', 'value', 'value']
    #     test_instances     = None,
    # ),
    # "EmbeddedControllerWorkflow": dict(
    #     automata_name   = "EmbeddedControllerWorkflow",
    #     filename        = "embedded_controller_workflow.dot",
    #     accuracy_threshold = 0.9,
    #     delta           = 0.01,
    #     tau             = 0.05,
    #     batch_size      = 1000,
    #     beam_size       = 1,
    #     init_num_samples= 1000,
    #     edit_distance   = 2,
    #     num_test_instances = 10,
    #     test_instance      = None, # ["boot", "sense", "compute", "actuate", "log", "sense", "compute", "log"]
    #     test_instances     = None,
    # ),
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

# Search strategy parameters
SA_STEPS         = 500
GA_POPULATION_SIZE = 10
PSO_PARTICLES    = 5
PSO_MAX_OPS_PER_ITERATION = 1

def random_walk_from_dfa(dfa, alphabet, max_length=10, min_length=1):
    """
    Generate one sequence by random walk from the DFA initial state.
    The sequence can be accepting or rejecting; label is computed later
    by the teacher DFA predictor.
    """
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
    dfa,
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
    Generate n test instances from a regular automaton by random walk.

    By default, keep only accepting sequences because local perturbations
    around rejecting sequences often produce almost all-negative samples,
    which makes RPNI / initial DFA construction fail.
    """
    random.seed(seed)

    if max_attempts is None:
        max_attempts = n * 1000

    test_instances = []
    seen = set()
    attempts = 0

    while len(test_instances) < n and attempts < max_attempts:
        attempts += 1

        seq = random_walk_from_dfa(
            dfa,
            alphabet,
            max_length=max_length,
            min_length=min_length,
        )

        key = tuple(seq)
        if key in seen:
            continue

        label = int(predict_fn([seq])[0])

        # Keep only desired label by default.
        # desired_label=1 means accepting sequence.
        # desired_label=None means keep both accept/reject.
        if desired_label is not None and label != desired_label:
            continue

        seen.add(key)
        test_instances.append(seq)

    if len(test_instances) < n:
        print(
            f"  [WARNING] Only generated {len(test_instances)} "
            f"test instances with desired_label={desired_label} "
            f"after {attempts} attempts."
        )

    return test_instances

# ==================================================================================
# Utility functions for loading/saving beam search results and shared initialization
# ==================================================================================
def _extract_accuracy(value) -> float:
    """Extract scalar accuracy from various formats."""
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(np.mean(value)) if len(value) > 1 else float(value[0])
    return float(value)


def eval_validation_acc(dfa, val_data, val_labels, learner) -> float:
    """DFA accuracy on validation set."""
    if dfa is None or len(val_data) == 0:
        return 0.0
    accepts = np.array([learner.check_path_accepted(dfa, seq) for seq in val_data])
    lbl     = np.asarray(val_labels)
    correct = int(np.sum((lbl == 1) & accepts) + np.sum((lbl == 0) & ~accepts))
    return correct / len(val_data)


def load_beam_results_from_disk(output_dir: str) -> Optional[dict]:
    """Load beam search results from disk."""
    if output_dir is None:
        return None

    shared_dir = os.path.join(output_dir, "shared")
    beam_results_path = os.path.join(shared_dir, "beam_results.pkl")

    if not os.path.exists(beam_results_path):
        return None
    try:
        with open(beam_results_path, 'rb') as f:
            beam_res = pickle.load(f)
        
        if not isinstance(beam_res, dict):
            return None
        
        cached_train = _extract_accuracy(beam_res.get('final_training_accuracy', 0.0))
        cached_val = _extract_accuracy(beam_res.get('final_validation_accuracy', 0.0))
        print(f"  [BeamResults] Loaded: train={cached_train:.4f}, val={cached_val:.4f}, "
              f"states={beam_res.get('final_state', '?')}")
        return beam_res
    except Exception as exc:
        print(f"  [BeamResults] Failed to load: {exc}")
        return None


def load_shared_init_from_disk(output_dir: str) -> Optional[dict]:
    """Load shared initialization from disk."""
    if output_dir is None:
        return None

    shared_dir = os.path.join(output_dir, "shared")
    shared_init_path = os.path.join(shared_dir, "shared_init.pkl")

    if not os.path.exists(shared_init_path):
        return None

    try:
        with open(shared_init_path, 'rb') as f:
            obj = pickle.load(f)
        
        if isinstance(obj, dict) and 'initial_dfa' in obj:
            print(f"  [SharedInit] Loaded: {len(obj['initial_dfa'].states)} states, "
                  f"{len(obj['validation_data'])} samples")
            return obj
        elif isinstance(obj, (tuple, list)) and len(obj) >= 4:
            initial_dfa, learner, validation_data, validation_labels = obj[:4]
            return {
                'initial_dfa': initial_dfa,
                'learner': learner,
                'validation_data': validation_data,
                'validation_labels': validation_labels,
            }
        return None
    except TypeError as e:
        print(f"  [SharedInit] Old format (will regenerate): {e}")
        return None
    except Exception as exc:
        print(f"  [SharedInit] Failed to load: {exc}")
        return None


def save_beam_results_to_disk(beam_results: dict, output_dir: str = None) -> bool:
    """Save beam search results to disk."""
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

def build_shared_init_from_beam(explainer, sampler_fn, output_dir: str = None, batch_size: int = 500) -> Optional[dict]:
    """Extract and save shared initialization from beam search."""
    mab = getattr(explainer, 'mab', None)
    automatas = getattr(mab, 'automatas', None) or []
    
    # Check if automatas list is empty (beam search failed)
    if not automatas:
        print(f"  [ERROR] No automatas found in explainer (beam search initialization failed)")
        return None
    
    initial_dfa = automatas[0].copy()
    validation_data = list(getattr(mab, 'validation_data'))
    validation_labels = np.asarray(getattr(mab, 'validation_labels'))

    from learner.dfa_learner import DFALearner
    learner = DFALearner()

    training_data_raw, training_labels = sampler_fn(num_samples=batch_size, compute_labels=True)
    training_data = list(training_data_raw)
    training_labels = np.asarray(training_labels)

    shared_data = {
        'initial_dfa': initial_dfa,
        'learner': learner,
        'validation_data': validation_data,
        'validation_labels': validation_labels,
        'training_data': training_data,
        'training_labels': training_labels,
    }

    if output_dir is not None:
        shared_dir = os.path.join(output_dir, "shared")
        try:
            os.makedirs(shared_dir, exist_ok=True)
            shared_init_path = os.path.join(shared_dir, "shared_init.pkl")
            with open(shared_init_path, 'wb') as f:
                pickle.dump(shared_data, f)
            print(f"  [SharedInit] Saved: {len(initial_dfa.states)} states, "
                  f"{len(validation_data)} validation samples")
        except Exception as exc:
            print(f"  [WARNING] Could not save shared init: {exc}")

    return shared_data

# ======================================================================
# Experiment
# ======================================================================
def run_one_automata(automata_code: str, cfg: dict, output_root: str) -> dict | None:
    print(f"\n{'=' * 70}")
    print(f"  AUTOMATA: {automata_code}")
    print(f"{'=' * 70}")

    # Load teacher DFA
    try:
        print(f"  [TEACHER] Loading automata DFA: {cfg['filename']}")
        teacher_dfa = load_dfa_from_dot(cfg["filename"])

        if teacher_dfa.initial_state is None or not hasattr(teacher_dfa.initial_state, "state_id"):
            for state_name in ["S0", "Start_A0C0H0", "P00_0", "B0", "q0"]:
                candidate = next(
                    (s for s in teacher_dfa.states if s.state_id == state_name),
                    None,
                )
                if candidate:
                    teacher_dfa.initial_state = candidate
                    print(f"  [INFO] Set initial state to: {state_name}")
                    break

            if teacher_dfa.initial_state is None:
                teacher_dfa.initial_state = teacher_dfa.states[0]
                print(
                    f"  [INFO] Set initial state to first state: "
                    f"{teacher_dfa.states[0].state_id}"
                )

        alphabet = get_automata_alphabet(teacher_dfa)
        predict_fn = create_automata_dfa_predictor(teacher_dfa)

        print(f"  DFA: {len(teacher_dfa.states)} states, alphabet={alphabet}")

    except Exception as e:
        print(f"  [ERROR] Failed to load automata DFA: {e}")
        traceback.print_exc()
        return None

    # Generate or load multiple test instances
    if cfg.get("test_instances") is not None:
        test_instances = [list(seq) for seq in cfg["test_instances"]]

    elif cfg.get("test_instance") is not None:
        test_instances = [list(cfg["test_instance"])]

    else:
        test_instances = get_test_instances(
            dfa=teacher_dfa,
            alphabet=alphabet,
            predict_fn=predict_fn,
            n=cfg.get("num_test_instances", 10),
            max_length=cfg.get("max_length", 20),
            min_length=cfg.get("min_test_length", 10),
            seed=cfg.get("seed", 42),
        )

    print(f"  Selected test instances: {len(test_instances)}")
    for i, inst in enumerate(test_instances):
        print(
            f"    [{i:02d}] len={len(inst)} "
            f"label={predict_fn([inst])[0]} seq={inst}"
        )

    # Run each instance independently
    all_instance_results = {}

    for instance_idx, test_instance in enumerate(test_instances):
        result_key = f"{automata_code}_instance_{instance_idx:02d}"

        try:
            all_instance_results[result_key] = run_one_automata_instance(
                automata_code=automata_code,
                cfg=cfg,
                output_root=output_root,
                instance_idx=instance_idx,
                test_instance=test_instance,
                teacher_dfa=teacher_dfa,
                predict_fn=predict_fn,
                alphabet=alphabet,
            )

        except Exception as exc:
            print(f"\n[ERROR] {result_key}: {exc}")
            print("[TRACEBACK]")
            traceback.print_exc()
            all_instance_results[result_key] = None

    return all_instance_results

def run_one_automata_instance(
    automata_code: str,
    cfg: dict,
    output_root: str,
    instance_idx: int,
    test_instance: list,
    teacher_dfa,
    predict_fn,
    alphabet,
) -> dict | None:
    """
    Run Beam / SA / GA / PSO for one random-walk test instance.
    """

    print(f"\n{'-' * 70}")
    print(f"  AUTOMATA: {automata_code} | INSTANCE {instance_idx:02d}")
    print(f"{'-' * 70}")

    out_dir = os.path.join(
        output_root,
        automata_code,
        f"instance_{instance_idx:02d}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Test instance: {test_instance}")
    print(f"  Label: {predict_fn([test_instance])[0]}")

    # Explainer + sampler
    print("  Initializing AnchorTabular with direct DFASampler...")

    explainer = AnchorTabular(
        predictor=predict_fn,
        feature_names=["seq_token"],
        categorical_names={},
        seed=42 + instance_idx,
    )

    explainer.fit(
        automaton_type="DFA",
        train_data=None,
        alphabet=alphabet,
        disc_perc=None,
        data_type="automata",
        initial_dfa=teacher_dfa,
    )

    for sampler in explainer.samplers:
        sampler.set_instance_label(test_instance)
        sampler.set_n_covered(10)
        sampler.edit_distance = cfg["edit_distance"]

    sampler_fn = explainer.samplers[0]
    max_evaluations = cfg.get("max_evaluations", None)
    
    # ══════════════════════════════════════════════════════════════════
    # Beam Search
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  ─── Beam Search ─────────────────────────────────────────")

    t_shared = time.time()
    shared_data = load_shared_init_from_disk(out_dir)
    beam_cached = load_beam_results_from_disk(out_dir)
    shared_build_time = time.time() - t_shared

    # reuse cached beam search results and shared initialization
    if shared_data is not None and beam_cached is not None:
        print("    [CACHED] Using previously saved results.")
        initial_state = beam_cached['initial_state']
        initial_train = _extract_accuracy(beam_cached['initial_training_accuracy'])
        initial_validation = _extract_accuracy(beam_cached['initial_validation_accuracy'])
        final_train = _extract_accuracy(beam_cached['final_training_accuracy'])
        final_validation = _extract_accuracy(beam_cached['final_validation_accuracy'])
        final_state = beam_cached['final_state']
        beam_time = beam_cached['time']
        beam_success = beam_cached['success']
        budget_used = beam_cached['budget_used']
    # run beam search and save results + shared initialization
    else:
        print("    [RUN] Running beam search...")
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
            task_type = "regular",
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
        
        if isinstance(final_state, list):
            final_state = final_state[-1] if final_state else 0
        final_state = int(final_state) if final_state else 0
        
        print(f"    train={final_train:.4f}  val={final_validation:.4f}  "
              f"states={final_state}  time={beam_time:.1f}s {'✓' if beam_success else '✗'}")

        t_shared = time.time()
        shared_data = build_shared_init_from_beam(explainer, sampler_fn, out_dir, batch_size=cfg['batch_size'])
        shared_build_time = time.time() - t_shared

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
        save_beam_results_to_disk(beam_results_dict, out_dir)

    if shared_data is None:
        print("  [SKIP] Could not extract shared_init.")
        return None
    
    shared = SharedInit(
        initial_dfa=shared_data['initial_dfa'],
        learner=shared_data['learner'],
        validation_data=shared_data['validation_data'],
        validation_labels=shared_data['validation_labels'],
        training_data=shared_data['training_data'],
        training_labels=shared_data['training_labels']
    )
    
    results: dict = {
        "automata"          : automata_code,
        "teacher_train_acc" : 1.0,
        "teacher_test_acc"  : 1.0,
        "initial_dfa_states": initial_state,
        "initial_train_acc" : initial_train,
        "initial_validation_acc": initial_validation,
        "shared_build_time" : shared_build_time,
        "teacher_dfa_states": len(teacher_dfa.states),
    }

    results['beam'] = dict(
        train_acc   = final_train,
        validation_acc = final_validation,
        states      = final_state,
        time        = beam_time,
        success     = beam_success,
    )

    # # ══════════════════════════════════════════════════════════════════════════════════════════════
    # # Method 2 – Simulated Annealing
    # # ══════════════════════════════════════════════════════════════════════════════════════════════
    # print(f"\n  ─── Simulated Annealing ───────────────────────────────")
    # t0 = time.time()
    # sa_res  = sa_dfa_search(
    #     data_type          = "Tabular",
    #     shared_init        = shared,
    #     accuracy_threshold = cfg['accuracy_threshold'],
    #     init_num_samples   = cfg['init_num_samples'],
    #     batch_size         = cfg['batch_size'],
    #     output_dir         = os.path.join(out_dir, "sa"),
    #     beam_size          = 1,
    #     steps              = SA_STEPS,
    #     T_max              = 10.0,
    #     T_min              = 0.001,
    #     max_evaluations    = max_evaluations,
    #     instance           = test_instance,
    # )
    # sa_time   = time.time() - t0
    # sa_dfa    = sa_res.get('automata')
    # sa_train  = _extract_accuracy(sa_res.get('training_accuracy', 0))
    # sa_validation = eval_validation_acc(sa_dfa, shared.validation_data, shared.validation_labels, shared.learner)
    # sa_states = int(sa_res.get('size', 0) or (len(sa_dfa.states) if sa_dfa else 0))
    # results['sa'] = dict(
    #     train_acc   = sa_train,
    #     validation_acc = sa_validation,
    #     states      = sa_states,
    #     time        = sa_time,
    #     success     = sa_res.get('success', False),
    # )
    # print(f"    train={sa_train:.4f}  validation={sa_validation:.4f}  "
    #       f"states={sa_states}  time={sa_time:.1f}s {'✓' if sa_res.get('success') else '✗'}")

    # # ══════════════════════════════════════════════════════════════════
    # # Method 3 – Genetic Algorithm
    # # ══════════════════════════════════════════════════════════════════
    # print(f"\n  ─── Genetic Algorithm ────────────────────────────────")
    # t0 = time.time()
    # ga_res  = ga_dfa_search(
    #     data_type          = "Tabular",
    #     shared_init        = shared,
    #     accuracy_threshold = cfg['accuracy_threshold'],
    #     init_num_samples   = cfg['init_num_samples'],
    #     batch_size         = cfg['batch_size'],
    #     output_dir         = os.path.join(out_dir, "ga"),
    #     population_size    = GA_POPULATION_SIZE,
    #     tournament_size    = 2,
    #     max_evaluations    = max_evaluations,
    #     instance           = test_instance,
    # )
    # ga_time   = time.time() - t0
    # ga_dfa    = ga_res.get('automata')
    # ga_train  = _extract_accuracy(ga_res.get('training_accuracy', 0))
    # ga_validation = eval_validation_acc(ga_dfa, shared.validation_data, shared.validation_labels, shared.learner)
    # ga_states = int(ga_res.get('size', 0) or (len(ga_dfa.states) if ga_dfa else 0))
    # results['ga'] = dict(
    #     train_acc   = ga_train,
    #     validation_acc = ga_validation,
    #     states      = ga_states,
    #     time        = ga_time,
    #     success     = ga_res.get('success', False),
    # )
    # print(f"    train={ga_train:.4f}  validation={ga_validation:.4f}  "
    #       f"states={ga_states}  time={ga_time:.1f}s {'✓' if ga_res.get('success') else '✗'}")

    # # ══════════════════════════════════════════════════════════════════
    # # Method 4 – Particle Swarm Optimisation
    # # ══════════════════════════════════════════════════════════════════
    # print(f"\n  ─── Particle Swarm Optimisation ─────────────────────")
    # t0 = time.time()
    # pso_res  = pso_dfa_search(
    #     data_type          = "Tabular",
    #     shared_init        = shared,
    #     accuracy_threshold = cfg['accuracy_threshold'],
    #     init_num_samples   = cfg['init_num_samples'],
    #     batch_size         = cfg['batch_size'],
    #     output_dir         = os.path.join(out_dir, "pso"),
    #     n_particles        = PSO_PARTICLES,
    #     beam_size          = 1,
    #     max_evaluations    = max_evaluations,
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
    #       f"states={pso_states}  time={pso_time:.1f}s {'✓' if pso_res.get('success') else '✗'}")

    # ── Cleanup ────────────────────────────────────────────────────────
    print(f"\n  [CLEANUP] Cleaning up resources...")
    import gc
    try:
        from dfa_optimization import _cxp_cache
        if _cxp_cache is not None and hasattr(_cxp_cache, 'clear'):
            _cxp_cache.clear()
    except:
        pass
    
    try:
        from search_baselines import _AUTO_INSTANCE
        _AUTO_INSTANCE = None
    except:
        pass
    
    for fname in ['dfa_explicit.mata', 'explanation.txt']:
        if os.path.exists(fname):
            try:
                os.remove(fname)
            except:
                pass
    
    gc.collect()

    return results


# ======================================================================
# Summary table + CSV export
# ======================================================================
METHODS = ['beam', 'sa', 'ga', 'pso']
METHOD_LABELS = {'beam': 'BeamSearch', 'sa': 'SA', 'ga': 'GA', 'pso': 'PSO'}

def print_summary(all_results: dict, accuracy_threshold: float = 0.8) -> None:
    print("\n\n" + "=" * 100)
    print(f"  AUTOMATA SEARCH SUMMARY (threshold={accuracy_threshold:.1f})")
    print("=" * 100)

    for automata_code, res in sorted(all_results.items()):
        if res is None:
            print(f"\n  {automata_code}: [NO DATA]")
            continue

        print(f"\n  {automata_code}  "
              f"(teacher_states={res['teacher_dfa_states']}  "
              f"teacher_train={res['teacher_train_acc']:.4f}  "
              f"teacher_test={res['teacher_test_acc']:.4f})")
        
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


def save_csv(all_results: dict, path: str) -> None:
    rows = []
    for automata, res in all_results.items():
        if res is None:
            continue
        for m in METHODS:
            r = res.get(m, {})
            rows.append({
                'automata'              : automata,
                'method'                : m,
                'teacher_states'        : res.get('teacher_dfa_states', ''),
                'teacher_train_acc'     : res.get('teacher_train_acc', ''),
                'teacher_test_acc'      : res.get('teacher_test_acc', ''),
                'initial_train_acc'     : res.get('initial_train_acc', ''),
                'final_train_acc'       : r.get('train_acc', ''),
                'initial_validation_acc': res.get('initial_validation_acc', ''),
                'final_validation_acc'  : r.get('validation_acc', ''),
                'states'                : r.get('states', ''),
                'time_s'                : r.get('time', ''),
                'success'               : int(r.get('success', False)),
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
        description="Run regular automata DFA search experiment"
    )

    parser.add_argument(
        "--languages",
        type=str,
        default="SecureHandshake,DocumentReleaseWorkflow,MultiObligationOrder",
        help="Comma-separated automata names to run",
    )

    # Global overrides. Use None so config defaults are preserved unless specified.
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=None,
        help="Override accuracy threshold for selected automata",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Override delta parameter for beam search",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override tau parameter for beam search",
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
        help="Override number of initial samples",
    )
    parser.add_argument(
        "--edit_distance",
        type=int,
        default=None,
        help="Override maximum edit distance for perturbations",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Override maximum sequence length",
    )
    parser.add_argument(
        "--max_evaluations",
        type=int,
        default=None,
        help="Override fixed candidate evaluation budget",
    )
    parser.add_argument(
        "--num_test_instances",
        type=int,
        default=None,
        help="Override number of test instances",
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
        "max_evaluations": args.max_evaluations,
        "num_test_instances": args.num_test_instances,
    }

    # Use get_language_config for regular experiments.
    AUTOMATA = get_languages_config(overrides=overrides)

    selected_names = [name.strip() for name in args.languages.split(",") if name.strip()]
    unknown_names = [name for name in selected_names if name not in AUTOMATA]

    if unknown_names:
        raise ValueError(
            f"Unknown automata name(s): {unknown_names}\n"
            f"Available automata: {list(AUTOMATA.keys())}"
        )

    AUTOMATA = {
        name: AUTOMATA[name]
        for name in selected_names
    }

    # For output folder name, use the first selected config after override.
    first_cfg = next(iter(AUTOMATA.values()))
    accuracy_threshold = first_cfg["accuracy_threshold"]
    batch_size = first_cfg["batch_size"]

    OUTPUT_ROOT = os.path.join(
        PROJECT_ROOT,
        "test_result",
        f"regular_{accuracy_threshold}_{batch_size}"
    )
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log_path = os.path.join(OUTPUT_ROOT, "experiment_log.txt")

    print(f"\n{'=' * 70}")
    print("  Regular Automata DFA Search Experiment")
    print(f"  Selected automata: {', '.join(AUTOMATA.keys())}")
    print(f"  Accuracy threshold: {accuracy_threshold}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output directory: {OUTPUT_ROOT}")
    print(f"{'=' * 70}\n")

    _orig_stdout = sys.stdout
    sys.stdout = Tee(log_path)

    try:
        all_results: dict = {}

        for automata_code, cfg in AUTOMATA.items():
            try:
                automata_results = run_one_automata(
                    automata_code,
                    cfg,
                    OUTPUT_ROOT,
                )
                if automata_results:
                    all_results.update(automata_results)
                else:
                    all_results[automata_code] = None
            except Exception as exc:
                print(f"\n[ERROR] {automata_code}: {exc}")
                print("[TRACEBACK]")
                traceback.print_exc()
                all_results[automata_code] = None

        print_summary(all_results, accuracy_threshold)
        save_csv(all_results, os.path.join(OUTPUT_ROOT, "results.csv"))

    finally:
        sys.stdout = _orig_stdout

    print(f"\nFull log → {log_path}")
