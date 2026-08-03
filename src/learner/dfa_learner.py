"""
DFA learner and refinement operators.

This module contains DFALearner, which creates initial DFA explanations
and proposes candidate DFAs through DELETE, MERGE, and DELTA operations.
"""

import copy
import gc
import importlib
import itertools
import random
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Tuple, List, List, Optional
from collections import defaultdict
import threading
import numpy as np
from aalpy.automata.Dfa import DfaState

# Add parent directory to path for scar_rpni import
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if _parent_path not in sys.path:
    sys.path.insert(0, _parent_path)

# import external_modules
_external_modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_modules/Explaining-FA'))
if _external_modules_path not in sys.path:
    sys.path.insert(0, _external_modules_path)

# import ExplainLanguage
try:
    ExplainLanguage = importlib.import_module("language.explain").Language
    EXPLAIN_LANGUAGE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] ExplainLanguage not available at module import: {e}")
    EXPLAIN_LANGUAGE_AVAILABLE = False
    ExplainLanguage = None

from .base import BaseAutomataLearner
from automaton.dfa_utils import (
    check_dfa_path_accepted,
    is_valid_dfa,
    remove_unreachable_states,
    dfa_to_mata,
    serialize_dfa,
)
from automaton.metrics import compute_acceptance_stats, compute_automaton_agreement

# Per-path hard cap on ExplainLanguage.explain_word() (SAT-based minimal
# hitting-set enumeration). Some (automaton, path) combinations make the
# underlying solver blow up (observed: minutes) and the library's own
# signal-based timeout does not actually interrupt it (its SIGALRM handler
# is a no-op). Paths that exceed this bound are treated the same as any
# other explain_word failure: no CXP record contributed for that path.
_CXP_EXPLAIN_TIMEOUT_SECONDS = 60

def _collect_cxp_records_for_path(args):
    dfa, path, alphabet_map, mata_path, collect_missing = args
    cxp_records = []
    missing_transition_records = []

    current = dfa.initial_state
    path_edges = []
    fully_traced = True
    missing_symbol = None
    missing_src_state_id = None

    for sym in path:
        next_state = current.transitions.get(sym)
        if next_state is None:
            fully_traced = False
            missing_symbol = sym
            missing_src_state_id = current.state_id
            break
        path_edges.append((current.state_id, sym, next_state.state_id))
        current = next_state

    if not path_edges and not fully_traced:
        if collect_missing and missing_src_state_id is not None and missing_symbol is not None:
            missing_transition_records.append((missing_src_state_id, missing_symbol))
        return cxp_records, missing_transition_records

    if not fully_traced:
        if collect_missing and missing_src_state_id is not None and missing_symbol is not None:
            missing_transition_records.append((missing_src_state_id, missing_symbol))
        return cxp_records, missing_transition_records

    if not path_edges:
        return cxp_records, missing_transition_records

    filtered_path_edges = []
    encoded_word = []
    for edge in path_edges:
        _, sym, _ = edge
        if sym not in alphabet_map:
            return cxp_records, missing_transition_records
        filtered_path_edges.append(edge)
        encoded_word.append(alphabet_map[sym])

    if not encoded_word:
        return cxp_records, missing_transition_records

    if not EXPLAIN_LANGUAGE_AVAILABLE or ExplainLanguage is None:
        return cxp_records, missing_transition_records

    try:
        engine = ExplainLanguage()
        result_data = engine.explain_word(
            mata_path,
            from_mata=True,
            word=encoded_word,
            ascii=encoded_word,
            target_axp=False,
            bootstrap_cxp_size_1=False,
            print_exp=False,
        )
    except Exception:
        return cxp_records, missing_transition_records

    if not isinstance(result_data, dict):
        return cxp_records, missing_transition_records

    cxp_raw = result_data.get("cxps", []) or []
    valid_cxps = []
    for cxp in cxp_raw:
        try:
            cxp_tuple = tuple(int(pos) for pos in cxp)
        except Exception:
            continue
        if cxp_tuple:
            valid_cxps.append(cxp_tuple)

    if valid_cxps:
        min_len = min(len(c) for c in valid_cxps)
        shortest_cxps = [c for c in valid_cxps if len(c) == min_len]
        for cxp_tuple in shortest_cxps:
            cxp_records.append({"cxp": cxp_tuple, "path_edges": filtered_path_edges})

    return cxp_records, missing_transition_records


def _delete_candidate_from_state(args):
    dfa, target_state_id = args
    try:
        target_state = next((state for state in dfa.states if state.state_id == target_state_id), None)
        if target_state is None:
            return None
        if target_state == dfa.initial_state or (target_state.is_accepting and sum(x.is_accepting for x in dfa.states) <= 1):
            return None

        new_dfa = dfa.copy()
        target_state_in_new = next((x for x in new_dfa.states if x.state_id == target_state.state_id), None)
        if target_state_in_new is None:
            return None

        outgoing = dict(target_state_in_new.transitions)
        for st in list(new_dfa.states):
            for sym, next_s in list(st.transitions.items()):
                if next_s == target_state_in_new:
                    st.transitions[sym] = outgoing.get(sym, st)

        if target_state_in_new in new_dfa.states:
            new_dfa.states.remove(target_state_in_new)

        for st in new_dfa.states:
            for sym, nxt in list(st.transitions.items()):
                if nxt not in new_dfa.states:
                    st.transitions[sym] = st

        remove_unreachable_states(new_dfa)
        if not any(st.is_accepting for st in new_dfa.states):
            return None
        return new_dfa
    except Exception:
        return None


def _merge_candidate_from_pair(args):
    dfa, state1_id, state2_id, majority_labels = args
    try:
        s1 = next((state for state in dfa.states if state.state_id == state1_id), None)
        s2 = next((state for state in dfa.states if state.state_id == state2_id), None)
        if s1 is None or s2 is None:
            return None
        if s1 == dfa.initial_state or s2 == dfa.initial_state:
            return None

        new_dfa = dfa.copy()
        s1_new = next((x for x in new_dfa.states if x.state_id == s1.state_id), None)
        s2_new = next((x for x in new_dfa.states if x.state_id == s2.state_id), None)
        if s1_new is None or s2_new is None:
            return None

        for st in list(new_dfa.states):
            for sym, nxt in list(st.transitions.items()):
                if nxt == s2_new:
                    st.transitions[sym] = s1_new

        for sym, nxt in s2_new.transitions.items():
            s1_new.transitions[sym] = nxt

        s1_majority = majority_labels.get(s1.state_id, 0)
        s2_majority = majority_labels.get(s2.state_id, 0)
        merged_majority = max([s1_majority, s2_majority], default=0)
        s1_new.is_accepting = (merged_majority == 1)

        if isinstance(new_dfa.states, set):
            new_dfa.states.discard(s2_new)
        else:
            try:
                new_dfa.states.remove(s2_new)
            except ValueError:
                pass

        remove_unreachable_states(new_dfa)
        if not any(st.is_accepting for st in new_dfa.states):
            return None
        return new_dfa
    except Exception:
        return None


def _delta_candidate_from_edge(args):
    dfa, source_state_id, symbol, target_state_id = args
    try:
        src_new = next((x for x in dfa.states if x.state_id == source_state_id), None)
        target_new = next((x for x in dfa.states if x.state_id == target_state_id), None)
        if src_new is None or target_new is None:
            return None
        if symbol not in src_new.transitions:
            return None

        candidate = dfa.copy()
        src_copy = next((x for x in candidate.states if x.state_id == source_state_id), None)
        target_copy = next((x for x in candidate.states if x.state_id == target_state_id), None)
        if src_copy is None or target_copy is None:
            return None

        src_copy.transitions[symbol] = target_copy
        remove_unreachable_states(candidate)
        if not any(st.is_accepting for st in candidate.states):
            return None
        return candidate
    except Exception:
        return None


class DFASampler:
    """
    Sequence sampler for local DFA explanation.

    Labels always mean local agreement:
    label = 1 if predict_fn(sample) == predict_fn(instance)
    label = 0 otherwise.
    """

    def __init__(
        self,
        predictor=None,
        alphabet: Optional[List] = None,
        seed: int | None = None,
        edit_distance: int = 1,
        use_prediction_cache: bool = True,
        prediction_cache_max_size: int | None = 200000,
    ):

        self.predictor = predictor
        self.alphabet = list(alphabet or [])
        self.edit_distance = edit_distance
        self.instance = None
        self.instance_label = None
        self.n_covered_ex = 10
        self.d_train_data = None

        # Cache teacher predictions by sequence.
        # This covers both real-world RNN teachers and regular DFA teachers,
        # because both are queried through self.predictor(samples).
        self.use_prediction_cache = bool(use_prediction_cache)
        self.prediction_cache_max_size = prediction_cache_max_size
        self._prediction_cache = {}
        self._prediction_cache_lock = threading.Lock()
        self._prediction_cache_hits = 0
        self._prediction_cache_misses = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _sequence_cache_key(self, seq):
        """Return a stable hashable key for a sequence."""
        if isinstance(seq, np.ndarray):
            seq = seq.tolist()
        return tuple(seq)

    def _normalise_prediction_value(self, value):
        """Convert numpy scalar predictions to plain Python values for caching."""
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
        return tuple(arr.tolist())

    def _cache_put_locked(self, key, value):
        """Write one prediction to the cache. Caller must hold the cache lock."""
        self._prediction_cache[key] = value
        if self.prediction_cache_max_size is not None:
            max_size = int(self.prediction_cache_max_size)
            while max_size > 0 and len(self._prediction_cache) > max_size:
                # FIFO-style eviction. Dict preserves insertion order in modern Python.
                self._prediction_cache.pop(next(iter(self._prediction_cache)))

    def _predict_with_cache(self, samples):
        """
        Predict teacher labels with a thread-safe sequence-level cache.

        The cache key is tuple(sequence).  This speeds up real-world tasks by
        avoiding repeated RNN predictions, and it also speeds up regular tasks
        by avoiding repeated teacher-DFA simulations for identical samples.
        """
        if not self.use_prediction_cache:
            return np.asarray(self.predictor(samples))

        samples = [list(seq) if not isinstance(seq, np.ndarray) else seq.tolist() for seq in samples]
        output = [None] * len(samples)
        miss_indices = []
        miss_keys = []
        miss_samples = []

        # Protect all reads from the shared dict and hit/miss counters.
        with self._prediction_cache_lock:
            for idx, seq in enumerate(samples):
                key = self._sequence_cache_key(seq)
                if key in self._prediction_cache:
                    output[idx] = self._prediction_cache[key]
                    self._prediction_cache_hits += 1
                else:
                    miss_indices.append(idx)
                    miss_keys.append(key)
                    miss_samples.append(seq)

        if miss_samples:
            # Do the expensive model/DFA prediction outside the lock so other
            # threads can keep reading cached values instead of blocking.
            miss_preds = np.asarray(self.predictor(miss_samples))
            miss_values = [self._normalise_prediction_value(v) for v in miss_preds]

            # Protect writes to the shared dict and miss counter.
            with self._prediction_cache_lock:
                for key, value in zip(miss_keys, miss_values):
                    self._cache_put_locked(key, value)
                    self._prediction_cache_misses += 1

            for idx, value in zip(miss_indices, miss_values):
                output[idx] = value

        return np.asarray(output)

    def prediction_cache_info(self):
        """Return lightweight cache statistics for debugging or experiment metadata."""
        with self._prediction_cache_lock:
            return {
                "enabled": self.use_prediction_cache,
                "size": len(self._prediction_cache),
                "max_size": self.prediction_cache_max_size,
                "hits": self._prediction_cache_hits,
                "misses": self._prediction_cache_misses,
            }

    def set_instance_label(self, X):
        """
        Set the target instance for DFA-based perturbation.
        """
        self.instance = list(X)
        self.instance_label = int(np.asarray(self._predict_with_cache([self.instance]))[0])


    def set_n_covered(self, n):
        self.n_covered_ex = n


    def build_lookups(self, data=None):
        return ({}, {}, {})
    

    def compare_labels(self, samples):
        """
        Return local agreement labels.

        labels[i] = 1 if teacher prediction on samples[i]
                    equals teacher prediction on the original instance.
        labels[i] = 0 otherwise.
        """
        preds = np.asarray(self._predict_with_cache(samples))
        return np.asarray(preds == self.instance_label, dtype=int)
    

    def perturbation(self, num_samples: int):
        """
        DFASampler perturbation - uses shared DFALearner._generate_perturbed_samples
        
        Generates unique perturbed samples with intelligent deduplication.
        
        ----------
        num_samples : int
            Target number of unique samples to generate
            
        Returns
        -------
        tuple
            (local_paths, d_samples) - both are lists of unique perturbed sequences
        """
        symbols = self.alphabet if self.alphabet else list(set(self.instance))
        if not symbols:
            raise ValueError("DFASampler requires a non-empty alphabet or non-empty instance.")
        
        edit_distance = self.edit_distance
        local_paths_set = set()  # Use set for automatic deduplication
        no_progress_count = 0  # Track consecutive failures
        max_no_progress = 50  # If no new samples in 100 tries, give up
        trials = 0
        max_trials = 10000
        
        while len(local_paths_set) < int(num_samples) and trials < max_trials:
            trials += 1
            
            new_instance = list(self.instance)
            op = random.choice(["replace", "insert", "delete"])
            max_edit = min(edit_distance, len(new_instance))
            edit_dist = random.randint(0, max_edit) if max_edit > 0 else 0

            if op == "replace":
                if len(new_instance) > 0 and edit_dist > 0:
                    replace_indices = random.sample(range(len(new_instance)), min(edit_dist, len(new_instance)))
                    for idx in replace_indices:
                        different_symbols = [s for s in symbols if s != new_instance[idx]]
                        if different_symbols:
                            new_instance[idx] = random.choice(different_symbols)

            elif op == "insert":
                for _ in range(edit_dist):
                    insert_idx = random.randint(0, len(new_instance))
                    new_instance.insert(insert_idx, random.choice(symbols))

            elif op == "delete":
                if len(new_instance) > 0 and edit_dist > 0:
                    delete_count = min(edit_dist, len(new_instance))
                    delete_indices = sorted(random.sample(range(len(new_instance)), delete_count), reverse=True)
                    for idx in delete_indices:
                        del new_instance[idx]

            # Convert to hashable tuple for deduplication
            hashable_instance = tuple(new_instance)
            
            # Track progress for early exit
            prev_size = len(local_paths_set)
            local_paths_set.add(hashable_instance)
            
            if len(local_paths_set) == prev_size:
                no_progress_count += 1
            else:
                no_progress_count = 0
            
            # Early exit if no progress for too long
            if no_progress_count > max_no_progress:
                break

        local_paths = [list(seq) for seq in local_paths_set]
        return local_paths, local_paths
        
        
    def __call__(self, num_samples, compute_labels=True):
        if self.instance is None:
            raise ValueError("DFASampler instance is not set. Call set_instance_label(X) first.")

        raw_data, d_raw_data = self.perturbation(num_samples)

        # Generate labels by comparing predictions with instance_label
        if compute_labels:
            labels = self.compare_labels(raw_data)
            return [raw_data, labels.astype(int)]

        return [d_raw_data]
    

class DFALearner(BaseAutomataLearner):
    
    def __init__(
        self,
        predictor=None,
        alphabet: List = [],
        seed: int = None,
        edit_distance: int = 1,
        data_type: str = "automata",
        initial_dfa=None,
    ):
        self.predictor = predictor
        self.initial_dfa = initial_dfa
        self.alphabet = alphabet
        self.edit_distance = edit_distance
        self.data_type = data_type
        self.instance_label = None
        self.instance = None
        self.n_covered_ex = 10
        # Last operation metadata used by baseline logging/statistics.
        self.last_proposed_operation = None
        self.last_proposed_ops = []
        # Reused across _propose_delete/_propose_merge/_propose_delta calls to
        # avoid paying process-pool startup cost on every candidate-generation
        # round. Lazily created on first use; see _get_process_pool().
        self._process_pool = None
        # Cumulative wall-clock time per candidate-generation phase, printed
        # by the caller when profile_time=True. Always accumulated (the
        # timer calls themselves are negligible cost); only the printing is
        # gated on the flag.
        self._profile_totals = defaultdict(float)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_process_pool(self):
        """Return a lazily-created, reusable ProcessPoolExecutor."""
        if self._process_pool is None:
            max_workers = os.cpu_count() or 1
            self._process_pool = ProcessPoolExecutor(max_workers=max_workers)
        return self._process_pool

    def __del__(self):
        pool = getattr(self, "_process_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    # ==============================================================
    # evaluation wrappers
    # ==============================================================
    def get_sampler(self):
        return DFASampler
    
    def check_path_accepted(self, dfa, path) -> bool:
        return check_dfa_path_accepted(dfa, path)

    def compute_agreement(self, dfa, data, labels) -> float:
        """Compute local prediction agreement for a DFA."""
        return compute_automaton_agreement(
            automaton=dfa,
            data=data,
            labels=labels,
            accept_fn=self.check_path_accepted,
        )

    def update_state_metrics(self, state, old_automaton, new_automaton, data, labels, op_name):
        """
        Update cached search statistics for a proposed DFA.

        Note
        ----
        Beam Search no longer calls this during candidate generation. Candidate
        generation should only construct DFA structures; KL-LUCB sampling is
        responsible for agreement/stat computation. This helper is kept only
        for compatibility with older code paths.
        """
        key_new = id(new_automaton)
        key_old = id(old_automaton)

        stats = compute_acceptance_stats(
            automaton=new_automaton,
            data=data,
            labels=labels,
            accept_fn=self.check_path_accepted,
        )

        state["t_nsamples"][key_new] = stats["total"]
        state["t_accepted"][key_new] = stats["n_accepted"]
        state["t_positives"][key_new] = stats["true_accept"]
        state["t_negatives"][key_new] = stats["true_reject"]
        state["t_order"][key_new] = list(state["t_order"].get(key_old, []))
        state["t_order"][key_new].append((op_name, "updated"))

    def is_valid_dfa(self, dfa) -> bool:
        return is_valid_dfa(dfa)

    def serialize_automaton(self, dfa) -> int:
        return serialize_dfa(dfa)


    # ==============================================================
    # DFA history management and utilities
    # ==============================================================
    def add_to_history(self, all_history: list, seen_ids: set, dfa, 
                       training_agreement: float, validation_agreement: float = None,
                       use_automata_key: bool = False) -> None:
        """
        Add a valid, not-yet-seen DFA to history (deduplication).
        
        Parameters
        ----------
        all_history : list
            History list to append to
        seen_ids : set
            Set of already-seen DFA signatures (by automaton structure)
        dfa : Dfa
            DFA to add
        training_agreement : float
            Training agreement of the DFA
        validation_agreement : float, optional
            Validation agreement (default: None)
        use_automata_key : bool
            If True, use 'automata' key in dict (for PSO); 
            if False, use 'dfa' key (for search_baselines)
        """
        try:
            dfa_sig = serialize_dfa(dfa)
        except Exception:
            # Fallback for unexpected/invalid objects to keep history robust.
            dfa_sig = id(dfa)

        if dfa_sig in seen_ids:
            return
        if not self.is_valid_dfa(dfa):
            return
        
        seen_ids.add(dfa_sig)
        
        # Use appropriate key based on caller preference
        dfa_key = 'automata' if use_automata_key else 'dfa'
        record = {
            dfa_key: dfa,
            'training_agreement': training_agreement,
            'validation_agreement': validation_agreement,
            'states': len(dfa.states),
        }
        all_history.append(record)

    
    def create_init_automata(self, data_type, positive_samples, negative_samples):
        """Create initial DFA using RPNI algorithm"""
        from aalpy.learning_algs import run_RPNI
        init_passive_data = []
        
        
        for i in positive_samples:
            init_passive_data.append([tuple(i), True])
        for i in negative_samples:
            init_passive_data.append([tuple(i), False])
        dfa = run_RPNI(init_passive_data, automaton_type='dfa', print_info=False)
        return dfa
    

    # ==============================================================
    # Refinement operators
    # ==============================================================
    def _propose_single_neighbor(self, dfa, state, data, labels, seen_signatures, max_attempts: int = 1):
        """
        Generate a single random neighbor DFA by:
        1. Randomly choose one operation (DELETE/MERGE/DELTA)
        2. Randomly select indices to operate on
        3. Call the corresponding _propose_*_single method (which has internal retry logic)
        
        Note: Each operation function has its own internal max_attempts loop for robustness,
        so we keep max_attempts=1 here (no retries needed at this level).
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
        max_attempts : int
            Max retries if operation fails (default 1, since operations retry internally)
            
        Returns
        -------
        Dfa
            A single new DFA, or original DFA if mutation fails
        """
        import gc
        
        # Select a random operation to try
        operation = random.choice(['DELETE', 'MERGE', 'DELTA'])
        self.last_proposed_operation = operation
        
        try:
            new_dfa = None
            
            if operation == 'DELETE':
                # Pick a random state index and try to delete it
                if not dfa.states:
                    gc.collect()
                    return dfa
                target_idx = random.randint(0, len(dfa.states) - 1)
                new_dfa = self._propose_delete_single(dfa, target_idx, data, labels, seen_signatures, max_attempts=10)
                
            elif operation == 'MERGE':
                # Pick two random state indices and try to merge them
                if len(dfa.states) < 2:
                    gc.collect()
                    return dfa
                state1_idx = random.randint(0, len(dfa.states) - 1)
                state2_idx = random.randint(0, len(dfa.states) - 1)
                new_dfa = self._propose_merge_single(dfa, state1_idx, state2_idx, data, labels, seen_signatures, max_attempts=10)
                
            elif operation == 'DELTA':
                # Pick a random source state and try to add/modify a transition
                if not dfa.states:
                    gc.collect()
                    return dfa
                source_idx = random.randint(0, len(dfa.states) - 1)
                transition_idx = random.randint(0, len(dfa.states) - 1)
                new_dfa = self._propose_delta_single(dfa, source_idx, transition_idx, data, labels, seen_signatures, max_attempts=10)
            
            # Operation functions always return a DFA (modified or original as fallback)
            gc.collect()
            return new_dfa if new_dfa is not None else dfa
            
        except Exception:
            self.last_proposed_operation = f"{operation}_FAILED"
            gc.collect()
            return dfa



    def propose_multiple_neighbors(
        self,
        dfa,
        state,
        data,
        labels,
        n_candidates: int,
        seen_signatures=None,
        max_attempts_per_candidate: int = 3,
    ):
        """
        Generate a small neighborhood of candidate DFAs from the same parent DFA.

        This is used by SA and PSO baselines when one search round should evaluate
        multiple candidate DFAs instead of only one.  Each candidate is produced by
        one random DELETE / MERGE / DELTA move.  Invalid or duplicate candidates are
        skipped when possible.
        """
        n_candidates = max(1, int(n_candidates))
        seen_signatures = seen_signatures if seen_signatures is not None else set()
        local_seen = set()
        candidates = []
        local_ops = []
        attempts = 0
        max_total_attempts = max(n_candidates * max(1, int(max_attempts_per_candidate)), n_candidates)

        while len(candidates) < n_candidates and attempts < max_total_attempts:
            attempts += 1
            candidate = self._propose_single_neighbor(
                dfa,
                state,
                data,
                labels,
                seen_signatures,
                max_attempts=max_attempts_per_candidate,
            )
            op_name = getattr(self, "last_proposed_operation", "UNKNOWN") or "UNKNOWN"
            if candidate is None:
                continue

            try:
                sig = serialize_dfa(candidate)
            except Exception:
                sig = id(candidate)

            if sig in local_seen:
                continue
            if not self.is_valid_dfa(candidate):
                continue

            local_seen.add(sig)
            candidates.append(candidate)
            local_ops.append(op_name)

        if not candidates:
            candidates.append(dfa.copy() if hasattr(dfa, "copy") else copy.deepcopy(dfa))
            local_ops.append("COPY")

        self.last_proposed_ops = local_ops
        return candidates


    def _propose_delete_single(self, dfa, target_state_idx: int, data, labels, seen_signatures, max_attempts: int = 10):
        """
        Propose a SINGLE new DFA by deleting a state.
        Tries up to max_attempts times with random state selections.
        Returns first successful deletion or original DFA if all attempts fail.
        """
        import gc
        max_attempts = max(1, int(max_attempts))
        
        for attempt in range(max_attempts):
            # Pick random state index
            idx = random.randint(0, len(dfa.states) - 1) if attempt > 0 else target_state_idx
            idx = max(0, min(idx, len(dfa.states) - 1))
            target_state = dfa.states[idx]
            
            # Skip initial state and sole accepting state
            if target_state == dfa.initial_state or (target_state.is_accepting and sum(x.is_accepting for x in dfa.states) <= 1):
                continue
            
            try:
                new_dfa = dfa.copy()
                target_state_in_new = next((x for x in new_dfa.states if x.state_id == target_state.state_id), None)
                if target_state_in_new is None:
                    continue
                
                outgoing = dict(target_state_in_new.transitions)
                for st in list(new_dfa.states):
                    for sym, next_s in list(st.transitions.items()):
                        if next_s == target_state_in_new:
                            st.transitions[sym] = outgoing.get(sym, st)
                
                if target_state_in_new in new_dfa.states:
                    new_dfa.states.remove(target_state_in_new)
                
                for st in new_dfa.states:
                    for sym, nxt in list(st.transitions.items()):
                        if nxt not in new_dfa.states:
                            st.transitions[sym] = st
                
                remove_unreachable_states(new_dfa)
                
                if not any(st.is_accepting for st in new_dfa.states):
                    continue
                
                sig = serialize_dfa(new_dfa)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    gc.collect()
                    return new_dfa
            except Exception:
                continue
        
        gc.collect()
        return dfa


    def _propose_merge_single(self, dfa, state1_idx: int, state2_idx: int, data, labels, seen_signatures, max_attempts: int = 10):
        """
        Propose a SINGLE new DFA by merging two states.
        Tries up to max_attempts times with random state pair selections.
        Returns first successful merge or original DFA if all attempts fail.
        
        Used by PSO to avoid generating multiple candidates per operation slot.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state1_idx : int
            Index of first state to merge
        state2_idx : int
            Index of second state to merge
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
        max_attempts : int
            Maximum number of merge attempts (default 10)
            
        Returns
        -------
        Dfa
            The merged DFA if merge is valid, or original DFA if all attempts fail
        """
        import gc
        max_attempts = max(1, int(max_attempts))
        
        for attempt in range(max_attempts):
            # Pick random state pair
            if attempt == 0:
                idx1, idx2 = state1_idx, state2_idx
            else:
                idx1 = random.randint(0, len(dfa.states) - 1)
                idx2 = random.randint(0, len(dfa.states) - 1)
                if idx1 == idx2:
                    continue
            
            # Clamp indices to valid range
            idx1 = max(0, min(idx1, len(dfa.states) - 1))
            idx2 = max(0, min(idx2, len(dfa.states) - 1))
            
            if idx1 == idx2:
                continue
            
            s1 = dfa.states[idx1]
            s2 = dfa.states[idx2]
            
            # Cannot merge initial state
            if s1 == dfa.initial_state or s2 == dfa.initial_state:
                continue
            
            try:
                new_dfa = dfa.copy()
                s1_new = next((x for x in new_dfa.states if x.state_id == s1.state_id), None)
                s2_new = next((x for x in new_dfa.states if x.state_id == s2.state_id), None)
                
                if s1_new is None or s2_new is None:
                    continue
                
                # Merge s2 into s1: redirect all incoming transitions to s1
                for st in new_dfa.states:
                    for sym, nxt in list(st.transitions.items()):
                        if nxt == s2_new:
                            st.transitions[sym] = s1_new
                
                # Add s2's outgoing transitions to s1
                for sym, nxt in s2_new.transitions.items():
                    if nxt == s2_new:
                        s1_new.transitions[sym] = s1_new
                    else:
                        s1_new.transitions[sym] = nxt
                
                # Merge accepting status based on majority label
                # Compute state_label_dist similar to collect_merge_pairs_simple
                state_label_counts_s1 = defaultdict(int)
                state_label_counts_s2 = defaultdict(int)
                for seq, y in zip(data, labels):
                    cur = dfa.initial_state
                    for sym in seq:
                        if sym not in cur.transitions:
                            break
                        cur = cur.transitions[sym]
                        if cur.state_id == s1.state_id:
                            state_label_counts_s1[y] += 1
                        if cur.state_id == s2.state_id:
                            state_label_counts_s2[y] += 1
                
                s1_majority = max(state_label_counts_s1, key=state_label_counts_s1.get) if state_label_counts_s1 else 0
                s2_majority = max(state_label_counts_s2, key=state_label_counts_s2.get) if state_label_counts_s2 else 0
                merged_majority = max([s1_majority, s2_majority], default=0)
                s1_new.is_accepting = (merged_majority == 1)
                
                # Remove s2
                if s2_new in new_dfa.states:
                    new_dfa.states.remove(s2_new)
                
                # Remove unreachable states
                remove_unreachable_states(new_dfa)
                
                # Check if there are still accepting states after merge
                if not any(st.is_accepting for st in new_dfa.states):
                    continue
                
                # Check for duplicates
                sig = serialize_dfa(new_dfa)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    gc.collect()
                    return new_dfa
            except Exception:
                continue
        
        # All attempts failed, return original DFA as fallback
        gc.collect()
        return dfa


    def _propose_delta_single(self, dfa, source_idx: int, transition_idx: int, data, labels, seen_signatures, max_attempts: int = 10):
        """
        Propose a SINGLE new DFA by rewiring one existing transition.

        This variant is intentionally lightweight: it does not run CXP analysis.
        Instead, it picks an existing edge from the DFA, then tries to redirect it
        to a different target state. This keeps PSO / single-neighbor search cheap
        and distinct from the beam-search DELTA path.

        Uses source_idx / transition_idx as deterministic hints for the first
        attempt, then falls back to random edge / target selection.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        source_idx : int
            Index of source state
        transition_idx : int
            Index into the alphabet to select a symbol
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
        max_attempts : int
            Maximum number of transition attempts (default 10)
            
        Returns
        -------
        Dfa
            The modified DFA if transition addition is valid, or original DFA if all attempts fail
        """
        max_attempts = max(1, int(max_attempts))

        if not getattr(dfa, "states", None):
            return dfa

        edge_pool = []
        for state in dfa.states:
            for symbol, target_state in state.transitions.items():
                edge_pool.append((state, symbol, target_state))

        if not edge_pool:
            return dfa

        for attempt in range(max_attempts):
            if attempt == 0:
                source_state = dfa.states[max(0, min(source_idx, len(dfa.states) - 1))]
                outgoing = list(source_state.transitions.items())
                if outgoing:
                    symbol = outgoing[max(0, min(transition_idx, len(outgoing) - 1))][0]
                else:
                    source_state, symbol, _ = random.choice(edge_pool)
            else:
                source_state, symbol, _ = random.choice(edge_pool)

            if symbol not in source_state.transitions:
                continue

            old_target = source_state.transitions[symbol]
            candidate_targets = [s for s in dfa.states if s.state_id != old_target.state_id]
            if not candidate_targets:
                continue

            if attempt == 0:
                ordered_targets = sorted(candidate_targets, key=lambda s: s.state_id)
                target_state = ordered_targets[max(0, min(transition_idx, len(ordered_targets) - 1))]
                target_choices = [target_state] + [s for s in ordered_targets if s.state_id != target_state.state_id]
            else:
                target_choices = candidate_targets[:]
                random.shuffle(target_choices)

            for target_state in target_choices:
                try:
                    new_dfa = dfa.copy()
                    src_new = next((x for x in new_dfa.states if x.state_id == source_state.state_id), None)
                    target_new = next((x for x in new_dfa.states if x.state_id == target_state.state_id), None)

                    if src_new is None or target_new is None:
                        continue

                    src_new.transitions[symbol] = target_new
                    remove_unreachable_states(new_dfa)

                    if not any(st.is_accepting for st in new_dfa.states):
                        continue

                    sig = serialize_dfa(new_dfa)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        gc.collect()
                        return new_dfa
                except Exception:
                    continue

        gc.collect()
        return dfa


    def _propose_delete(self, dfa, state, data, labels, seen_signatures, beam_size, sig_cache=None):
        """
        Propose new DFAs by deleting states from the given DFA.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new DFAs created by deletion
        """
        import gc

        states = list(dfa.states)
        if len(states) <= 1:
            return []

        # Mirror the validity check _delete_candidate_from_state performs
        # after copying, so we skip pickling/copying the DFA for states that
        # are always rejected (the initial state, or the sole remaining
        # accepting state). Does not change which candidates are produced,
        # since these tasks would return None regardless.
        num_accepting = sum(s.is_accepting for s in states)
        deletable_state_ids = [
            s.state_id for s in states
            if s is not dfa.initial_state and not (s.is_accepting and num_accepting <= 1)
        ]
        if not deletable_state_ids:
            return []

        tasks = [(dfa, state_id) for state_id in deletable_state_ids]
        try:
            candidates = list(self._get_process_pool().map(_delete_candidate_from_state, tasks))
        except Exception:
            self._process_pool = None
            candidates = [_delete_candidate_from_state(task) for task in tasks]

        new_dfas = []
        for new_dfa in candidates:
            if new_dfa is None:
                continue
            try:
                sig = serialize_dfa(new_dfa)
            except Exception:
                sig = id(new_dfa)
            if sig_cache is not None:
                sig_cache[id(new_dfa)] = sig
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            new_dfas.append(new_dfa)

        gc.collect()
        return new_dfas


    def collect_merge_pairs_simple(self, dfa, data, labels, max_pairs=20):
        """
        Compute the label distribution for each state based on the training data.
        Then score pairs of states based on how similar their label distributions are, prioritizing pairs with the same main label.
        main label: the label with the highest count in the distribution
        
        Returns
        -------
        tuple
            (merge_pairs, state_label_dist) where state_label_dist is a dict of state_id -> {label: count}
        """
        from collections import defaultdict
        
        #  label distribution for each state
        state_label_dist = defaultdict(lambda: defaultdict(int))
        for seq, y in zip(data, labels):
            cur = dfa.initial_state
            path_states = [cur]  # Include initial state in the path
            for sym in seq:
                if sym not in cur.transitions:
                    break
                cur = cur.transitions[sym]
                path_states.append(cur)
            
            # Record label on all states in the path
            for state in path_states:
                state_label_dist[state.state_id][y] += 1
        
        # compute the main label for each state
        def main_label(state_id):
            dist = state_label_dist[state_id]
            return max(dist, key=dist.get) if dist else None
        
        # select pairs of states with the same main label
        pair_scores = []
        for s1, s2 in itertools.combinations(dfa.states, 2):
            # if s1 == dfa.initial_state or s2 == dfa.initial_state:
            #     continue
            if s1.is_accepting != s2.is_accepting:
                continue
            
            # prioritize pairs with the same main label
            if main_label(s1.state_id) == main_label(s2.state_id):
                pair_scores.append((1.0, s1, s2))
            else:
                dist1 = state_label_dist[s1.state_id]
                dist2 = state_label_dist[s2.state_id]
                # Jaccard similarity of label distributions
                all_labels = set(dist1.keys()) | set(dist2.keys())
                inter = sum(min(dist1.get(y,0), dist2.get(y,0)) for y in all_labels)
                union = sum(max(dist1.get(y,0), dist2.get(y,0)) for y in all_labels)
                sim = inter / union if union > 0 else 0
                if sim > 0:
                    pair_scores.append((sim, s1, s2))
        
        pair_scores.sort(reverse=True, key=lambda x: x[0])
        
        merge_pairs = [(s1, s2) for _, s1, s2 in pair_scores[:max_pairs]]
        return merge_pairs, state_label_dist
    

    def _propose_merge(self, dfa, state, data, labels, seen_signatures, beam_size, sig_cache=None):
        """
        Propose new DFAs by merging pairs of states in the given DFA.
        Uses intelligent scoring to prioritize high-impact merges.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new DFAs created by merging states
        """
        import gc

        new_dfas = []

        # Floors at the previous hard-coded 20 so beam_size=1 (the default in
        # every existing experiment config) reproduces identical candidates;
        # larger beam_size scales up merge-pair diversity.
        max_pairs = max(20, int(beam_size))
        feasible_pairs, state_label_dist = self.collect_merge_pairs_simple(dfa, data, labels, max_pairs=max_pairs)
        if not feasible_pairs:
            return new_dfas

        majority_labels = {
            state_id: (max(dist, key=dist.get) if dist else 0)
            for state_id, dist in state_label_dist.items()
        }

        # Mirror the validity check _merge_candidate_from_pair performs after
        # copying (neither state may be the initial state), so we skip
        # pickling/copying the DFA for pairs that are always rejected. The
        # top-20 ranking above is unchanged; this only drops guaranteed-None
        # tasks from the already-selected pair list.
        mergeable_pairs = [
            (s1, s2) for s1, s2 in feasible_pairs
            if s1 is not dfa.initial_state and s2 is not dfa.initial_state
        ]
        if not mergeable_pairs:
            return new_dfas

        tasks = [(dfa, s1.state_id, s2.state_id, majority_labels) for s1, s2 in mergeable_pairs]
        try:
            candidates = list(self._get_process_pool().map(_merge_candidate_from_pair, tasks))
        except Exception:
            self._process_pool = None
            candidates = [_merge_candidate_from_pair(task) for task in tasks]

        for new_dfa in candidates:
            if new_dfa is None:
                continue
            try:
                sig = serialize_dfa(new_dfa)
            except Exception:
                sig = id(new_dfa)
            if sig_cache is not None:
                sig_cache[id(new_dfa)] = sig
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            new_dfas.append(new_dfa)

        gc.collect()
        return new_dfas


    def _trace_path(self, dfa, path: list) -> Tuple[DfaState, bool]:
        """
        trace a path through the DFA, returning the final state and whether it was fully traced
        
        Parameters
        ----------
        dfa : Dfa
        path : list
            
        Returns
        -------
        (final_state, fully_traced) : Tuple[DfaState, bool]
            final_state: trace to the final state
            fully_traced: whether the path was fully traced (True) or stopped early (False)
        """
        current = dfa.initial_state
        for symbol in path:
            if symbol not in current.transitions:
                return current, False
            current = current.transitions[symbol]
        return current, True


    def _extract_path_edges(self, dfa, path: list) -> Tuple[List[Tuple[str, str, str]], bool]:
        """
        extract edges from a path through the DFA, returning a list of (src_state_id, symbol, tgt_state_id) and whether the path was fully traced
        
        Parameters
        ----------
        dfa : Dfa
        path : list
            
        Returns
        -------
        (edges, fully_traced) : Tuple[List[Tuple], bool]
            edges: list of (src_state_id, symbol, tgt_state_id) tuples
            fully_traced: whether the path was fully traced (True) or stopped early (False)
        """
        edges = []
        current = dfa.initial_state
        for sym in path:
            next_state = current.transitions.get(sym)
            if next_state is None:
                return edges, False
            edges.append((current.state_id, sym, next_state.state_id))
            current = next_state
        return edges, True

   
    def _aggregate_cxp_analysis(
        self,
        dfa,
        false_reject_paths,
        false_accept_paths,
        alphabet_map,
        mata_path,
        all_data,
        all_labels,
        all_accepts,
        batch_size=200,
        top_k=2,
    ):
        """
        Batch CXP analysis for DELTA with separate ranking for:
        1. false reject paths (label=1, DFA rejects)
        2. false accept paths (label=0, DFA accepts)

        Returns
        -------
        dict
            {
                "fr_rewire_edges": [(src_state_id, symbol), ...],
                "fa_rewire_edges": [(src_state_id, symbol), ...],
                "missing_edges":   [(src_state_id, missing_symbol), ...],
            }

        Scoring
        -------
        score(edge) = blamed_freq(edge) / (correct_usage(edge) + 1)

        - FR score is computed only from false-reject shortest CXPs
        - FA score is computed only from false-accept shortest CXPs
        - missing_edges are only collected from partially traced false-reject paths
        """

        def _sample_paths(paths, max_n):
            if len(paths) > max_n:
                return random.sample(paths, max_n)
            return paths

        def _collect_cxp_records(paths, collect_missing=False, tag="FR"):
            """
            For a set of misclassified paths:
            - fully traced paths -> compute CXP and keep each path's shortest CXP(s)
            - partially traced paths:
                * if collect_missing=True, record first missing transition
                * otherwise ignore for CXP

            Each path's CXP computation (including the ExplainLanguage /
            SAT-solver call in explain_word) is independent of every other
            path, so this is farmed out to the process pool via the
            module-level _collect_cxp_records_for_path worker, which performs
            the identical per-path logic. Same inputs -> same cxp_records /
            missing_transition_records, just computed concurrently instead of
            in a Python for-loop.

            Each task is bounded by _CXP_EXPLAIN_TIMEOUT_SECONDS: a path whose
            explain_word() call runs longer than that contributes no CXP
            record for that path (same effect as the existing "explain_word
            raised / returned malformed data" fallback below), instead of
            letting one pathological path stall the whole DELTA round.
            """
            if not paths:
                return [], []

            tasks = [(dfa, path, alphabet_map, mata_path, collect_missing) for path in paths]
            cxp_records = []
            missing_transition_records = []

            try:
                pool = self._get_process_pool()
                futures = [pool.submit(_collect_cxp_records_for_path, task) for task in tasks]
            except Exception:
                self._process_pool = None
                results = [_collect_cxp_records_for_path(task) for task in tasks]
                for path_cxp_records, path_missing_records in results:
                    cxp_records.extend(path_cxp_records)
                    missing_transition_records.extend(path_missing_records)
                return cxp_records, missing_transition_records

            had_timeout = False
            for fut in futures:
                try:
                    path_cxp_records, path_missing_records = fut.result(timeout=_CXP_EXPLAIN_TIMEOUT_SECONDS)
                except FutureTimeoutError:
                    had_timeout = True
                    continue
                except Exception:
                    continue
                cxp_records.extend(path_cxp_records)
                missing_transition_records.extend(path_missing_records)

            if had_timeout:
                # A worker process is still busy running the timed-out task
                # (ProcessPoolExecutor cannot forcibly kill an in-progress
                # task). Discard the pool so subsequent calls get a fresh one
                # instead of silently losing that worker slot for the rest
                # of the search.
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._process_pool = None

            return cxp_records, missing_transition_records

        def _count_blamed_edges(cxp_records):
            edge_freq = defaultdict(int)
            for rec in cxp_records:
                blamed_edges_in_this_cxp = set()
                for pos in rec["cxp"]:
                    if pos < 0 or pos >= len(rec["path_edges"]):
                        continue
                    src_state_id, sym, _ = rec["path_edges"][pos]
                    blamed_edges_in_this_cxp.add((src_state_id, sym))

                for edge in blamed_edges_in_this_cxp:
                    edge_freq[edge] += 1
            return edge_freq

        def _rank_edges(blamed_edge_freq, correct_usage):
            ranked_items = []
            for edge, bfreq in blamed_edge_freq.items():
                cusage = correct_usage.get(edge, 0)
                score = bfreq / (cusage + 1)
                ranked_items.append((score, bfreq, cusage, edge))

            ranked_items.sort(
                key=lambda item: (-item[0], -item[1], item[2], str(item[3][0]), str(item[3][1]))
            )
            return ranked_items

        # Sample FR / FA paths separately
        false_reject_paths = _sample_paths(false_reject_paths, batch_size)
        false_accept_paths = _sample_paths(false_accept_paths, batch_size)


        # Collect CXP records separately
        #    - FR: compute CXPs + collect missing transitions
        #    - FA: compute CXPs only
        fr_cxp_records, missing_transition_records = _collect_cxp_records(
            false_reject_paths,
            collect_missing=True,
            tag="FR",
        )
        fa_cxp_records, _ = _collect_cxp_records(
            false_accept_paths,
            collect_missing=False,
            tag="FA",
        )

        # Count blamed frequencies separately
        fr_blamed_edge_freq = _count_blamed_edges(fr_cxp_records)
        fa_blamed_edge_freq = _count_blamed_edges(fa_cxp_records)

        # correct usage from correctly classified paths
        # (accepts already computed once by the caller for this dfa/data pair)
        accepts = all_accepts
        all_labels = np.asarray(all_labels)
        correct_indices = np.where(
            ((all_labels == 1) & (accepts == True)) |
            ((all_labels == 0) & (accepts == False))
        )[0]

        correct_usage = defaultdict(int)

        for idx in correct_indices:
            path = all_data[idx]
            path_edges, _ = self._extract_path_edges(dfa, path)

            seen_edges_in_path = set()
            for src_state_id, sym, _ in path_edges:
                seen_edges_in_path.add((src_state_id, sym))

            for edge in seen_edges_in_path:
                correct_usage[edge] += 1

        # Rank FR / FA edges separately + missing edges
        # Then find the SINGLE best edge across all types
        ranked_fr_items = _rank_edges(fr_blamed_edge_freq, correct_usage)
        ranked_fa_items = _rank_edges(fa_blamed_edge_freq, correct_usage)

        # Rank missing transitions (from FR partial traces)
        missing_edge_freq = defaultdict(int)
        for missing_edge in missing_transition_records:
            missing_edge_freq[missing_edge] += 1

        ranked_missing_items = sorted(
            missing_edge_freq.items(),
            key=lambda item: (-item[1], str(item[0][0]), str(item[0][1]))
        )
        
        # Combine all edges with unified scoring
        # Form: (score, bfreq, cusage, edge, edge_type)
        all_scored_edges = []
        
        # Add FR rewire edges
        for score, bfreq, cusage, edge in ranked_fr_items:
            all_scored_edges.append((score, bfreq, cusage, edge, "FR_rewire"))
        
        # Add FA rewire edges
        for score, bfreq, cusage, edge in ranked_fa_items:
            all_scored_edges.append((score, bfreq, cusage, edge, "FA_rewire"))
        
        # Add missing edges (score = frequency)
        for missing_edge, freq in ranked_missing_items:
            all_scored_edges.append((float(freq), freq, 0, missing_edge, "missing"))
        
        # Sort by score (descending)
        all_scored_edges.sort(key=lambda x: (-x[0], -x[1], x[2], str(x[3][0]), str(x[3][1])))
        
        # Return the top-k best edges (k=top_k, floored at the previous
        # hard-coded 1 so beam_size=1 reproduces the old single-edge behavior).
        fr_rewire_edges = []
        fa_rewire_edges = []
        missing_edges = []
        n_top = max(1, int(top_k))
        for score, bfreq, cusage, best_edge, edge_type in all_scored_edges[:n_top]:
            if edge_type == "FR_rewire":
                fr_rewire_edges.append(best_edge)
            elif edge_type == "FA_rewire":
                fa_rewire_edges.append(best_edge)
            else:  # missing
                missing_edges.append(best_edge)

        return {
            "fr_rewire_edges": fr_rewire_edges,
            "fa_rewire_edges": fa_rewire_edges,
            "missing_edges": missing_edges,
        }


    def _propose_delta(self, dfa, state, data, labels, seen_signatures, batch_size=32, top_k=1, sig_cache=None):
        """
        Propose new DFAs by separately repairing:

        1. false rejects (label=1 but DFA rejects)
        - rewire FR-blamed edges toward accepting-reachable states
        - add missing transitions toward accepting-reachable states

        2. false accepts (label=0 but DFA accepts)
        - rewire FA-blamed edges toward non-accepting-oriented states
            (prefer states that cannot reach accepting)
        """

        new_dfas = [dfa]  # always include original DFA

        accepts = np.array([self.check_path_accepted(dfa, p) for p in data])

        false_reject_indices = np.where((labels == 1) & (accepts == False))[0]
        false_accept_indices = np.where((labels == 0) & (accepts == True))[0]

        if len(false_reject_indices) == 0 and len(false_accept_indices) == 0:
            return new_dfas

        false_reject_paths = [data[i] for i in false_reject_indices.tolist()]
        false_accept_paths = [data[i] for i in false_accept_indices.tolist()]

        # export DFA to mata for CXP
        _, alphabet_map, _, _ = dfa_to_mata(dfa, self.mata_path)

        cxp_start = time.perf_counter()
        delta_result = self._aggregate_cxp_analysis(
            dfa,
            false_reject_paths,
            false_accept_paths,
            alphabet_map,
            self.mata_path,
            all_data=data,
            all_labels=labels,
            all_accepts=accepts,
            batch_size=batch_size,
            top_k=top_k,
        )
        self._profile_totals["delta_cxp_analysis"] += time.perf_counter() - cxp_start

        if not delta_result:
            return new_dfas
      
        # Build state mapping
        orig_state_by_id = {st.state_id: st for st in dfa.states}

        # Collect all blamed edges and try connecting to all states
        all_blamed_edges = []
        all_blamed_edges.extend(delta_result.get("fr_rewire_edges", []))
        all_blamed_edges.extend(delta_result.get("fa_rewire_edges", []))
        all_blamed_edges.extend(delta_result.get("missing_edges", []))
        
        if all_blamed_edges:
            tasks = []
            for src_state_id, symbol in all_blamed_edges:
                src_state = orig_state_by_id.get(src_state_id)
                if src_state is None:
                    continue

                old_target = src_state.transitions.get(symbol)
                target_candidates = [st for st in dfa.states if (old_target is None or st.state_id != old_target.state_id)]
                for target_state in target_candidates:
                    tasks.append((dfa, src_state_id, symbol, target_state.state_id))

            if tasks:
                try:
                    candidates = list(self._get_process_pool().map(_delta_candidate_from_edge, tasks))
                except Exception:
                    self._process_pool = None
                    candidates = [_delta_candidate_from_edge(task) for task in tasks]

                for new_dfa in candidates:
                    if new_dfa is None:
                        continue
                    try:
                        sig = serialize_dfa(new_dfa)
                    except Exception:
                        sig = id(new_dfa)
                    if sig_cache is not None:
                        sig_cache[id(new_dfa)] = sig
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)
                    new_dfas.append(new_dfa)

        return new_dfas


    def propose_automata(self, dfas, state, iteration, previous_best: list, output_dir: str, beam_size: int = 10, batch_size: int = 32):
        """
        Propose new DFA candidates by expanding existing DFAs.

        This function only constructs candidate DFA structures. It intentionally
        does not compute agreement/statistics for each candidate; candidate
        evaluation is delayed to KL-LUCB in AutomataBeamSearch.
        
        This method coordinates three strategies:
        - DELETE: Remove states from DFA (even iterations)
        - MERGE: Combine pairs of states (even iterations)
        - DELTA: Modify transitions based on CXP analysis (odd iterations)
        
        Parameters
        ----------
        dfas : list
            List of current DFAs
        state : dict
            State dictionary for tracking metrics
        sample_fcn : object
            Sampling function with feature values
        iteration : int
            Current iteration number
        previous_best : list
            List of best DFAs from previous iteration
        data_type : str
            Type of data ('Tabular', 'Text', etc.)
            
        Returns
        -------
        list
            List of proposed new DFAs
        """
        # Store output directory for mata file generation
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.mata_path = os.path.join(self.output_dir, "dfa_explicit.mata")
        
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]

        # Initialize metrics for first iteration
        if iteration == 0:
            for dfa in dfas:
                accepts = np.array([self.check_path_accepted(dfa, p) for p in data])
                true_accept = np.sum((labels == 1) & (accepts == True))
                false_reject = np.sum((labels == 0) & (accepts == False))
                
                dfa_id = id(dfa)
                state['t_nsamples'][dfa_id] = float(len(data))
                state['t_accepted'][dfa_id] = float(np.sum(accepts))
                state['t_positives'][dfa_id] = float(true_accept)
                state['t_negatives'][dfa_id] = float(false_reject)
                state['t_order'][dfa_id].append(dfa_id)

                print("--------------------------------------------")
            
            return dfas

        seen_signatures = set()
        new_dfas = []
        # Populated by _propose_delete/_propose_merge/_propose_delta as they
        # compute each candidate's signature for their own seen_signatures
        # dedup, so the final unique_dfas pass below doesn't need to
        # re-serialize the same DFA a second time.
        sig_cache: dict = {}

        for dfa in previous_best:
            if iteration % 2 == 0:
                # Even iterations: DELETE and MERGE
                t0 = time.perf_counter()
                delete_candidates = self._propose_delete(dfa, state, data, labels, seen_signatures, beam_size, sig_cache=sig_cache)
                self._profile_totals["propose_delete"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                merge_candidates = self._propose_merge(dfa, state, data, labels, seen_signatures, beam_size, sig_cache=sig_cache)
                self._profile_totals["propose_merge"] += time.perf_counter() - t0
                new_dfas.extend(delete_candidates)
                new_dfas.extend(merge_candidates)
            else:
                # Odd iterations: DELTA
                t0 = time.perf_counter()
                delta_candidates = self._propose_delta(dfa, state, data, labels, seen_signatures, batch_size, top_k=max(1, int(beam_size)), sig_cache=sig_cache)
                self._profile_totals["propose_delta_total"] += time.perf_counter() - t0
                new_dfas.extend(delta_candidates)

        unique_dfas = []
        seen = set()
        for dfa in new_dfas:
            sig = sig_cache.get(id(dfa))
            if sig is None:
                sig = serialize_dfa(dfa)
            if sig not in seen:
                seen.add(sig)
                unique_dfas.append(dfa)
            else:
                del dfa
        # Run GC once per candidate-generation round instead of once per candidate.
        gc.collect()
        return unique_dfas


__all__ = [
    'DFALearner',
]
