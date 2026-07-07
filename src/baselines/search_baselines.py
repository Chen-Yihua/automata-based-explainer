"""
DFA Search Baselines: SA, GA, PSO.

This module contains all non-beam baseline methods used in the experiments.
PSOAutomataOptimizer is merged here intentionally so there is no separate
learner/pso_optimizer.py dependency to keep synchronized.
"""

from __future__ import annotations
import copy
import math
import gc
import os
import numpy as np
from collections import defaultdict, Counter
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from deap import base, creator, tools

from simanneal import Annealer
from copy import deepcopy

try:
    from pyswarms.single.global_best import GlobalBestPSO
    PSO_AVAILABLE = True
except ImportError:
    PSO_AVAILABLE = False
    print("[WARNING] pyswarms not found. PSO baseline will not be available.")
from automaton.dfa_utils import remove_unreachable_states, dfa_to_graphviz
from automaton.metrics import compute_automaton_agreement

# -----------------------------------------------------------------------
# Module-level reference to the learner instance (mirrors anchor_base.py)
# -----------------------------------------------------------------------
_AUTO_INSTANCE = None  # set by _common_init() or build_shared_init()


# -----------------------------------------------------------------------
# Candidate logging helpers
# -----------------------------------------------------------------------
def _dfa_state_count(dfa) -> int:
    """Return the number of states in a DFA-like object."""
    return len(dfa.states) if dfa is not None and hasattr(dfa, "states") else 0


def _candidate_signature(dfa) -> str:
    """Return a short stable-ish signature for readable candidate logs."""
    try:
        if _AUTO_INSTANCE is not None and hasattr(_AUTO_INSTANCE, "serialize_automaton"):
            sig = _AUTO_INSTANCE.serialize_automaton(dfa)
        else:
            sig = id(dfa)
    except Exception:
        sig = id(dfa)
    return str(sig)[-8:]


def _candidate_loss(agreement: float, states: int, initial_states: int) -> float:
    """Shared baseline objective used only for logging/comparison."""
    initial_states = max(1, int(initial_states))
    return -float(agreement) + (float(states) / initial_states)


def _candidate_op_hint(parent_states: int, candidate_states: int, ops=None) -> str:
    """Best-effort operation hint for logs."""
    if ops:
        if isinstance(ops, (list, tuple)):
            return "+".join(str(x) for x in ops) if ops else "N/A"
        return str(ops)
    if candidate_states < parent_states:
        return "DELETE/MERGE"
    if candidate_states == parent_states:
        return "DELTA/COPY"
    return "EXPAND"


def _print_candidate_log(method: str, round_label: str, rows: list, selected_idx=None) -> None:
    """Candidate-level logging disabled to reduce I/O overhead."""
    return

def _normalise_operator_name(op) -> str:
    """Normalize noisy op strings to DELETE / MERGE / DELTA / COPY / UNKNOWN."""
    text = str(op or "UNKNOWN").upper()
    if "DELETE" in text:
        return "DELETE"
    if "MERGE" in text:
        return "MERGE"
    if "DELTA" in text:
        return "DELTA"
    if "COPY" in text or "FALLBACK" in text:
        return "COPY"
    return "UNKNOWN"


def _record_operator(counter, op) -> str:
    """Record one operator use and return the normalized operator label."""
    label = _normalise_operator_name(op)
    if counter is not None:
        counter[label] += 1
    return label


def _print_operator_summary(method: str, counter, evaluations_used: int = None, max_evaluations: int = None) -> None:
    """Print DELETE / MERGE / DELTA usage ratios for one baseline."""
    if not counter:
        print(f"[{method}] Operator summary: no operator statistics collected.")
        return

    total = sum(counter.values())
    budget = ""
    if evaluations_used is not None and max_evaluations is not None:
        budget = f" | evaluations={evaluations_used}/{max_evaluations}"
    print(f"[{method}] Operator summary{budget}")
    for op in ("DELETE", "MERGE", "DELTA", "COPY", "UNKNOWN"):
        count = int(counter.get(op, 0))
        if count == 0 and op not in ("DELETE", "MERGE", "DELTA"):
            continue
        ratio = (count / total) if total else 0.0
        print(f"  {op:7s}: {count:5d} ({ratio:6.2%})")



# ======================================================================
# PSO optimizer (merged from learner/pso_optimizer.py)
# ======================================================================

class PSOAutomataOptimizer:
    """
    PSO-guided iterative DFA refinement.

    Difference from the older position-to-DFA decoder:
    - Each particle keeps its own current DFA.
    - In each iteration, candidates are generated from that particle's current DFA.
    - The particle position is only an operation-preference vector over
      DELETE / MERGE / DELTA, not a full DFA encoding.
    - This makes the baseline closer to Beam Search: a population of DFAs is
      iteratively expanded and improved using the same refinement operators.
    """

    OP_NAMES = ("DELETE", "MERGE", "DELTA")

    def __init__(self,
                 initial_dfa,
                 threshold: float = 0.8,
                 data: List = None,
                 labels: np.ndarray = None,
                 learner=None,
                 validation_data: List = None,
                 validation_labels: np.ndarray = None,
                 n_particles: int = 10,
                 n_iterations: int = 20,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 verbose: bool = True,
                 max_evaluations: Optional[int] = None,
                 slot_beam_size: Optional[int] = None,
                 max_ops_per_iteration: Optional[int] = None,
                 candidate_pool_size: Optional[int] = None):
        self.initial_dfa = initial_dfa
        self.initial_states = max(1, len(initial_dfa.states)) if hasattr(initial_dfa, "states") else 1
        self.threshold = threshold
        self.data = data or []
        self.labels = labels if labels is not None else np.array([])
        self.validation_data = validation_data or []
        self.validation_labels = validation_labels if validation_labels is not None else np.array([])
        self.learner = learner
        self.verbose = verbose

        # In the iterative PSO version, a particle position is an operation
        # preference vector: [delete_score, merge_score, delta_score].
        self.dimensions = 3
        self.n_particles = int(max(1, n_particles))
        self.n_iterations = int(max(1, n_iterations))
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.candidate_pool_size = max(1, int(candidate_pool_size or slot_beam_size or 5))
        self.max_ops_per_iteration = max(1, int(max_ops_per_iteration or 1))

        self.state_metrics: Dict[int, Dict] = {}
        self._init_state_metrics()
        self.seen_signatures: set = set()
        self.all_history: List[Dict] = []
        self.seen_ids: set = set()
        self.operator_stats = Counter()
        self.evaluations_count: int = 0
        self.max_evaluations: Optional[int] = max_evaluations
        self.reached_two_states: bool = False
        self.pso_iteration_count: int = 0

        self.gbest_dfa = None
        self.gbest_fitness = float("inf")
        self.gbest_agreement = 0.0
        self.gbest_val_agreement = 0.0
        self.gbest_states = float("inf")
        self.gbest_position = np.zeros(self.dimensions, dtype=float)

        self.pso_no_improve_count = 0
        self.pso_no_improve_threshold = 10
        self.pso_last_best_fitness = float("inf")

        # Particle states. Every particle starts from the same initial DFA, but
        # has a different operation-preference vector and velocity.
        self.positions = np.random.normal(0.0, 0.5, size=(self.n_particles, self.dimensions))
        self.velocities = np.zeros((self.n_particles, self.dimensions), dtype=float)
        self.current_dfas = [deepcopy(self.initial_dfa) for _ in range(self.n_particles)]
        self.current_losses = np.full(self.n_particles, float("inf"), dtype=float)
        self.current_agreements = np.zeros(self.n_particles, dtype=float)
        self.current_states = np.full(self.n_particles, self.initial_states, dtype=int)
        self.pbest_dfas = [deepcopy(self.initial_dfa) for _ in range(self.n_particles)]
        self.pbest_losses = np.full(self.n_particles, float("inf"), dtype=float)
        self.pbest_positions = self.positions.copy()

        # Count the initial DFA once as a baseline candidate.
        initial_agreement, initial_states, initial_loss = self._evaluate_and_cache(self.initial_dfa)
        initial_val_agreement = self._compute_validation_agreement(self.initial_dfa)
        self.evaluations_count += 1
        self._add_to_history(self.initial_dfa, initial_agreement, initial_val_agreement)
        self._update_gbest(self.initial_dfa, initial_agreement, initial_states, initial_loss, initial_val_agreement)

        for i in range(self.n_particles):
            self.current_losses[i] = initial_loss
            self.current_agreements[i] = initial_agreement
            self.current_states[i] = initial_states
            self.pbest_losses[i] = initial_loss

        if self.verbose:
            val_str = f", val_agreement={initial_val_agreement:.4f}" if len(self.validation_data) > 0 else ""
            print(
                f"[PSO Init] Initial DFA: {initial_states} states, "
                f"agreement={initial_agreement:.4f}{val_str}"
            )
            print("[PSO Init] Iterative DFA refinement mode: each particle keeps a current DFA.")

    def _init_state_metrics(self):
        self.state_metrics = {
            "t_nsamples": {},
            "t_positives": {},
            "t_negatives": {},
        }

    def _add_to_history(self, dfa, training_agreement: float, validation_agreement: float) -> None:
        self.learner.add_to_history(
            self.all_history,
            self.seen_ids,
            dfa,
            training_agreement,
            validation_agreement,
            use_automata_key=True,
        )

    def _compute_validation_agreement(self, dfa) -> float:
        if self.learner is None or len(self.validation_data) == 0 or len(self.validation_labels) == 0:
            return 0.0
        try:
            return compute_automaton_agreement(
                automaton=dfa,
                data=self.validation_data,
                labels=self.validation_labels,
                accept_fn=self.learner.check_path_accepted,
            )
        except Exception:
            return 0.0

    def _evaluate_and_cache(self, dfa):
        dfa_id = id(dfa)
        n_samples = len(self.data) if len(self.data) > 0 else 1

        if dfa_id in self.state_metrics["t_nsamples"]:
            true_pos = self.state_metrics["t_positives"].get(dfa_id, 0)
            true_neg = self.state_metrics["t_negatives"].get(dfa_id, 0)
            agreement = (true_pos + true_neg) / n_samples
        else:
            if self.learner and len(self.data) > 0 and len(self.labels) > 0:
                accepts = np.array([self.learner.check_path_accepted(dfa, p) for p in self.data])
                true_accept = np.sum((self.labels == 1) & (accepts == True))
                true_reject = np.sum((self.labels == 0) & (accepts == False))
            else:
                true_accept = 0
                true_reject = 0

            self.state_metrics["t_nsamples"][dfa_id] = float(n_samples)
            self.state_metrics["t_positives"][dfa_id] = float(true_accept)
            self.state_metrics["t_negatives"][dfa_id] = float(true_reject)
            agreement = (true_accept + true_reject) / n_samples

        num_states = len(dfa.states)
        loss = self._compute_loss(agreement, num_states)
        return float(agreement), int(num_states), float(loss)

    def _compute_loss(self, agreement: float, num_states: int) -> float:
        return -float(agreement) + float(num_states) / float(self.initial_states)

    def _update_gbest(self, dfa, agreement: float, num_states: int, loss: float, val_agreement: float = 0.0, *, particle_id=None, candidate_idx=None):
        if loss < self.gbest_fitness:
            self.gbest_dfa = deepcopy(dfa)
            self.gbest_fitness = float(loss)
            self.gbest_agreement = float(agreement)
            self.gbest_val_agreement = float(val_agreement)
            self.gbest_states = int(num_states)
            if particle_id is not None:
                self.gbest_position = self.positions[int(particle_id)].copy()
            val_str = f", val_agr={val_agreement:.4f}" if val_agreement > 0 else ""
            src = ""
            if particle_id is not None:
                src = f" iter={self.pso_iteration_count} particle={particle_id}"
                if candidate_idx is not None:
                    src += f" candidate={candidate_idx}"
            if self.verbose:
                print(
                    f"    [GBEST UPDATE]{src} New best: {num_states} states, "
                    f"agreement={agreement:.4f}{val_str}, loss={loss:.4f}"
                )

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        denom = float(np.sum(exp_scores))
        if denom <= 0 or not np.isfinite(denom):
            return np.ones(len(scores), dtype=float) / len(scores)
        return exp_scores / denom

    def _sample_operation(self, scores: np.ndarray) -> str:
        probs = self._softmax(scores)
        idx = int(np.random.choice(len(self.OP_NAMES), p=probs))
        return self.OP_NAMES[idx]

    def _apply_operation_to_parent(self, parent_dfa, op_type: str):
        """Apply one DELETE / MERGE / DELTA operation to a particle's current DFA."""
        if self.learner is None:
            raise RuntimeError("Learner is required for PSO operations")

        n_states = len(parent_dfa.states)
        if n_states <= 0:
            return deepcopy(parent_dfa), f"{op_type}✗EMPTY"

        try:
            if op_type == "DELETE":
                target_idx = int(np.random.randint(0, n_states))
                next_dfa = self.learner._propose_delete_single(
                    parent_dfa,
                    target_idx,
                    self.data,
                    self.labels,
                    self.seen_signatures,
                )
                descriptor = f"DELETE[s{target_idx}]"

            elif op_type == "MERGE":
                if n_states < 2:
                    return deepcopy(parent_dfa), "MERGE✗INVALID(only_1_state)"
                state1_idx = int(np.random.randint(0, n_states))
                state2_idx = int(np.random.randint(0, n_states))
                if state1_idx == state2_idx:
                    state2_idx = (state1_idx + 1) % n_states
                next_dfa = self.learner._propose_merge_single(
                    parent_dfa,
                    state1_idx,
                    state2_idx,
                    self.data,
                    self.labels,
                    self.seen_signatures,
                )
                descriptor = f"MERGE[s{state1_idx}↔s{state2_idx}]"

            elif op_type == "DELTA":
                source_idx = int(np.random.randint(0, n_states))
                target_idx = int(np.random.randint(0, n_states))
                next_dfa = self.learner._propose_delta_single(
                    parent_dfa,
                    source_idx,
                    target_idx,
                    self.data,
                    self.labels,
                    self.seen_signatures,
                )
                descriptor = f"DELTA[s{source_idx}→s{target_idx}]"
            else:
                return deepcopy(parent_dfa), "UNKNOWN✗COPY"

            if next_dfa is None:
                return deepcopy(parent_dfa), f"{descriptor}✗COPY"
            return next_dfa, f"{descriptor}✓"

        except Exception as exc:
            if self.verbose:
                print(f"  [WARN] PSO operation {op_type} failed: {str(exc)[:80]}")
            return deepcopy(parent_dfa), f"{op_type}✗COPY"

    def _generate_particle_candidates(self, particle_id: int):
        """Generate a local candidate pool from one particle's current DFA."""
        parent_dfa = self.current_dfas[particle_id]
        parent_states = len(parent_dfa.states)
        batch = []
        seen = set()

        for _ in range(self.candidate_pool_size):
            # Local jitter keeps candidates diverse while preserving the particle's
            # learned operation preference.
            op_scores = self.positions[particle_id] + np.random.normal(0.0, 0.35, size=self.dimensions)
            op_type = self._sample_operation(op_scores)

            candidate = deepcopy(parent_dfa)
            ops = []
            for _step in range(self.max_ops_per_iteration):
                candidate, descriptor = self._apply_operation_to_parent(candidate, op_type)
                ops.append(f"{op_type}:{descriptor}")
                # Re-sample the next op if multiple edits are allowed.
                if self.max_ops_per_iteration > 1:
                    op_scores = self.positions[particle_id] + np.random.normal(0.0, 0.35, size=self.dimensions)
                    op_type = self._sample_operation(op_scores)

            remove_unreachable_states(candidate)
            try:
                sig = self.learner.serialize_automaton(candidate) if hasattr(self.learner, "serialize_automaton") else id(candidate)
            except Exception:
                sig = id(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            batch.append((candidate, ops, parent_states))

        if not batch:
            batch.append((deepcopy(parent_dfa), ["fallback_current"], parent_states))
        return batch

    def _update_particle_position(self, particle_id: int):
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)
        cognitive = self.c1 * r1 * (self.pbest_positions[particle_id] - self.positions[particle_id])
        social = self.c2 * r2 * (self.gbest_position - self.positions[particle_id])
        self.velocities[particle_id] = self.w * self.velocities[particle_id] + cognitive + social
        self.positions[particle_id] = self.positions[particle_id] + self.velocities[particle_id]
        self.positions[particle_id] = np.clip(self.positions[particle_id], -5.0, 5.0)

    def objective_function(self, X: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        One iterative PSO refinement round.

        This method is kept for compatibility with older call sites, but this
        class now runs its own optimization loop in optimize().  It evaluates one
        local candidate pool per particle and updates that particle's current DFA.
        """
        n_particles = self.n_particles
        losses = np.zeros(n_particles, dtype=float)
        iteration_improved = False
        iteration_best_fitness = self.pso_last_best_fitness
        self.pso_iteration_count += 1
        iter_label = f"iter{self.pso_iteration_count}"

        for particle_id in range(n_particles):
            if self.max_evaluations is not None and self.evaluations_count >= self.max_evaluations:
                if self.verbose:
                    print(f"  [PSO-obj] Budget exhausted before particle {particle_id}: {self.evaluations_count}/{self.max_evaluations}")
                raise RuntimeError("PSO budget exhausted")

            candidate_batch = self._generate_particle_candidates(particle_id)
            best_particle_loss = float("inf")
            best_candidate_idx = None
            best_candidate_dfa = None
            best_candidate_agreement = 0.0
            best_candidate_states = 0
            candidate_rows = []

            for cand_idx, (dfa, ops, parent_states) in enumerate(candidate_batch):
                if self.max_evaluations is not None and self.evaluations_count >= self.max_evaluations:
                    break

                self.evaluations_count += 1
                agreement, num_states, loss = self._evaluate_and_cache(dfa)
                self._add_to_history(dfa, agreement, 0.0)

                op_hint = _candidate_op_hint(parent_states, num_states, ops)
                _record_operator(self.operator_stats, op_hint)

                candidate_rows.append({
                    "idx": cand_idx,
                    "sig": _candidate_signature(dfa),
                    "op": op_hint,
                    "states": num_states,
                    "agreement": agreement,
                    "loss": loss,
                })

                if loss < best_particle_loss:
                    best_particle_loss = loss
                    best_candidate_idx = cand_idx
                    best_candidate_dfa = dfa
                    best_candidate_agreement = agreement
                    best_candidate_states = num_states

                if loss < iteration_best_fitness:
                    iteration_best_fitness = loss
                    self.pso_last_best_fitness = loss
                    iteration_improved = True

            _print_candidate_log("PSO", f"{iter_label}-particle{particle_id}", candidate_rows, selected_idx=best_candidate_idx)

            if best_candidate_dfa is not None:
                # Move the particle's DFA along the best local refinement step.
                # This is what makes the method comparable to Beam Search.
                self.current_dfas[particle_id] = deepcopy(best_candidate_dfa)
                self.current_losses[particle_id] = best_particle_loss
                self.current_agreements[particle_id] = best_candidate_agreement
                self.current_states[particle_id] = best_candidate_states

                if best_particle_loss < self.pbest_losses[particle_id]:
                    self.pbest_losses[particle_id] = best_particle_loss
                    self.pbest_dfas[particle_id] = deepcopy(best_candidate_dfa)
                    self.pbest_positions[particle_id] = self.positions[particle_id].copy()

                val_agreement = 0.0
                self._update_gbest(
                    best_candidate_dfa,
                    best_candidate_agreement,
                    best_candidate_states,
                    best_particle_loss,
                    val_agreement,
                    particle_id=particle_id,
                    candidate_idx=best_candidate_idx,
                )
            else:
                best_particle_loss = self.current_losses[particle_id]

            losses[particle_id] = best_particle_loss
            self._update_particle_position(particle_id)

        if iteration_improved:
            self.pso_no_improve_count = 0
        else:
            self.pso_no_improve_count += 1
            if self.pso_no_improve_count >= self.pso_no_improve_threshold:
                if self.verbose:
                    print(f"  [PSO] Early stopping: {self.pso_no_improve_count} iterations without improvement")
                raise RuntimeError(f"PSO early stopping: no improvement for {self.pso_no_improve_threshold} iterations")

        return losses

    def optimize(self,
                 n_particles: Optional[int] = None,
                 n_iterations: Optional[int] = None,
                 save_trajectory: bool = False) -> Dict[str, Any]:
        n_particles = int(n_particles or self.n_particles)
        n_iterations = int(n_iterations or self.n_iterations)
        if n_particles != self.n_particles:
            # The optimizer is initialized with particle-specific DFA state, so
            # changing particle count after initialization is intentionally not
            # supported. Keep running with the initialized count.
            n_particles = self.n_particles

        if self.verbose:
            print("\n[PSO] Starting iterative DFA-level optimization...")
            print(f"  Particles: {self.n_particles}")
            print(f"  Candidate pool per particle: {self.candidate_pool_size}")
            print(f"  Iterations: {n_iterations}")
            print(f"  Initial states: {len(self.initial_dfa.states)}")
            print("  Particle state: current DFA is carried to the next iteration")

        trajectory = []
        stop_reason = ""
        for _ in range(n_iterations):
            try:
                losses = self.objective_function()
                trajectory.append(float(np.min(losses)) if len(losses) else float("inf"))
            except RuntimeError as exc:
                stop_reason = str(exc)
                if self.verbose:
                    print(f"[PSO] Optimization stopped: {stop_reason}")
                break

            if self.max_evaluations is not None and self.evaluations_count >= self.max_evaluations:
                stop_reason = "PSO budget exhausted"
                if self.verbose:
                    print(f"[PSO] Optimization stopped: {stop_reason}")
                break

        if self.gbest_dfa is not None:
            self.gbest_val_agreement = self._compute_validation_agreement(self.gbest_dfa)

        if self.verbose:
            print("\n[PSO] Finished optimization")
            print(f"  Evaluations: {self.evaluations_count}/{self.max_evaluations}")
            print(f"  Best agreement: {self.gbest_agreement:.4f}")
            val_str = f", val_agr={self.gbest_val_agreement:.4f}" if self.gbest_val_agreement > 0 else ""
            best_states_str = f"{int(self.gbest_states)}" if not np.isinf(self.gbest_states) else "No valid solution found"
            print(f"  Best states: {best_states_str}{val_str}")

        best_states = int(self.gbest_states) if not np.isinf(self.gbest_states) else -1
        best_loss = float(self.gbest_fitness) if not np.isinf(self.gbest_fitness) else -1.0
        success = not np.isinf(self.gbest_fitness) and self.gbest_dfa is not None
        if success:
            reason = f"PSO found solution with agreement {self.gbest_agreement:.4f} and {best_states} states"
            if stop_reason:
                reason += f" (stopped: {stop_reason})"
        else:
            reason = "PSO failed to find valid solution"
            if stop_reason:
                reason += f" (stopped: {stop_reason})"

        result = {
            "best_dfa": self.gbest_dfa,
            "best_agreement": float(self.gbest_agreement),
            "best_validation_agreement": float(self.gbest_val_agreement),
            "best_states": best_states,
            "best_loss": best_loss,
            "iterations": self.pso_iteration_count,
            "threshold": self.threshold,
            "all_history": self.all_history,
            "evaluations": int(self.evaluations_count),
            "max_evaluations": int(self.max_evaluations) if self.max_evaluations is not None else None,
            "success": success,
            "reason": reason,
            "operator_counts": dict(self.operator_stats),
        }
        if save_trajectory:
            result["trajectory"] = trajectory
        return result

# ======================================================================
# Shared initialisation object
# ======================================================================

class SharedInit(NamedTuple):
    """
    Carries the artefacts produced by a single RPNI run so that every
    search method (beam search, SA, GA, PSO) can start from the exact
    same initial DFA and the same training data.

    Fields
    ------
    initial_dfa : object       – the RPNI-generated DFA (call .copy() before use)
    learner     : DFALearner  – the learner instance (holds alphabet_map etc.)
    validation_data : list       – validation samples (from beam search)
    validation_labels : np.ndarray – validation labels (from beam search)
    training_data : list       – perturbation training samples (FIXED, shared across SA/GA/PSO)
    training_labels : np.ndarray – perturbation training labels (FIXED, shared across SA/GA/PSO)
    """
    initial_dfa: object
    learner: object
    validation_data: list
    validation_labels: np.ndarray
    training_data: list
    training_labels: np.ndarray

# ======================================================================
# Shared helpers
# ======================================================================

def _common_init(shared_init: SharedInit,
                 batch_size: int,
                 output_dir: str) -> Tuple[object, int, list, np.ndarray, dict, list, np.ndarray]:
    """
    Initialize from SharedInit object prepared by beam search.
    
    Parameters
    ----------
    shared_init : SharedInit
        Required. Contains initial_dfa, learner, validation_data, validation_labels,
        training_data, training_labels (all fixed and shared across SA/GA/PSO).
    batch_size : int
        Used for pre-allocation of state dict
    output_dir : str
        Output directory for any files
    
    Returns
    -------
    initial_dfa, initial_states (int), validation_data (list), validation_labels (ndarray), state (dict), 
    training_data (list), training_labels (ndarray)
    """
    global _AUTO_INSTANCE

    os.makedirs(output_dir, exist_ok=True)

    # Extract from SharedInit and SET GLOBAL LEARNER INSTANCE
    _AUTO_INSTANCE = shared_init.learner
    initial_dfa = shared_init.initial_dfa.copy() if hasattr(shared_init.initial_dfa, 'copy') else copy.deepcopy(shared_init.initial_dfa)
    initial_states = len(initial_dfa.states) if hasattr(initial_dfa, 'states') else 0
    validation_data = list(shared_init.validation_data)
    validation_labels = np.array(shared_init.validation_labels)
    training_data = list(shared_init.training_data)
    training_labels = np.array(shared_init.training_labels)
    
    print(f"[Init] Using shared init: {initial_states} states, {len(validation_data)} validation samples, {len(training_data)} training samples")

    # Minimal state required by DFALearner.propose_automata().
    # Coverage fields from the original Anchor implementation are intentionally omitted.
    prealloc_size = batch_size * 10_000
    state: dict = {
        't_nsamples':       defaultdict(lambda: 0.),
        't_accepted':       defaultdict(lambda: 0.),
        't_order':          defaultdict(list),
        't_positives':      defaultdict(lambda: 0.),
        't_negatives':      defaultdict(lambda: 0.),
        'prealloc_size':    prealloc_size,
        'data':             list(validation_data),
        'labels':           np.zeros(prealloc_size, dtype=np.float64),
        'current_idx':      len(validation_data),
    }
    state['t_order'][()] = []

    return initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels


def _compute_agreement(automaton, data, labels) -> float:
    """Compute local agreement using the shared automata learner."""
    if _AUTO_INSTANCE is None or automaton is None or data is None or labels is None:
        return 0.0
    if len(data) == 0 or len(labels) == 0:
        return 0.0
    return compute_automaton_agreement(
        automaton=automaton,
        data=data,
        labels=labels,
        accept_fn=_AUTO_INSTANCE.check_path_accepted,
    )

def _select_final(all_history: list,
                  select_by: str,
                  agreement_threshold: float,
                  state_threshold: int,
                  automaton_type: str,
                  initial_dfa,
                  initial_states: int,
                  output_dir: str,
                  validation_data: list = None,
                  validation_labels: np.ndarray = None) -> dict:
    """
    Select the final automaton from all evaluated candidates.

    Selection modes
    ---------------
    "agreement"
        Among candidates with training_agreement >= agreement_threshold,
        select the one with the fewest states. If none qualify, return the
        highest-agreement candidate as best effort.
    "state"
        Among candidates with states <= state_threshold, select the one with
        the highest training agreement. If none qualify, return best effort.
    """
    select_by = "agreement" if select_by in (None, "agreement") else select_by

    def _state_count(automaton) -> int:
        return len(automaton.states) if hasattr(automaton, "states") else 0

    def _cleanup(best_record: dict, success: bool, reason: str = "") -> dict:
        automata = best_record.get("automata") or best_record.get("dfa")
        if automata is None:
            automata = initial_dfa

        if automaton_type.upper() == "DFA" and automata is not None:
            remove_unreachable_states(automata)
            try:
                dfa_to_graphviz(automata, filename="final_automata", output_dir=output_dir)
            except Exception as exc:
                print(f"  [WARNING] Could not save final DFA graph: {exc}")

        final_val_agreement = _compute_agreement(automata, validation_data if validation_data is not None else [], validation_labels if validation_labels is not None else [])
        training_agreement = float(best_record.get("training_agreement", 0.0) or 0.0)
        states = int(best_record.get("states") or _state_count(automata))

        return {
            "automata": automata,
            "training_agreement": training_agreement,
            "validation_agreement": final_val_agreement,
            "size": states,
            "initial_states": initial_states,
            "coverage": [],
            "examples": [],
            "success": success,
            "reason": reason,
            "false_accept": [],
            "true_reject": [],
        }

    print(f"\n[SELECT] select_by='{select_by}', "
          f"agreement_threshold={agreement_threshold}, state_threshold={state_threshold}")
    print(f"  Total candidates in history: {len(all_history)}")
    print(f"  Initial states (from shared_init): {initial_states}")

    if not all_history:
        print("  [SELECT] No candidates – returning initial DFA.")
        initial_train = _compute_agreement(initial_dfa, validation_data if validation_data is not None else [], validation_labels if validation_labels is not None else [])
        return {
            "automata": initial_dfa,
            "training_agreement": initial_train,
            "validation_agreement": initial_train,
            "size": _state_count(initial_dfa),
            "initial_states": initial_states,
            "coverage": [],
            "examples": [],
            "success": False,
            "reason": "No candidates generated. Returning initial automaton only.",
            "false_accept": [],
            "true_reject": [],
        }

    best_by_agreement = max(all_history, key=lambda x: float(x.get("training_agreement", 0.0) or 0.0))

    if select_by == "agreement":
        qualified = [
            r for r in all_history
            if float(r.get("training_agreement", 0.0) or 0.0) >= agreement_threshold
        ]
        if qualified:
            best = min(qualified, key=lambda x: int(x.get("states", 0) or 0))
            print(f"  [agreement mode] {len(qualified)} candidate(s) meet training_agreement >= {agreement_threshold}.")
            print(f"  Selected: states={best['states']}, training_agreement={best['training_agreement']:.4f}")
            return _cleanup(best, success=True, reason="Found candidate meeting agreement threshold")

        print(f"  [agreement mode] No candidate meets agreement >= {agreement_threshold}. Using best-effort.")
        print(f"  Best-effort: states={best_by_agreement['states']}, training_agreement={best_by_agreement['training_agreement']:.4f}")
        return _cleanup(
            best_by_agreement,
            success=False,
            reason=f"No candidate meets agreement >= {agreement_threshold}. Returning best-effort with highest agreement.",
        )

    if select_by == "state":
        under = [r for r in all_history if int(r.get("states", 0) or 0) <= state_threshold]
        if under:
            best = max(under, key=lambda x: float(x.get("training_agreement", 0.0) or 0.0))
            print(f"  [state mode] {len(under)} candidate(s) have states <= {state_threshold}.")
            print(f"  Selected: states={best['states']}, training_agreement={best['training_agreement']:.4f}")
            return _cleanup(best, success=True, reason="Found candidate meeting state threshold")

        print(f"  [state mode] No candidate has states <= {state_threshold}. Using best-effort.")
        return _cleanup(
            best_by_agreement,
            success=False,
            reason=f"No candidate has states <= {state_threshold}. Returning best-effort with highest agreement.",
        )

    raise ValueError(f"Unknown select_by='{select_by}'. Use 'agreement' or 'state'.")

# ======================================================================
# Simulated Annealing Baseline (using simanneal.Annealer)
# ======================================================================

class DFAAnnealer(Annealer):
    """Simulated Annealing for DFA optimization."""
    
    def __init__(self, initial_dfa, training_data, training_labels, 
                 validation_data, validation_labels, state, 
                 output_dir, beam_size, max_evaluations,
                 agreement_threshold: float = 0.8,
                 candidate_pool_size: int = 5):
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.propose_state = state
        self.output_dir = output_dir
        self.beam_size = beam_size
        self.max_evaluations = max_evaluations
        self.agreement_threshold = float(agreement_threshold)
        self.candidate_pool_size = max(1, int(candidate_pool_size))
        self.evaluations_count = 0
        self.iteration_count = 0
        self.all_history = []
        self.seen_ids = set()
        self.initial_states = max(1, len(initial_dfa.states)) if hasattr(initial_dfa, 'states') else 1
        self.operator_stats = Counter()
        
        # Track best solution seen globally
        self.best_dfa = initial_dfa.copy() if hasattr(initial_dfa, 'copy') else copy.deepcopy(initial_dfa)
        self.best_agreement = 0.0
        self.best_energy = float('inf')
        
        # Early stopping: track consecutive iterations without improvement
        self.no_improve_count = 0
        self.no_improve_threshold = 10  # Stop after 10 consecutive iterations without improvement
        self.last_best_energy = float('inf')
        
        # Initialize parent Annealer with initial_state as argument
        super().__init__(initial_state=initial_dfa.copy())
        self.Tmax = 10.0
        self.Tmin = 0.001
        self.steps = 100
    
    def move(self):
        """
        Multi-candidate SA move.

        In one SA round, generate a pool of neighboring DFAs from the current DFA,
        evaluate every candidate in that pool, and use the lowest-energy candidate
        as the proposal passed to simanneal's Metropolis acceptance rule.
        """
        if self.evaluations_count >= self.max_evaluations:
            print(f"[SA] Budget exhausted: {self.evaluations_count} >= {self.max_evaluations}")
            self.user_exit = True
            return

        self.iteration_count += 1
        remaining_budget = self.max_evaluations - self.evaluations_count
        pool_size = min(self.candidate_pool_size, remaining_budget)

        try:
            candidates = _AUTO_INSTANCE.propose_multiple_neighbors(
                self.state,
                self.propose_state,
                self.training_data,
                self.training_labels,
                n_candidates=pool_size,
                seen_signatures=set(),
                max_attempts_per_candidate=3,
            )
        except Exception as e:
            print(f"  [SA] Error generating candidate pool: {e}")
            candidates = []

        if not candidates:
            print(f"  [SA-iter{self.iteration_count}] Candidate generation failed")
            return

        generated_ops = list(getattr(_AUTO_INSTANCE, "last_proposed_ops", []))
        best_candidate = None
        best_candidate_idx = None
        best_candidate_agreement = 0.0
        best_candidate_energy = float("inf")
        parent_states = _dfa_state_count(self.state)
        candidate_rows = []

        for cand_idx, candidate in enumerate(candidates[:pool_size]):
            self.evaluations_count += 1
            candidate_train_agreement = _compute_agreement(candidate, self.training_data, self.training_labels)
            candidate_val_agreement = 0.0

            _AUTO_INSTANCE.add_to_history(
                self.all_history,
                self.seen_ids,
                candidate,
                candidate_train_agreement,
                candidate_val_agreement,
                use_automata_key=True,
            )

            candidate_states = _dfa_state_count(candidate) or self.initial_states
            candidate_energy = _candidate_loss(candidate_train_agreement, candidate_states, self.initial_states)
            raw_op = generated_ops[cand_idx] if cand_idx < len(generated_ops) else _candidate_op_hint(parent_states, candidate_states)
            op_hint = _record_operator(self.operator_stats, raw_op)
            candidate_rows.append({
                "idx": cand_idx,
                "sig": _candidate_signature(candidate),
                "op": op_hint,
                "states": candidate_states,
                "agreement": candidate_train_agreement,
                "loss": candidate_energy,
            })

            if candidate_energy < best_candidate_energy:
                best_candidate = candidate
                best_candidate_idx = cand_idx
                best_candidate_agreement = candidate_train_agreement
                best_candidate_energy = candidate_energy

            if self.evaluations_count >= self.max_evaluations:
                break

        _print_candidate_log("SA", f"iter{self.iteration_count}", candidate_rows, selected_idx=best_candidate_idx)

        if best_candidate is None:
            return

        self.state = best_candidate

        if best_candidate_energy < self.best_energy:
            self.best_dfa = copy.deepcopy(best_candidate)
            self.best_agreement = best_candidate_agreement
            self.best_energy = best_candidate_energy
            best_states = len(best_candidate.states)
            self.no_improve_count = 0
            print(
                f"  [SA-iter{self.iteration_count}] NEW best from pool({pool_size}): "
                f"agreement={best_candidate_agreement:.4f}, states={best_states}, "
                f"evals: {self.evaluations_count}/{self.max_evaluations}"
            )
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.no_improve_threshold:
                print(f"  [SA] Early stopping: {self.no_improve_count} iterations without improvement")
                self.user_exit = True
                return

    def energy(self):
        """
        Evaluate the energy of the current state.
        Energy = -agreement + (states / initial_states)
        simanneal minimizes energy.
        """
        # self.state is managed by simanneal and will be reverted if move rejected
        agreement = _compute_agreement(self.state, self.training_data, self.training_labels)
        current_states = len(self.state.states) if hasattr(self.state, 'states') else self.initial_states
        return -agreement + (float(current_states) / self.initial_states)


def sa_dfa_search(data_type: str,
                  shared_init: SharedInit,
                  *,
                  agreement_threshold: float = 1.0,
                  state_threshold: int = 5,
                  select_by: str = "agreement",
                  init_num_samples: int = 1000,
                  batch_size: int = 100,
                  output_dir: str = "test_result/sa",
                  beam_size: int = 1,
                  steps: int = 500,
                  T_max: float = 10.0,
                  T_min: float = 0.001,
                  max_evaluations: int = 500,
                  sa_candidate_pool_size: int = 5,
                  **kwargs) -> dict:
    """
    Simulated Annealing for DFA search using simanneal.Annealer.
    
    Requires SharedInit from beam search containing initial DFA, validation data, and FIXED training data.
    
    Parameters
    ----------
    shared_init       : SharedInit – from beam search (required, includes training_data/training_labels)
    steps             : int – SA steps
    T_max, T_min      : float – temperature range
    max_evaluations   : int – max propose_automata() calls (budget limit)
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[SA] Initialising Simulated Annealing…")
    
    if shared_init is None:
        raise ValueError("[SA] FATAL: shared_init is required (from beam search)")
    
    initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels = _common_init(
        shared_init, batch_size, output_dir
    )

    # Create annealer
    annealer = DFAAnnealer(
        initial_dfa, training_data, training_labels,
        validation_data, validation_labels, state,
        output_dir, beam_size, max_evaluations,
        agreement_threshold=agreement_threshold,
        candidate_pool_size=sa_candidate_pool_size,
    )
    annealer.Tmax = T_max
    annealer.Tmin = T_min
    # Ensure enough total moves to fully consume evaluation budget
    effective_steps = max(int(steps), int(max_evaluations) + 1)
    annealer.steps = effective_steps
    
    # Run SA
    print(f"[SA] Running {effective_steps} steps with T_max={T_max}, T_min={T_min}, candidate_pool={sa_candidate_pool_size} (budget={max_evaluations})")
    print(f"[SA] Initial DFA: {initial_states} states, initial training agreement: {_compute_agreement(initial_dfa, training_data, training_labels):.4f}")
    best_dfa, best_energy = annealer.anneal()
    
    # Prepare all_history with collected candidates
    all_history = annealer.all_history
    initial_train_agreement = _compute_agreement(initial_dfa, training_data, training_labels)
    initial_val_agreement = _compute_agreement(initial_dfa, validation_data, validation_labels)
    _AUTO_INSTANCE.add_to_history(all_history, annealer.seen_ids, initial_dfa, initial_train_agreement, initial_val_agreement, use_automata_key=True)
    
    # Use annealer.best_dfa (which we maintain and simanneal's reversion doesn't affect)
    print(f"[SA] Completed {annealer.evaluations_count} evaluations")
    print(f"[SA] Best DFA: {len(annealer.best_dfa.states)} states, Best agreement: {annealer.best_agreement:.4f}")
    _print_operator_summary("SA", annealer.operator_stats, annealer.evaluations_count, max_evaluations)
        
    gc.collect()

    result = _select_final(
        all_history, select_by, agreement_threshold, state_threshold,
        "DFA", initial_dfa, initial_states, output_dir,
        validation_data=validation_data, validation_labels=validation_labels
    )
    result['initial_states'] = initial_states  # Add missing initial_states to result
    result['initial_train_agreement'] = initial_train_agreement
    result['initial_val_agreement'] = initial_val_agreement
    result['operator_counts'] = dict(annealer.operator_stats)
    result['evaluations_used'] = int(annealer.evaluations_count)
    result['max_evaluations'] = int(max_evaluations)
    return result


# ======================================================================
# Genetic Algorithm (Round-based, Comparable to Beam Search)
# ======================================================================

# DEAP requires global creator registration; guard against double-registration.
if not hasattr(creator, 'DFAFitness'):
    creator.create("DFAFitness", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'DFAIndividual'):
    creator.create("DFAIndividual", list, fitness=creator.DFAFitness)


def ga_dfa_search(data_type: str,
                  shared_init: SharedInit,
                  *,
                  agreement_threshold: float = 1.0,
                  state_threshold: int = 5,
                  select_by: str = "agreement",
                  init_num_samples: int = 1000,
                  batch_size: int = 100,
                  output_dir: str = "test_result/ga",
                  population_size: int = 1,
                  tournament_size: int = 2,
                  max_evaluations: int = 500,
                  **kwargs) -> dict:
    """
    Standard Genetic Algorithm for DFA search (no crossover, mutation-only).
    
    Since DFA crossover is not well-defined, uses only selection + mutation.
    
    Algorithm:
    1. Initialize population with initial_dfa
    2. While budget allows:
       a) Generate population_size new offspring via tournament selection + mutation
       b) Evaluate all offspring
       c) Replace entire population with new offspring (generational replacement)
    3. Return best individual from all history
    
    Requires SharedInit from beam search containing initial DFA, validation data, and FIXED training data.
    
    Parameters
    ----------
    shared_init       : SharedInit – from beam search (required, includes training_data/training_labels)
    population_size   : int – fixed population size
    tournament_size   : int – tournament selection size
    max_evaluations   : int – max candidates generated (budget limit)
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    def _create_individual(dfa_seed):
        # dfa_seed is already a fresh object from _propose_single_neighbor or fallback copy
        return creator.DFAIndividual([dfa_seed])

    def _evaluate(individual):
        dfa = individual[0]
        train_agreement = _compute_agreement(dfa, training_data, training_labels)
        current_states = len(dfa.states) if hasattr(dfa, 'states') else initial_states
        fitness_value = train_agreement - current_states / initial_states
        candidate_eval_log[id(dfa)] = {
            "agreement": float(train_agreement),
            "states": int(current_states),
            "loss": _candidate_loss(train_agreement, current_states, initial_states),
            "fitness": float(fitness_value),
            "sig": _candidate_signature(dfa),
        }
        # Compute validation agreement (needed for final selection)
        val_agreement = 0.0
        _AUTO_INSTANCE.add_to_history(all_history, seen_ids, dfa, train_agreement, val_agreement, use_automata_key=True)
        return (fitness_value,)

    def _as_fitness_tuple(fit):
        # DEAP fitness.values expects a sequence matching Fitness.weights length.
        return fit if isinstance(fit, tuple) else (float(fit),)
    
    print("=" * 70)
    print("[GA-Original] Initialising Genetic Algorithm (mutation-only, no crossover)…")
    
    if shared_init is None:
        raise ValueError("[GA] FATAL: shared_init is required (from beam search)")
    
    # Get initial artefacts from shared_init
    initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels = _common_init(
        shared_init, batch_size, output_dir
    )
    
    # Initialize history containers BEFORE using them
    all_history: List[dict] = []
    seen_ids: set = set()
    candidate_eval_log: Dict[int, dict] = {}
    operator_stats = Counter()
    evaluations_count = 0
    
    # Seed history with initial DFA
    initial_train_agreement = _compute_agreement(initial_dfa, training_data, training_labels)
    initial_val_agreement = _compute_agreement(initial_dfa, validation_data, validation_labels)
    _AUTO_INSTANCE.add_to_history(all_history, seen_ids, initial_dfa, initial_train_agreement, initial_val_agreement, use_automata_key=True)
    evaluations_count += 1  # Count the initial DFA evaluation
    

    # DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("map", map)
    toolbox.register("evaluate", _evaluate)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Initialize population: generate population_size individuals
    print(f"[GA-Init] Generating initial population of {population_size} individuals...")
    print(f"[GA-Init] Initial DFA: {initial_states} states, initial training agreement: {initial_train_agreement:.4f}")
    population_dfas = []
    population_ops = []
    seen_signatures_init = set()
    
    for _ in range(population_size):
        try:
            neighbor = _AUTO_INSTANCE._propose_single_neighbor(
                initial_dfa, state, training_data, training_labels, seen_signatures_init,
                max_attempts=5
            )
            population_dfas.append(neighbor)
            population_ops.append(getattr(_AUTO_INSTANCE, "last_proposed_operation", "UNKNOWN"))
        except Exception as e:
            print(f"[GA-Init] Warning: neighbor generation failed: {e}")
            population_dfas.append(initial_dfa.copy() if hasattr(initial_dfa, 'copy') else copy.deepcopy(initial_dfa))
            population_ops.append("COPY")
    population = [_create_individual(dfa) for dfa in population_dfas]
    
    # Evaluate initial population
    try:
        print("[GA-Init] Evaluating initial population")
        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = _as_fitness_tuple(fit)
    except Exception as e:
        print(f"[GA] Parallel evaluation failed, falling back to sequential: {e}")
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = _as_fitness_tuple(fit)
    evaluations_count += len(population)
    init_rows = []
    for cand_idx, ind in enumerate(population):
        dfa = ind[0]
        detail = candidate_eval_log.get(id(dfa), {})
        cand_states = detail.get("states", _dfa_state_count(dfa))
        raw_op = population_ops[cand_idx] if cand_idx < len(population_ops) else _candidate_op_hint(initial_states, cand_states)
        op_hint = _record_operator(operator_stats, raw_op)
        init_rows.append({
            "idx": cand_idx,
            "sig": detail.get("sig", _candidate_signature(dfa)),
            "op": op_hint,
            "states": cand_states,
            "agreement": detail.get("agreement", 0.0),
            "loss": detail.get("loss", _candidate_loss(detail.get("agreement", 0.0), cand_states, initial_states)),
        })
    _print_candidate_log("GA", "init", init_rows, selected_idx=None)
    gen = 0

    
    # Evolution loop: Standard GA - generate entire new generation each iteration
    # Early stopping: track best fitness across generations
    ga_no_improve_count = 0
    ga_no_improve_threshold = 10
    ga_best_fitness = -float('inf')
    
    while evaluations_count < max_evaluations:
        gen += 1
        print(f"\n[GA-Gen] Generation {gen}, evals: {evaluations_count}/{max_evaluations}")
        
        # Calculate how many offspring we can generate with remaining budget
        offspring_target = min(population_size, max_evaluations - evaluations_count)
        
        # Batch select parents
        selected_parents = toolbox.select(population, offspring_target)
        parent_state_counts = [_dfa_state_count(parent[0]) for parent in selected_parents]
        
        # For each parent, generate exactly one random neighbor
        seen_signatures_batch = set()
        candidates = []
        candidate_ops = []
        
        for parent_idx, parent in enumerate(selected_parents):
            dfa_parent = parent[0]
            try:
                neighbor = _AUTO_INSTANCE._propose_single_neighbor(
                    dfa_parent, state, training_data, training_labels, seen_signatures_batch,
                    max_attempts=1
                )
                candidates.append(neighbor)
                candidate_ops.append(getattr(_AUTO_INSTANCE, "last_proposed_operation", "UNKNOWN"))
            except Exception as exc:
                print(f"      [GA-Mutation] Parent {parent_idx}: {exc}, using copy")
                candidates.append(dfa_parent.copy() if hasattr(dfa_parent, 'copy') else copy.deepcopy(dfa_parent))
                candidate_ops.append("COPY")
        print(f"    [GA-Batch] Generated {len(candidates)} neighbors")
        
        # Batch evaluate all candidates in parallel via toolbox.map
        offspring = [_create_individual(c) for c in candidates]
        try:
            fitnesses = list(toolbox.map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = _as_fitness_tuple(fit)
            evaluations_count += len(offspring)
        except Exception as e:
            print(f"    [GA-Batch] Parallel evaluation failed, falling back to sequential: {e}")
            for ind in offspring:
                try:
                    fit = toolbox.evaluate(ind)
                    ind.fitness.values = _as_fitness_tuple(fit)
                    evaluations_count += 1
                except Exception as exc:
                    print(f"      [Warning] Sequential evaluation failed: {exc}")
                    # Last resort: use parent's fitness
                    if len(population) > 0:
                        best_ind = max(population, key=lambda x: x.fitness.values[0])
                        ind.fitness.values = best_ind.fitness.values
        
        gen_rows = []
        selected_idx = None
        selected_fitness = -float("inf")
        for cand_idx, ind in enumerate(offspring):
            dfa = ind[0]
            detail = candidate_eval_log.get(id(dfa), {})
            cand_states = detail.get("states", _dfa_state_count(dfa))
            parent_states = parent_state_counts[cand_idx] if cand_idx < len(parent_state_counts) else initial_states
            fitness = ind.fitness.values[0] if getattr(ind, "fitness", None) and ind.fitness.valid else detail.get("fitness", -float("inf"))
            if fitness > selected_fitness:
                selected_fitness = fitness
                selected_idx = cand_idx
            raw_op = candidate_ops[cand_idx] if cand_idx < len(candidate_ops) else _candidate_op_hint(parent_states, cand_states)
            op_hint = _record_operator(operator_stats, raw_op)
            gen_rows.append({
                "idx": cand_idx,
                "sig": detail.get("sig", _candidate_signature(dfa)),
                "op": op_hint,
                "states": cand_states,
                "agreement": detail.get("agreement", 0.0),
                "loss": detail.get("loss", _candidate_loss(detail.get("agreement", 0.0), cand_states, initial_states)),
            })
        _print_candidate_log("GA", f"gen{gen}", gen_rows, selected_idx=selected_idx)

        # keep the best individual(s) from current population
        elite_size = max(1, population_size // 10)  # Keep top 10%
        elite = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)[:elite_size]
        
        # Replace population with new offspring + elite
        population[:] = offspring + elite
        if len(population) > population_size:
            population[:] = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)[:population_size]
        
        # Check early stopping condition based on no improvement
        sizes = [len(ind[0].states) if hasattr(ind[0], 'states') else 0 for ind in population]
        best_states_this_gen = min(sizes) if sizes else 0
        current_best_fitness = max([ind.fitness.values[0] for ind in population]) if population else -float('inf')
        
        if current_best_fitness > ga_best_fitness:
            ga_best_fitness = current_best_fitness
            ga_no_improve_count = 0
            print(f"  [GA-Gen] Min states: {best_states_this_gen}, Avg states: {np.mean(sizes):.1f} [IMPROVED]")
        else:
            ga_no_improve_count += 1
            print(f"  [GA-Gen] Min states: {best_states_this_gen}, Avg states: {np.mean(sizes):.1f} (no improve: {ga_no_improve_count}/{ga_no_improve_threshold})")
            
            # Early stopping if no improvement for threshold generations
            if ga_no_improve_count >= ga_no_improve_threshold:
                print(f"  [GA] Early stopping: {ga_no_improve_count} generations without improvement")
                break  
    
    print(f"\n[GA-Original] Completed {gen} generations, {len(all_history)} candidates total, final evals: {evaluations_count}/{max_evaluations}")
    
    gc.collect()

    result = _select_final(
        all_history, select_by, agreement_threshold, state_threshold,
        "DFA", initial_dfa, initial_states, output_dir,
        validation_data=validation_data, validation_labels=validation_labels
    )
    result['initial_states'] = initial_states  # Add missing initial_states to result
    result['initial_train_agreement'] = initial_train_agreement
    result['initial_val_agreement'] = initial_val_agreement
    result['operator_counts'] = dict(operator_stats)
    result['evaluations_used'] = int(evaluations_count)
    result['max_evaluations'] = int(max_evaluations)
    return result


# ======================================================================
# Particle Swarm Optimisation (using pyswarms.GlobalBestPSO)
# ======================================================================

def pso_dfa_search(data_type: str,
                   shared_init: SharedInit,
                   *,
                   agreement_threshold: float = 1.0,
                   state_threshold: int = 5,
                   select_by: str = "agreement",
                   init_num_samples: int = 1000,
                   batch_size: int = 100,
                   output_dir: str = "test_result/pso",
                   beam_size: int = 1,
                   n_particles: int = 10,
                   max_evaluations: int = 500,
                   pso_candidate_pool_size: int = 5,
                   **kwargs) -> dict:
    """
    Particle Swarm Optimisation for DFA search.
    
    Uses the local PSOAutomataOptimizer class for state minimization
    while maintaining agreement above a threshold.
    
    Requires SharedInit from beam search containing initial DFA, validation data, and FIXED training data.
    
    Parameters
    ----------
    shared_init       : SharedInit – from beam search (required, includes training_data/training_labels)
    n_particles       : int – number of particles in swarm
    beam_size         : int – beam_size for candidate generation (not used, kept for compatibility)
    max_evaluations   : int – controls max iterations
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[PSO] Initialising Particle Swarm Optimisation (PSOAutomataOptimizer)…")
    
    if shared_init is None:
        raise ValueError("[PSO] FATAL: shared_init is required (from beam search)")
    
    initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels = _common_init(
        shared_init, batch_size, output_dir
    )
    initial_train_agreement = _compute_agreement(initial_dfa, training_data, training_labels)
    initial_val_agreement = _compute_agreement(initial_dfa, validation_data, validation_labels)

    all_history: List[dict] = []
    seen_ids: set = set()
    pso_verbose = bool(kwargs.get('pso_verbose', True))
    pso_max_ops_per_iteration = int(kwargs.get('pso_max_ops_per_iteration', 1))
    pso_candidate_pool_size = max(1, int(kwargs.get('pso_candidate_pool_size', pso_candidate_pool_size)))
    
    # Calculate iterations based on max_evaluations and n_particles
    # Each PSO iteration evaluates n_particles candidates, so:
    evals_per_iteration = max(1, n_particles) * max(1, pso_candidate_pool_size)
    n_iterations = max(1, (max_evaluations - 1) // evals_per_iteration)
    
    # Create PSOAutomataOptimizer
    try:
        optimizer = PSOAutomataOptimizer(
            initial_dfa=initial_dfa,
            threshold=agreement_threshold,
            data=training_data,
            labels=training_labels,            
            validation_data=validation_data,
            validation_labels=validation_labels,            
            learner=shared_init.learner,
            n_particles=n_particles,
            n_iterations=n_iterations,
            w=0.7,
            c1=1.5,
            c2=1.5,
            verbose=pso_verbose,
            max_evaluations=max_evaluations,
            slot_beam_size=max(1, beam_size),
            max_ops_per_iteration=pso_max_ops_per_iteration,
            candidate_pool_size=pso_candidate_pool_size,
        )
    except Exception as e:
        print(f"[PSO ERROR] Failed to initialize PSOAutomataOptimizer: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return initial DFA
        result = _select_final(
            all_history, select_by, agreement_threshold, state_threshold,
            "DFA", initial_dfa, initial_states, output_dir,
            validation_data=validation_data, validation_labels=validation_labels
        )
        result['initial_states'] = initial_states
        return result
    
    # Run PSO optimization
    print(f"\n[PSO] Starting optimization...")
    print(f"  Initial DFA: {initial_states} states, initial training agreement: {initial_train_agreement:.4f}")
    print(f"  Particles={n_particles}, candidate_pool_per_particle={pso_candidate_pool_size}, iterations={n_iterations}")
    try:
        pso_result = optimizer.optimize(n_particles=n_particles, n_iterations=n_iterations, save_trajectory=True)
        
        best_agreement = pso_result.get('best_agreement', 0.0)
        best_validation_agreement = pso_result.get('best_validation_agreement', 0.0)

        print(f"\n[PSO] Optimization completed successfully")
        print(f"  Best states: {pso_result.get('best_states', 'N/A')}")
        print(f"  Best agreement: {float(best_agreement):.4f}")
        print(f"  Best validation agreement: {float(best_validation_agreement):.4f}")
        print(f"  Best loss: {float(pso_result.get('best_loss', 0.0)):.4f}")
        
        # Collect all candidates from PSO history into all_history (use add_to_history for consistency)
        if 'all_history' in pso_result:
            for candidate in pso_result['all_history']:
                try:
                    _AUTO_INSTANCE.add_to_history(all_history, seen_ids, candidate['automata'],
                                    candidate.get('training_agreement', 0.0),
                                    candidate.get('validation_agreement', 0.0),
                                    use_automata_key=True)
                except Exception:
                    continue
        # Optionally report how many evaluations PSO performed
        if 'evaluations' in pso_result:
            print(f"  [PSO] Evaluations used: {pso_result['evaluations']} / {pso_result.get('max_evaluations')}")
        
        # Check if we have a 2-state solution from PSO
        min_states = min([record['states'] for record in all_history] if all_history else [float('inf')])
        if min_states == 2:
            print(f"  [PSO] Early stopping: reached 2 states")
        
    except Exception as e:
        print(f"[PSO WARNING] Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[PSO] Total candidates collected: {len(all_history)}")
    _print_operator_summary(
        "PSO",
        getattr(optimizer, "operator_stats", Counter()),
        getattr(optimizer, "evaluations_count", 0),
        max_evaluations,
    )
    
    gc.collect()

    result = _select_final(
        all_history, select_by, agreement_threshold, state_threshold,
        "DFA", initial_dfa, initial_states, output_dir,
        validation_data=validation_data, validation_labels=validation_labels
    )
    result['initial_states'] = initial_states
    result['initial_train_agreement'] = initial_train_agreement
    result['initial_val_agreement'] = initial_val_agreement
    result['operator_counts'] = dict(getattr(optimizer, "operator_stats", Counter()))
    result['evaluations_used'] = int(getattr(optimizer, "evaluations_count", 0))
    result['max_evaluations'] = int(max_evaluations)
    return result
