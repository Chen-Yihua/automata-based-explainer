"""
PSO-based DFA Optimizer

This module implements a Particle Swarm Optimization (PSO) approach for optimizing
Deterministic Finite Automata (DFA).

Key Components:
1. Objective function: training accuracy and state-ratio tradeoff
2. Discrete space mapping: PSO particle positions → DFA modifications
3. Integration with existing DFA operators: _propose_delete, _propose_merge, _propose_delta
4. State sanitation: Applies remove_unreachable_states after each DFA modification
"""

import gc
import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from copy import deepcopy
import random

try:
    from pyswarms.single.global_best import GlobalBestPSO
    PSO_AVAILABLE = True
except ImportError:
    PSO_AVAILABLE = False
    print("[WARNING] pyswarms not found. PSO optimizer will not be available.")

from .dfa_learner import remove_unreachable_states

class PSOAutomataOptimizer:
    """
    Particle Swarm Optimization-based DFA optimizer that searches for the minimal
    DFA satisfying accuracy constraints while minimizing state count.
    
    Core Concept:
    - Particles represent sequences of DFA modifications (DELETE, MERGE, DELTA)
    - Particle positions are continuous vectors mapped to discrete DFA operations
    - Fitness is computed based on: Accuracy (must satisfy THRESHOLD) + State Count
    - PSO's pbest and gbest guide particles toward better DFAs
    
    Attributes:
        initial_dfa: Starting DFA for optimization
        threshold: Minimum required accuracy (0.0 to 1.0)
        data: Training sequences
        labels: Training labels (binary)
        learner: DFALearner instance for accessing _propose_delete, _propose_merge, _propose_delta
        state_metrics: Dictionary for tracking DFA evaluation metrics
        n_particles: Number of particles in the swarm
        n_iterations: Number of PSO iterations
        w: Inertia weight for PSO
        c1: Cognitive coefficient for PSO
        c2: Social coefficient for PSO
        verbose: Enable detailed logging
    """
    
    def __init__(self, 
                 initial_dfa,
                 threshold: float = 0.8,
                 data: List = None,
                 labels: np.ndarray = None,
                 learner = None,
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
                 max_ops_per_iteration: Optional[int] = None):
        """
        Initialize PSO optimizer.
        
        Parameters
        ----------
        initial_dfa : Dfa
            Starting DFA for optimization
        threshold : float
            Minimum required training accuracy (default: 0.8)
        data : list
            Training sequences
        labels : np.ndarray
            Training labels (1 for accept, 0 for reject)
        learner : DFALearner
            DFALearner instance with access to _propose_delete, _propose_merge, _propose_delta
        validation_data : list, optional
            Validation sequences for evaluating DFA quality (default: None)
        validation_labels : np.ndarray, optional
            Validation labels (default: None)
        n_particles : int
            Number of particles in swarm (default: 10)
        n_iterations : int
            Number of PSO iterations (default: 20)
        w : float
            Inertia weight (default: 0.7)
        c1 : float
            Cognitive coefficient (default: 1.5)
        c2 : float
            Social coefficient (default: 1.5)
        verbose : bool
            Enable logging (default: True)
        """
        if not PSO_AVAILABLE:
            raise ImportError("pyswarms is required for PSOAutomataOptimizer. Install with: pip install pyswarms")
        
        self.initial_dfa = initial_dfa
        self.initial_states = max(1, len(initial_dfa.states)) if hasattr(initial_dfa, 'states') else 1
        self.threshold = threshold
        self.data = data or []
        self.labels = labels if labels is not None else np.array([])
        self.validation_data = validation_data or []
        self.validation_labels = validation_labels if validation_labels is not None else np.array([])
        self.learner = learner
        self.verbose = verbose
        
        # PSO edit-space configuration
        # max_ops_per_iteration controls how many discrete DFA edits one particle can apply.
        # Keep default behavior at 1, but allow larger values for deeper compression trajectories.
        configured_ops = int(max_ops_per_iteration) if max_ops_per_iteration is not None else 1
        self.slot_count = max(1, configured_ops)
        self.slot_width = 3                 # (operation_type, target1_idx, target2_idx)
        self.max_ops_per_iteration = max(1, configured_ops)
        slot_beam_default = slot_beam_size if slot_beam_size is not None else 5
        self.slot_beam_size = max(1, slot_beam_default)
        self.dimensions = self.slot_count * self.slot_width

        # PSO hyperparameters
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # State tracking
        self.state_metrics: Dict[int, Dict] = {}
        self._init_state_metrics()
        
        # Deduplication tracking (for _propose_delete/merge/delta)
        self.seen_signatures: set = set()
        
        # Global best tracking
        self.gbest_dfa = None
        self.gbest_fitness = float('inf')
        self.gbest_accuracy = 0.0
        self.gbest_val_accuracy = 0.0
        self.gbest_states = float('inf')
        
        # Early stopping: track consecutive iterations without improvement
        self.pso_no_improve_count = 0
        self.pso_no_improve_threshold = 10
        self.pso_last_best_fitness = float('inf')
        
        # All candidates history (for comparison with other methods)
        self.all_history: List[Dict] = []
        self.seen_ids: set = set()
        # Evaluation budget tracking
        self.evaluations_count: int = 0
        self.max_evaluations: Optional[int] = max_evaluations
        self.reached_two_states: bool = False
        
        # Evaluate initial DFA and set as initial global best.
        # _evaluate_and_cache returns only (accuracy, num_states, loss).
        # Validation accuracy is computed separately for initial/final reporting.
        initial_accuracy, initial_states, initial_loss = self._evaluate_and_cache(self.initial_dfa)
        initial_val_acc = self._compute_validation_accuracy(self.initial_dfa)
        self.evaluations_count += 1  # Count initial evaluation against budget
        
        # initialize gbest with initial_dfa, regardless of accuracy threshold
        # This ensures PSO has a baseline to work from and can explore toward the threshold
        self.gbest_dfa = deepcopy(self.initial_dfa)
        self.gbest_fitness = initial_loss
        self.gbest_accuracy = initial_accuracy
        self.gbest_val_accuracy = initial_val_acc
        self.gbest_states = initial_states
        self._add_to_history(self.initial_dfa, initial_accuracy, initial_val_acc)
        
        if self.verbose:
            val_str = f", val_acc={initial_val_acc:.4f}" if len(self.validation_data) > 0 else ""
            print(f"[PSO Init] Initial DFA: {len(self.initial_dfa.states)} states, accuracy={initial_accuracy:.4f}{val_str}")
    
    def _init_state_metrics(self):
        """Initialize state metrics dictionary (used by learner's update_state_metrics)"""
        self.state_metrics = {
            't_nsamples': {},      # Total number of samples
            't_positives': {},     # True positives (correctly accepted)
            't_negatives': {},     # True negatives (correctly rejected)
        }
    

    def _add_to_history(self, dfa, training_accuracy: float, validation_accuracy: float) -> None:
        """Track DFA in history for final comparison (via learner)."""
        self.learner.add_to_history(
            self.all_history, self.seen_ids, dfa,
            training_accuracy, validation_accuracy,
            use_automata_key=True
        )
    
    def _compute_validation_accuracy(self, dfa) -> float:
        """Compute validation accuracy for reporting (initial/final only)."""
        if self.learner is None or len(self.validation_data) == 0 or len(self.validation_labels) == 0:
            return 0.0
        try:
            val_accepts = np.array([self.learner.check_path_accepted(dfa, p) for p in self.validation_data])
            val_correct = np.sum(val_accepts == self.validation_labels)
            return float(val_correct) / len(self.validation_labels)
        except Exception:
            return 0.0

    def _evaluate_and_cache(self, dfa):
        """
        Evaluate DFA and cache metrics.
        
        Parameters
        ----------
        dfa : Dfa
            DFA to evaluate
            
        Returns
        -------
        tuple
            (accuracy, num_states, loss)
        """
        dfa_id = id(dfa)
        n_samples = len(self.data) if len(self.data) > 0 else 1
        
        # Check if cached
        if dfa_id in self.state_metrics['t_nsamples']:
            true_pos = self.state_metrics['t_positives'].get(dfa_id, 0)
            true_neg = self.state_metrics['t_negatives'].get(dfa_id, 0)
            accuracy = (true_pos + true_neg) / n_samples
        else:
            # Evaluate using learner's path checking method
            if self.learner and len(self.data) > 0 and len(self.labels) > 0:
                accepts = np.array([self.learner.check_path_accepted(dfa, p) for p in self.data])
                true_accept = np.sum((self.labels == 1) & (accepts == True))
                false_reject = np.sum((self.labels == 0) & (accepts == False))
            else:
                true_accept = 0
                false_reject = 0
            
            # Cache metrics
            self.state_metrics['t_nsamples'][dfa_id] = float(n_samples)
            self.state_metrics['t_positives'][dfa_id] = float(true_accept)
            self.state_metrics['t_negatives'][dfa_id] = float(false_reject)
            
            accuracy = (true_accept + false_reject) / n_samples
        
        num_states = len(dfa.states)
        loss = self._compute_loss(accuracy, num_states)
        return accuracy, num_states, loss
    
    def _compute_loss(self, accuracy: float, num_states: int) -> float:
        """
        Compute PSO loss using the original baseline formula.

        Loss = -accuracy + (num_states / initial_states)
        
        Parameters
        ----------
        accuracy : float
            Training accuracy of the DFA
        num_states : int
            Number of states in the DFA
            
        Returns
        -------
        float
            Loss value to minimize
        """
        return -accuracy+num_states/self.initial_states
    
    def _update_gbest(self, dfa, accuracy: float, num_states: int, loss: float, val_accuracy: float = 0.0):
        """Update global best if current DFA has better loss (NO threshold check).
        
        This is identical to GA/SA behavior: optimize freely during search,
        let _select_final() handle threshold filtering at the end.
        """
        if loss < self.gbest_fitness:
            self.gbest_dfa = deepcopy(dfa)
            self.gbest_fitness = loss
            self.gbest_accuracy = accuracy
            self.gbest_val_accuracy = val_accuracy
            self.gbest_states = num_states
            val_str = f", val_acc={val_accuracy:.4f}" if val_accuracy > 0 else ""
            if self.verbose:
                print(f"    [GBEST UPDATE] New best: {num_states} states, accuracy={accuracy:.4f}{val_str}, loss={loss:.4f}")
    
    def objective_function(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        PSO objective function (to minimize).
        
        Parameters
        ----------
        X : np.ndarray
            Particle positions with shape (n_particles, n_dimensions)
            Each row is a particle's position vector
        **kwargs : dict
            Additional keyword arguments (e.g., callback from pyswarms)
            
        Returns
        -------
        np.ndarray
            Loss values for each particle (to minimize)
        """
        n_particles = X.shape[0]
        losses = np.zeros(n_particles)
        iteration_improved = False
        iteration_best_fitness = self.pso_last_best_fitness
        
        for particle_id in range(n_particles):
            # Budget check: Each particle evaluation counts as 1 unit
            # (Standard PSO: evaluations_count = n_particles evaluated per iteration)
            if self.max_evaluations is not None and self.evaluations_count >= self.max_evaluations:
                # Budget exhausted - signal to stop optimizer
                if self.verbose:
                    print(f"  [PSO-obj] Budget exhausted before particle {particle_id}: {self.evaluations_count}/{self.max_evaluations}")
                raise RuntimeError("PSO budget exhausted")
            
            # Map continuous position to discrete DFA sequence
            # (Multiple internal slot operations may occur, but counted as 1 particle evaluation)
            dfa, ops = self._map_position_to_dfa(X[particle_id])
            
            # Increment evaluation count (1 per particle, not per slot operation)
            # This correctly accounts for PSO budget: max_evals = n_particles × n_iterations
            self.evaluations_count += 1
            
            # Ensure DFA is valid (no unreachable states)
            remove_unreachable_states(dfa)
            
            # Evaluate DFA
            accuracy, num_states, loss = self._evaluate_and_cache(dfa)

            # Track in history
            self._add_to_history(dfa, accuracy, 0.0)
            
            # Update global best
            self._update_gbest(dfa, accuracy, num_states, loss, 0.0)
            
            # Track whether this iteration improved the running best.
            if loss < iteration_best_fitness:
                iteration_best_fitness = loss
                self.pso_last_best_fitness = loss
                iteration_improved = True

            losses[particle_id] = loss

        # Early stopping is based on full PSO iterations, not individual particles.
        if iteration_improved:
            self.pso_no_improve_count = 0
        else:
            self.pso_no_improve_count += 1
            if self.pso_no_improve_count >= self.pso_no_improve_threshold:
                if self.verbose:
                    print(f"  [PSO] Early stopping: {self.pso_no_improve_count} iterations without improvement")
                raise RuntimeError(
                    f"PSO early stopping: no improvement for {self.pso_no_improve_threshold} iterations"
                )
        
        return losses
    def _map_position_to_dfa(self, position: np.ndarray) -> Tuple[Any, List[str]]:
        """
        Map continuous particle position to a DFA through discrete operations.
        
        Strategy:
        1. Start from INITIAL DFA
        2. Divide dimensions into slots (operation slots)
        3. Each slot = [op_type_value, target1_value, target2_value]
        4. Decode op_type_value → 'DELETE' / 'MERGE' / 'DELTA' via tanh normalization
        5. Select target candidate(s) via normalized target values
        6. Apply the SPECIFIC operation via _apply_slot_operation(op_type)
        7. Ensure validity with remove_unreachable_states
        
        The key difference from trajectory-based approaches:
        - Every particle position uniquely maps to one DFA
        - PSO velocity updates modify positions in a metric space
        - pbest and gbest are meaningful position references
        
        Parameters
        ----------
        position : np.ndarray
            Continuous position vector (typically in [-1, 1] range from PSO)
            
        Returns
        -------
        tuple
            (modified_dfa, list_of_operations_applied)
        """
        # Always start from initial DFA
        # This ensures same position → same DFA (deterministic mapping)
        current_dfa = deepcopy(self.initial_dfa)
        operations = []
        
        slot_vectors = self._extract_slot_vectors(position)
        if not slot_vectors:
            return current_dfa, operations

        max_slots = min(self.max_ops_per_iteration, len(slot_vectors))

        for slot_idx in range(max_slots):
            slot = slot_vectors[slot_idx]
            op_type = self._decode_operation_type(slot['op_value'])
            if op_type == "SKIP":
                operations.append("SKIP")
                continue

            next_dfa, descriptor = self._apply_slot_operation(
                current_dfa,
                op_type,
                slot['target1_value'],
                slot['target2_value']
            )

            if next_dfa is None:
                operations.append(f"{descriptor}")
                continue

            current_dfa = next_dfa
            operations.append(f"{op_type}:{descriptor}")
        
        return current_dfa, operations

    def _extract_slot_vectors(self, position: np.ndarray) -> List[Dict[str, float]]:
        """Extract slot vectors from position. Each slot encodes an operation with two independent targets."""
        slot_vectors: List[Dict[str, float]] = []
        for slot_idx in range(self.slot_count):
            base_idx = slot_idx * self.slot_width
            if base_idx + self.slot_width > len(position):
                break
            slot_vectors.append({
                'slot_idx': slot_idx,
                'op_value': float(position[base_idx]),          # Operation type encoding
                'target1_value': float(position[base_idx + 1]),  # Primary target (DELETE: state; MERGE: 1st state; DELTA: src state)
                'target2_value': float(position[base_idx + 2]),  # Secondary target (MERGE: 2nd state; DELTA: dst state)
            })
        return slot_vectors

    def _decode_operation_type(self, raw_value: float) -> str:
        normalized = self._normalize_to_unit_interval(raw_value)
        if normalized < 1 / 3:
            return "DELETE"
        elif normalized < 2 / 3:
            return "MERGE"
        else:
            return "DELTA"

    def _apply_slot_operation(self, current_dfa, op_type: str, target1_value: float, target2_value: float) -> Tuple[Optional[Any], str]:
        """
        Apply a specific DFA operation (DELETE, MERGE, or DELTA) based on op_type.
        
        Uses the new single-candidate generators (_propose_delete_single, etc.) to avoid
        generating multiple candidates per operation. Each operation now maps directly
        to a single candidate DFA, reducing computational waste.
        
        The two target values provide independent control:
        - DELETE: uses target1_value to select state to delete; target2_value ignored
        - MERGE: uses target1_value and target2_value to independently select two states to merge
        - DELTA: uses target1_value as source state, target2_value as destination state
        
        Parameters
        ----------
        current_dfa : DFA
            Current DFA to modify
        op_type : str
            Operation type to apply: 'DELETE', 'MERGE', or 'DELTA'
        target1_value : float
            Primary continuous value for selection index
        target2_value : float
            Secondary continuous value for selection index (used in MERGE and DELTA)
            
        Returns
        -------
        tuple
            (modified_dfa, descriptor_string)
        """
        if self.learner is None:
            raise RuntimeError("Learner is required for PSO operations")

        next_dfa = None
        descriptor = ""
        n_states = len(current_dfa.states)

        try:
            # Decode both target values to state indices
            target1_idx = self._select_candidate_index(target1_value, n_states)
            target2_idx = self._select_candidate_index(target2_value, n_states)
            
            if op_type == "DELETE":
                # Delete the state at target1_idx
                next_dfa = self.learner._propose_delete_single(
                    current_dfa,
                    target1_idx,
                    self.data,
                    self.labels,
                    self.seen_signatures
                )
                descriptor = f"DELETE[s{target1_idx}]"
                    
            elif op_type == "MERGE":
                # Merge state at target1_idx with state at target2_idx (independent selection)
                # Ensure we don't merge a state with itself
                if target1_idx == target2_idx:
                    if n_states > 1:
                        target2_idx = (target1_idx + 1) % n_states
                    else:
                        # n_states=1: cannot merge single state
                        descriptor = f"MERGE✗INVALID(only_1_state)"
                        return deepcopy(current_dfa), descriptor
                
                next_dfa = self.learner._propose_merge_single(
                    current_dfa,
                    target1_idx,
                    target2_idx,
                    self.data,
                    self.labels,
                    self.seen_signatures
                )
                descriptor = f"MERGE[s{target1_idx}↔s{target2_idx}]"
                    
            elif op_type == "DELTA":
                # Add transition from state target1_idx to state target2_idx
                # More precise than fixed (target_idx, target_idx)
                next_dfa = self.learner._propose_delta_single(
                    current_dfa,
                    target1_idx,
                    target2_idx,
                    self.data,
                    self.labels,
                    self.seen_signatures
                )
                descriptor = f"DELTA[s{target1_idx}→s{target2_idx}]"
            else:
                # Invalid operation type
                descriptor = "INVALID_OP_TYPE"
            
            if next_dfa is not None:
                # DFA validity will be ensured in objective_function
                descriptor += "✓"
            else:
                # No valid candidate generated for this specific operation
                next_dfa = deepcopy(current_dfa)
                descriptor += "✗COPY"
                
        except Exception as exc:
            if self.verbose:
                print(f"  [WARN] Operation {op_type} failed: {str(exc)[:50]}")
            # Fallback: copy current
            next_dfa = deepcopy(current_dfa)
            descriptor = f"{op_type}✗COPY"

        # Sanity check
        if next_dfa is None:
            next_dfa = deepcopy(current_dfa)
            descriptor = f"{descriptor}→NULL_FALLBACK"
        
        return next_dfa, descriptor

    def _select_candidate_index(self, raw_value: float, num_candidates: int) -> int:
        if num_candidates <= 0:
            return 0
        normalized = self._normalize_to_unit_interval(raw_value)
        scaled = int(normalized * num_candidates)
        return min(max(scaled, 0), num_candidates - 1)

    def _normalize_to_unit_interval(self, raw_value: float) -> float:
        bounded = math.tanh(raw_value)
        normalized = (bounded + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))

    def optimize(self, 
                 n_particles: Optional[int] = None,
                 n_iterations: Optional[int] = None,
                 save_trajectory: bool = False) -> Dict[str, Any]:
        """
        Run PSO optimization to find minimal DFA.
        
        Parameters
        ----------
        n_particles : int, optional
            Override default number of particles
        n_iterations : int, optional
            Override default number of iterations
        save_trajectory : bool
            If True, return fitness trajectory for each iteration
            
        Returns
        -------
        dict
            Contains:
            - 'best_dfa': Best DFA found (minimal states, accuracy >= threshold)
            - 'best_accuracy': Accuracy of best DFA
            - 'best_states': Number of states in best DFA
            - 'best_loss': Loss value of best DFA
            - 'iterations': Number of iterations completed
            - 'trajectory': List of best loss per iteration (if save_trajectory=True)
        """
        if not PSO_AVAILABLE:
            raise ImportError("pyswarms is required for optimization")
        
        n_particles = n_particles or self.n_particles
        n_iterations = n_iterations or self.n_iterations
        
        if self.verbose:
            print(f"\n[PSO] Starting optimization...")
            print(f"  Particles: {n_particles}")
            print(f"  Iterations: {n_iterations}")
            print(f"  Initial states: {len(self.initial_dfa.states)}")
        
        # PSO configuration
        options = {
            'c1': self.c1,
            'c2': self.c2,
            'w': self.w,
            'k': n_particles,
            'p': 2,  # 2D neighborhood topology
        }
        
        # Determine dimensionality from slot encoding (op type, target, prob)
        dimensions = self.dimensions
        
        # Initialize PSO optimizer
        optimizer = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            init_pos=np.random.uniform(-1, 1, size=(n_particles, dimensions))
        )
        
        stop_reason = ""
        try:
            # Run PSO optimization
            # Note: pyswarms evaluates n_particles candidates per iteration
            # with max_evaluations budget distributed across iterations
            final_cost, final_pos = optimizer.optimize(
                self.objective_function,
                iters=n_iterations,
                verbose=self.verbose
            )
        except RuntimeError as e:
            stop_reason = str(e)
            if self.verbose:
                if "early stopping" in stop_reason.lower() or "budget exhausted" in stop_reason.lower():
                    print(f"[PSO] Optimization stopped: {stop_reason}")
                else:
                    print(f"[PSO WARNING] Optimization interrupted: {stop_reason}")
        except Exception as e:
            stop_reason = str(e)
            if self.verbose:
                print(f"[PSO WARNING] Optimization interrupted: {stop_reason}")
        
        if self.verbose:
            print(f"\n[PSO] Optimization complete!")
            print(f"  Best loss: {self.gbest_fitness if not np.isinf(self.gbest_fitness) else 'No valid solution found'}")
            print(f"  Best accuracy: {self.gbest_accuracy:.4f}")
            val_str = f", val_acc={self.gbest_val_accuracy:.4f}" if self.gbest_val_accuracy > 0 else ""
            best_states_str = f"{int(self.gbest_states)}" if not np.isinf(self.gbest_states) else "No valid solution found"
            print(f"  Best states: {best_states_str}{val_str}")
        
        # Safe conversion handling infinity
        best_states = int(self.gbest_states) if not np.isinf(self.gbest_states) else -1
        best_loss = float(self.gbest_fitness) if not np.isinf(self.gbest_fitness) else -1.0
        
        # Determine success and reason based on whether a valid solution was found
        success = not np.isinf(self.gbest_fitness) and self.gbest_dfa is not None
        if success:
            reason = f"PSO found solution with accuracy {self.gbest_accuracy:.4f} and {int(self.gbest_states)} states"
            if stop_reason:
                reason += f" (stopped: {stop_reason})"
        else:
            reason = "PSO failed to find valid solution (no automaton met validation criteria)"
            if stop_reason:
                reason += f" (stopped: {stop_reason})"
        
        result = {
            'best_dfa': self.gbest_dfa,
            'best_accuracy': float(self.gbest_accuracy),
            'best_validation_accuracy': float(self.gbest_val_accuracy),
            'best_states': best_states,
            'best_loss': best_loss,
            'iterations': n_iterations,
            'threshold': self.threshold,
            'all_history': self.all_history,  # Include all candidates for comparison
            'evaluations': int(self.evaluations_count),
            'max_evaluations': int(self.max_evaluations) if self.max_evaluations is not None else None,
            'success': success,                # Whether optimization found a valid solution
            'reason': reason,                  # Explanation of the result
        }
        
        if save_trajectory:
            result['trajectory'] = None  # pyswarms doesn't provide trajectory via callback
        
        return result


# ============================================================================
# Convenience function for easy integration
# ============================================================================

# def optimize_dfa_with_pso(initial_dfa,
#                           learner,
#                           data: List,
#                           labels: np.ndarray,
#                           threshold: float = 0.8,
#                           n_particles: int = 10,
#                           n_iterations: int = 20,
#                           verbose: bool = True) -> Dict[str, Any]:
#     """
#     Convenience function to quickly optimize a DFA using PSO.
    
#     Parameters
#     ----------
#     initial_dfa : Dfa
#         Starting DFA for optimization
#     learner : DFALearner
#         Learner instance with propose methods
#     data : list
#         Training sequences
#     labels : np.ndarray
#         Training labels (binary)
#     threshold : float
#         Minimum required accuracy (default: 0.8)
#     n_particles : int
#         Number of particles (default: 10)
#     n_iterations : int
#         Number of iterations (default: 20)
#     verbose : bool
#         Enable logging (default: True)
        
#     Returns
#     -------
#     dict
#         Optimization results with 'best_dfa', 'best_accuracy', 'best_states', etc.
#     """
#     optimizer = PSOAutomataOptimizer(
#         initial_dfa=initial_dfa,
#         threshold=threshold,
#         data=data,
#         labels=labels,
#         learner=learner,
#         n_particles=n_particles,
#         n_iterations=n_iterations,
#         verbose=verbose
#     )
    
#     return optimizer.optimize(save_trajectory=True)


__all__ = [
    'PSOAutomataOptimizer',
    'optimize_dfa_with_pso',
]
