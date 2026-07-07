"""
Register Automata Learner

This module contains:
1. Register Automaton structure (RAState, RegisterAutomaton)
2. RA manipulation functions (clone, serialize, etc.)
3. RA learning algorithms
4. RA visualization

The structure is designed to be easily replaceable with external libraries
(e.g., RALib, LearnLib) in the future.
"""
import random
import numpy as np
from .base import BaseAutomataLearner
from external_modules.interpretera.src.local_serach_synthesis import dra_learner
from external_modules.interpretera.src.local_serach_synthesis.register_automata import DRA

class RASampler:
    def __init__(self, predictor, alphabet, base_sampler=None, seed=None, edit_distance=1):
        
        self.predictor = predictor
        self.tab_sampler = base_sampler
        self.alphabet = alphabet
        self.edit_distance = edit_distance
        self.instance = None
        self.instance_label = None
        self.n_covered_ex = 10

        if seed is not None:
            random.seed(seed)
        else:
            self.seed = seed

    def set_instance_label(self, X):
        self.tab_sampler.set_instance_label(X)
        self.instance = list(X[0]) if isinstance(X, (list, np.ndarray)) else list(X)
        self.instance_label = self.tab_sampler.instance_label

    def set_n_covered(self, n):
        self.n_covered_ex = n
        self.tab_sampler.set_n_covered(n)

    def build_lookups(self, data=None):
        """AnchorTabular 預期 sampler 有 build_lookups"""
        self._built = True
        return ({}, {}, {})
    
    def compare_labels(self, samples: np.ndarray):
        return self.tab_sampler.compare_labels(samples)

    def perturbation(self, num_samples, max_trials=5000):
        alphabet = self.alphabet
        edit_distance = self.edit_distance

        results = set()
        trials = 0
        while len(results) < num_samples and trials < max_trials:
            trials += 1
            if trials >= max_trials:
                break

            new_instance = list(self.instance).copy()
            op = random.choice(["replace", "insert", "delete"])
            edit_dist = min(edit_distance, len(new_instance)) if new_instance else 1

            if op == "replace":
                idxs = random.sample(range(len(new_instance)), edit_dist)
                for i in idxs:
                    if alphabet:
                        new_instance[i] = random.choice(alphabet)

            elif op == "insert":
                for _ in range(edit_dist):
                    pos = random.randint(0, len(new_instance))
                    if alphabet:
                        new_instance.insert(pos, random.choice(alphabet))

            elif op == "delete" and len(new_instance) > edit_dist:
                idxs = sorted(random.sample(range(len(new_instance)), edit_dist), reverse=True)
                for i in idxs:
                    del new_instance[i]

            if tuple(new_instance) != tuple(self.instance):
                results.add(tuple(new_instance))

        samples = [list(seq) for seq in results]
        return samples, samples
    
    def __call__(self, num_samples, compute_labels=True):
        raw_data, d_raw_data = self.perturbation(num_samples)

        if compute_labels:
            labels = self.compare_labels(raw_data)
            
            print(f"[RASampler DEBUG] instance_label = {self.instance_label}")
            print(f"[RASampler DEBUG] Generated {len(raw_data)} samples")
            print(f"[RASampler DEBUG] Labels: True={np.sum(labels)}, False={len(labels)-np.sum(labels)}")
            if len(raw_data) > 0:
                print(f"[RASampler DEBUG] First sample: {raw_data[0][:3]}... label={labels[0]}")

            # variable-length
            if getattr(self.tab_sampler, "d_train_data", np.array([])).dtype == object:
                covered_true = [seq for seq, lab in zip(raw_data, labels) if lab == 1][:self.n_covered_ex]
                covered_false = [seq for seq, lab in zip(raw_data, labels) if lab == 0][:self.n_covered_ex]
                covered_true = np.array(covered_true, dtype=object)
                covered_false = np.array(covered_false, dtype=object)
            else:
                covered_true = raw_data[labels][:self.n_covered_ex]
                covered_false = raw_data[np.logical_not(labels)][:self.n_covered_ex]

            return [raw_data, labels.astype(int)]
        else:
            return [d_raw_data]


class RegisterAutomataLearner(BaseAutomataLearner):
    """
    Register Automata Learner.
    Register Automata can store and compare data values in registers,
    making them more expressive than DFAs for certain data-dependent languages.
    """
    
    def __init__(self, num_states=6, num_registers=2, constants=None, theory="Integer", debug=False):
        super().__init__()
        self.num_states = num_states
        self.num_registers = num_registers
        self.constants = constants if constants is not None else [0, 1]
        self.theory = theory
        self.debug = debug
        self.learner = None
        self.dra = None
    

    def get_sampler(self):
        from learner.ra_learner import RASampler
        return RASampler

    
    def create_init_automata(self, data_type, positive_samples, negative_samples, golden_pos_samples=[]):
        """
        Create initial DRA using a simple but effective passive learning strategy.
        
        Strategy:
        1. Build a chain-like DRA where each state transitions to the next
        2. Use universal guards (TT) to accept all symbols initially
        3. Set final states based on positive sample length distribution
        4. This gives a reasonable baseline (~40-60% accuracy) for beam search
        
        Args:
            data_type: Type of data ('Tabular', 'Text', etc.)
            positive_samples: List of positive examples
            negative_samples: List of negative examples
            golden_pos_samples: Optional golden positive samples
        """
        # Create DRAlearner to get condition_map and assignment_map
        self.learner = dra_learner.DRAlearner(
            num_states=self.num_states,
            num_registers=self.num_registers,
            constants=self.constants,
            pos_samples=positive_samples,
            neg_samples=negative_samples,
            golden_pos_samples=golden_pos_samples,
            auto=None,
            theory=self.theory,
            debug=self.debug,
        )
        
        # Build a smarter initial DRA structure
        dra = self._build_passive_dra(positive_samples, negative_samples)
        self.dra = dra
        
        self.pairwise_det_map = getattr(self.learner, "pairwise_det_map", {})
        return self.dra
    

    def _build_passive_dra(self, positive_samples, negative_samples):
        """
        Build a generic initial DRA that serves as a good starting point for beam search.
        
        Strategy:
        - Create a chain structure: 0 → 1 → 2 → ... with various guards
        - State 0 (initial): Use TT to accept first symbol and initialize registers
        - Subsequent states: Use diverse guards to create exploration space
        - Let beam search refine the structure through DELETE/MERGE/DELTA operations
        """
        from external_modules.interpretera.src.local_serach_synthesis.register_automata import Operator, DoubleOperator
        
        # Get condition and assignment maps from learner
        condition_map = self.learner.condition_map
        assignment_map = self.learner.assignment_map
        
        all_conds = list(condition_map.keys())
        tt_cond_id = None
        useful_conds = []
        
        if self.theory == "Integer":
            for cid, (operand, op) in condition_map.items():
                if op == Operator.TT:
                    tt_cond_id = cid
                elif op != Operator.FF:
                    useful_conds.append(cid)
        else:  # Double theory
            for cid, (lower, op1, upper, op2) in condition_map.items():
                if lower == "nop" and upper == "nop":
                    tt_cond_id = cid
                else:
                    useful_conds.append(cid)
        
        # 找各種 assignments
        identity_assign_id = None
        curr_assign_ids = []  # r_i := curr
        
        for aid, assigns in assignment_map.items():
            is_identity = all(lhs == rhs for lhs, rhs in assigns)
            if is_identity:
                identity_assign_id = aid
            else:
                # 檢查是否有 r_i := curr
                for lhs, rhs in assigns:
                    if rhs == "curr":
                        curr_assign_ids.append(aid)
                        break
        
        if identity_assign_id is None:
            identity_assign_id = 0
        if not curr_assign_ids:
            curr_assign_ids = [identity_assign_id]
            
        print(f"[INIT DRA] Building generic initial DRA")
        print(f"  Theory: {self.theory}")
        print(f"  States: {self.num_states}")
        print(f"  TT condition: {tt_cond_id}")
        print(f"  Useful conditions: {len(useful_conds)}")
        print(f"  Curr assignments: {len(curr_assign_ids)}")
        
        # Build DRA structure - 通用策略
        states = set(range(self.num_states))
        initial_state = 0
        transitions = {}
        register_transitions = {}
        
        # 確保有 TT 條件
        if tt_cond_id is None:
            tt_cond_id = all_conds[0] if all_conds else 0
        
        for i in range(self.num_states):
            trans = []
            regs = []
            
            if i == 0:
                # State 0 (initial): 用 TT 接受第一個符號並初始化 register
                # 這對所有資料集都適用
                next_state = 1 if self.num_states > 1 else 0
                trans.append((tt_cond_id, next_state))
                regs.append((curr_assign_ids[0], next_state))
            else:
                # 其他狀態: 創建多條分支，使用不同的 guards
                # 這讓 beam search 有更多探索空間
                next_state_1 = (i + 1) % self.num_states
                next_state_2 = i  # 自環
                
                # 添加 2-3 條轉移，使用不同的 guards
                num_transitions = min(3, len(useful_conds)) if useful_conds else 1
                
                for j in range(num_transitions):
                    if useful_conds:
                        cond_idx = (i * num_transitions + j) % len(useful_conds)
                        cond_id = useful_conds[cond_idx]
                    else:
                        cond_id = tt_cond_id
                    
                    assign_idx = j % len(curr_assign_ids)
                    assign_id = curr_assign_ids[assign_idx]
                    
                    # 交替使用自環和前進
                    if j == 0:
                        trans.append((cond_id, next_state_2))  # 自環
                        regs.append((assign_id, next_state_2))
                    else:
                        trans.append((cond_id, next_state_1))  # 前進
                        regs.append((assign_id, next_state_1))
            
            transitions[i] = trans
            register_transitions[i] = regs
        
        # Final states: 根據樣本長度分佈決定
        # 讓多個狀態成為 final state，beam search 會精煉
        pos_lengths = [len(s) for s in positive_samples] if positive_samples else [1]
        neg_lengths = [len(s) for s in negative_samples] if negative_samples else [1]
        
        # 計算正樣本長度的分佈
        avg_pos_len = sum(pos_lengths) / len(pos_lengths)
        
        # 讓 state 1 和根據長度計算的狀態都是 final
        final_states = {1}  # State 1 通常是主要的處理狀態
        if self.num_states > 2:
            # 根據正樣本長度添加更多 final states
            for l in set(pos_lengths):
                final_states.add(l % self.num_states)
        
        # 確保至少有一個 non-final state (除了 state 0)
        if len(final_states) >= self.num_states - 1:
            # 保留 state 0 為 non-final，其他都是 final
            final_states.discard(0)
        
        print(f"[INIT DRA] Final states: {final_states}")
        print(f"[INIT DRA] Avg positive sample length: {avg_pos_len:.1f}")
        
        # Create DRA instance
        dra = DRA(
            states=states,
            num_registers=self.num_registers,
            constants=self.constants,
            condition_map=condition_map,
            assignment_map=assignment_map,
            transitions=transitions,
            register_transitions=register_transitions,
            initial_state=initial_state,
            final_states=final_states,
            theory=self.theory
        )
        
        return dra


    # def check_path_exist(self, dra, path) -> bool:
    #     try:
    #         current = dra.initial_state
    #         for symbol in path:
    #             found = False
    #             for (guard, action), next_state in current.transitions.items():
    #                 if hasattr(dra, '_evaluate_guard'):
    #                     if dra._evaluate_guard(guard, symbol, [None]*dra.num_registers):
    #                         current = next_state
    #                         found = True
    #                         break
    #                 else:
    #                     # fallback: accept all transitions
    #                     current = next_state
    #                     found = True
    #                     break
    #             if not found:
    #                 return False
    #         return True
    #     except Exception:
    #         return False


    def check_path_accepted(self, dra, path) -> bool:
        """
        Check if a path is accepted by the DRA.
        Uses the teacher's official accepts_input method from the DRA class.
        
        Args:
            dra: The DRA object (can be original or cloned/modified)
            path: List of input symbols (sequence)
        Returns:
            bool: True if the path is accepted by the automaton, False otherwise.
        """
        # Use debug=False to avoid teacher's decode bug with Double theory
        return dra.accepts_input(path, debug=False)
    
    
    def _evaluate_guard_tuple(self, guard_tuple, symbol, registers, constants):
        """
        Evaluate guard when condition_map contains tuples instead of functions.
        
        For Integer theory: guard_tuple is (lhs, Operator)
        For Double theory: guard_tuple is (lower, op1, upper, op2)
        
        Args:
            guard_tuple: Tuple representing the guard condition
            symbol: Current input symbol
            registers: Current register values
            constants: List of constants
            
        Returns:
            bool: True if guard is satisfied, False otherwise
        """
        from external_modules.interpretera.src.local_serach_synthesis.register_automata import Operator, DoubleOperator
        
        # Helper function to resolve operand value
        def resolve_value(operand):
            if operand is None:
                return None
            if isinstance(operand, (int, float)):
                return operand
            if operand == "nop":
                return None
            if operand.startswith("r"):
                reg_idx = int(operand[1:])
                return registers[reg_idx] if reg_idx < len(registers) else 0.0
            elif operand.startswith("c"):
                const_idx = int(operand[1:])
                return constants[const_idx] if const_idx < len(constants) else 0.0
            else:
                return None
        
        # Integer theory: (lhs, op)
        if len(guard_tuple) == 2:
            lhs, op = guard_tuple
            
            # Special cases: TT and FF
            if lhs is None and op == Operator.TT:
                return True
            if lhs is None and op == Operator.FF:
                return False
            
            # Regular comparison
            lhs_value = resolve_value(lhs)
            if lhs_value is None:
                return True  # Safe fallback
            
            if op == Operator.EQ:
                return lhs_value == symbol
            elif op == Operator.NEQ:
                return lhs_value != symbol
            else:
                return True  # Unknown operator, safe fallback
        
        # Double theory: (lower, op1, upper, op2)
        elif len(guard_tuple) == 4:
            lower, op1, upper, op2 = guard_tuple
            
            lower_value = resolve_value(lower)
            upper_value = resolve_value(upper)
            
            # Check lower bound
            lower_satisfied = True
            if lower_value is not None:
                if op1 == DoubleOperator.LT:
                    lower_satisfied = lower_value < symbol
                elif op1 == DoubleOperator.LEQ:
                    lower_satisfied = lower_value <= symbol
                elif op1 == DoubleOperator.EQ:
                    lower_satisfied = lower_value == symbol
                elif op1 == DoubleOperator.NEQ:
                    lower_satisfied = lower_value != symbol
            
            # Check upper bound
            upper_satisfied = True
            if upper_value is not None:
                if op2 == DoubleOperator.LT:
                    upper_satisfied = symbol < upper_value
                elif op2 == DoubleOperator.LEQ:
                    upper_satisfied = symbol <= upper_value
                elif op2 == DoubleOperator.EQ:
                    upper_satisfied = symbol == upper_value
                elif op2 == DoubleOperator.NEQ:
                    upper_satisfied = symbol != upper_value
            
            return lower_satisfied and upper_satisfied
        
        else:
            # Unknown format, safe fallback
            return True

    # def clone_automaton(self, dra):
    #     import copy
    #     return copy.deepcopy(dra)

    # def serialize_automaton(self, dra) -> int:
    #     return hash(str(dra))

    # def automaton_to_graphviz(self, dra, filename="dra_graph", output_dir="output") -> str:
    #     dra.visualize(filename, output_dir=output_dir)
    #     print("Proposed: ", dra)
    #     return f"{output_dir}/{filename}.dot.png"
    
    def _get_useful_assignments(self, ra, max_assignments=5):
        """
        Get the most useful assignment IDs to try in DELTA operations.
        
        Strategy:
        1. Always include identity assignment
        2. Include assignments that store current value (r_i := curr)
        3. Include assignments that copy between registers
        4. Prefer simpler assignments (fewer modifications)
        
        Returns:
            list: Assignment IDs to try
        """
        assignment_map = ra.assignment_map
        useful = []
        
        # 1. Identity assignment (always useful)
        identity_id = self._find_identity_assignment_id(ra)
        useful.append(identity_id)
        
        # Categorize assignments by type
        curr_assignments = []  # Store curr value (no constants)
        register_copy = []     # Copy between registers (no constants)
        other_assignments = [] # Other types
        
        for aid, assigns in assignment_map.items():
            if aid == identity_id:
                continue
            
            has_curr = False
            has_reg_copy = False
            has_constant = False
            
            for lhs, rhs in assigns:
                if rhs == "curr":
                    has_curr = True
                elif rhs != lhs and (rhs.startswith("r") if isinstance(rhs, str) else False):
                    has_reg_copy = True
                # 检测常量赋值：r_i := c_j 或 r_i := 数值
                elif isinstance(rhs, str) and rhs.startswith("c"):
                    has_constant = True
                elif isinstance(rhs, (int, float)):
                    has_constant = True
            
            # 只添加不含常量的赋值
            if has_constant:
                continue  # 跳过包含常量的赋值
            
            if has_curr:
                curr_assignments.append(aid)
            elif has_reg_copy:
                register_copy.append(aid)
            else:
                other_assignments.append(aid)
        
        # 2. Prioritize: curr assignments > register copies > others
        # This ensures we try semantically meaningful assignments first
        for aid in curr_assignments:
            if len(useful) >= max_assignments:
                break
            useful.append(aid)
        
        for aid in register_copy:
            if len(useful) >= max_assignments:
                break
            useful.append(aid)
        
        for aid in other_assignments:
            if len(useful) >= max_assignments:
                break
            useful.append(aid)
        
        # Debug output
        # if self.debug:
        #     print(f"    [DEBUG] Assignment selection:")
        #     print(f"      Identity: {identity_id}")
        #     print(f"      Curr-storing: {curr_assignments[:3]}")
        #     print(f"      Register-copy: {register_copy[:3]}")
        #     print(f"      Selected: {useful}")
        #     for aid in useful:
        #         print(f"        {aid}: {assignment_map[aid]}")
        
        return useful

    def _find_identity_assignment_id(self, ra):
        """
        Find the assignment ID that represents the identity assignment.
        """
        for aid, assigns in ra.assignment_map.items():
            ok = True
            for lhs, rhs in assigns:
                if lhs != rhs:
                    ok = False
                    break
            if ok:
                return aid
        raise RuntimeError("No identity assignment found in assignment_map")
    
    def _dedup_outgoing(self, ra, src, verbose=False):
        """
        Remove duplicate (guard, assignment, dst) transitions
        while preserving order.
        """
        seen = set()
        new_trans = []
        new_regs = []

        trans = ra.transitions.get(src, [])
        regs = ra.register_transitions.get(src, [])
        
        if len(trans) != len(regs):
            print(f"[DEDUP WARNING] State {src}: transitions({len(trans)}) != register_transitions({len(regs)})")

        if verbose:
            print(f"[DEDUP] State {src} BEFORE: {len(trans)} transitions")
        for i, ((cond, dst), (aid, _)) in enumerate(zip(trans, regs)):
            if verbose:
                print(f"  [{i}] guard={cond}, assign={aid}, dst={dst}")
            key = (cond, aid, dst)
            if key in seen:
                if verbose:
                    print(f"      ^ DUPLICATE! Skipping...")
                continue
            seen.add(key)
            new_trans.append((cond, dst))
            new_regs.append((aid, dst))
        
        if len(new_trans) < len(trans):
            print(f"[DEDUP] State {src}: Removed {len(trans) - len(new_trans)} duplicates ({len(trans)} -> {len(new_trans)})")
        elif verbose:
            print(f"[DEDUP] State {src}: No duplicates found ({len(new_trans)} transitions)")

        ra.transitions[src] = new_trans
        ra.register_transitions[src] = new_regs

    def _guards_compatible(self, g1, g2):
        """
        Return True iff g1 and g2 are NOT semantically disjoint.
        i.e., ∃ curr such that g1(curr) ∧ g2(curr)
        """
        if g1 == g2:
            return True
        # pairwise_det_map : deterministic guard
        return (g1, g2) not in self.pairwise_det_map and (g2, g1) not in self.pairwise_det_map

    def _has_guard_conflict(self, transitions):
        """
        transitions: List[(cond_id, dst)]
        Return True if any two transitions have same guard but different dst.
        """
        seen = {}
        for g, dst in transitions:
            if g in seen and seen[g] != dst:
                return True
            seen[g] = dst
        return False
    
    def _get_reachable_states(self, ra):
        """
        Find all states reachable from initial state via BFS.
        Returns set of reachable state IDs.
        """
        reachable = set()
        queue = [ra.initial_state]
        reachable.add(ra.initial_state)
        
        while queue:
            current = queue.pop(0)
            if current in ra.transitions:
                for cond_id, dst in ra.transitions[current]:
                    if dst not in reachable:
                        reachable.add(dst)
                        queue.append(dst)
        
        return reachable
    
    def _remove_unreachable_states(self, ra, verbose=False):
        """
        Remove states that are not reachable from initial state.
        Also removes transitions pointing to unreachable states.
        Returns True if any states were removed, False otherwise.
        """
        reachable = self._get_reachable_states(ra)
        unreachable = ra.states - reachable
        
        if unreachable:
            if verbose:
                print(f"    [CLEANUP] Removing {len(unreachable)} unreachable states: {unreachable}")
            
            # Remove unreachable states
            for state in unreachable:
                ra.states.discard(state)
                ra.transitions.pop(state, None)
                ra.register_transitions.pop(state, None)
                ra.final_states.discard(state)
            
            # Also remove transitions pointing TO unreachable states
            for src in list(ra.states):
                if src in ra.transitions:
                    old_trans = ra.transitions[src]
                    old_regs = ra.register_transitions.get(src, [])
                    
                    new_trans = []
                    new_regs = []
                    for i, (cond, dst) in enumerate(old_trans):
                        if dst in ra.states:  # Keep only if destination still exists
                            new_trans.append((cond, dst))
                            if i < len(old_regs):
                                new_regs.append(old_regs[i])
                    
                    if len(new_trans) < len(old_trans):
                        if verbose:
                            print(f"    [CLEANUP] State {src}: removed {len(old_trans) - len(new_trans)} edges to unreachable states")
                        ra.transitions[src] = new_trans
                        ra.register_transitions[src] = new_regs
            
            return True
        
        return False


    def _propose_delete(self, ra, state, data, labels, seen_signatures):
        """
        Propose new RAs by deleting states from the given RA.
        
        Parameters
        ----------
        ra : RegisterAutomaton
            The RA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen RA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new RAs created by deletion
        """
        print(f"\n[DEBUG _propose_delete] Starting with {len(ra.states)} states")
        new_ras = []
        ID_ASSIGN = self._find_identity_assignment_id(ra)

        for delete_state in list(ra.states):
            # Skip initial state and the last accepting state
            if delete_state == ra.initial_state:
                continue
            
            new_ra = self.clone_automaton(ra)
            print(f"Deleting state {delete_state}")

            # Redirect transitions pointing to successors or self loop
            # successors: outgoing transitions of deleted state
            successors = new_ra.transitions.get(delete_state, []) # delete_state → guard_out → successors

            # redirect incoming transitions pointing to delete_state
            for src in new_ra.states: # src: any state with outgoing transition
                new_trans = []
                new_regs = []

                trans = new_ra.transitions.get(src, []) # src → guard → dst
                regs = new_ra.register_transitions.get(src, []) # src → assignment → dst

                for i, (guard_in, dst) in enumerate(trans):
                    # outgoing transition NOT pointing to deleted state, retain guard and assignment
                    if dst != delete_state:
                        new_trans.append((guard_in, dst))
                        if i < len(regs):
                            a_id, _ = regs[i]
                            new_regs.append((a_id, dst))
                        continue
                    
                    # outgoing transition points to deleted state, need to redirect
                    for (guard_out, succ) in successors:
                        # If guards are compatible, redirect src --guard_out--> succ
                        if self._guards_compatible(guard_in, guard_out):
                            new_trans.append((guard_out, succ))
                            # Preserve original assignment from src
                            if i < len(regs):
                                a_id, _ = regs[i]
                                new_regs.append((a_id, succ))
                            else:
                                new_regs.append((ID_ASSIGN, succ))

                # remove duplicate transitions
                new_ra.transitions[src] = new_trans
                new_ra.register_transitions[src] = new_regs
                self._dedup_outgoing(new_ra, src)

                # semantic determinism check
                if self._has_guard_conflict(new_trans):
                    print(f"  [REJECT] State {src} has guard conflict after deleting {delete_state}")
                    break
            
            else:
                # Remove the state
                new_ra.states.remove(delete_state)
                new_ra.transitions.pop(delete_state, None)
                new_ra.register_transitions.pop(delete_state, None)
                new_ra.final_states.discard(delete_state)
                
                # Dedup all modified states to ensure no duplicates
                for st in new_ra.states:
                    if st in new_ra.transitions:
                        self._dedup_outgoing(new_ra, st)
                
                # 關鍵修復：移除不可達狀態（確保連通圖）
                self._remove_unreachable_states(new_ra, verbose=True)
                
                # 關鍵檢查：確保至少保留一個接受狀態和一個拒絕狀態
                if len(new_ra.final_states) == 0:
                    print(f"  [REJECT] Deleting {delete_state} would remove all accepting states")
                    continue
                if len(new_ra.final_states) == len(new_ra.states):
                    print(f"  [REJECT] Deleting {delete_state} would make all states accepting")
                    continue
                
                # 確保至少有2個狀態（刪除+清理後）
                if len(new_ra.states) < 2:
                    print(f"  [REJECT] Deleting {delete_state} would leave < 2 states after cleanup")
                    continue

                sig = self.serialize_automaton(new_ra)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    self.update_state_metrics(state, ra, new_ra, data, labels, "DELETE")
                    new_ras.append(new_ra)
                    print(f"  [SUCCESS] Deleted state {delete_state}, new RA added")
                else:
                    print(f"  [REJECT] Duplicate signature after deleting {delete_state}")
        
        print(f"[DEBUG _propose_delete] Generated {len(new_ras)} candidates")
        return new_ras
    
    def _can_merge_states(self, s1, s2):
        """
        Check if s1 and s2 can be merged without guard conflict.
        """
        out1 = self.ra.transitions.get(s1, [])
        out2 = self.ra.transitions.get(s2, [])

        guards1 = {g for g, _ in out1}
        guards2 = {g for g, _ in out2}

        # same guard to different dst is forbidden
        if guards1 & guards2:
            return False

        return True


    def _propose_merge(self, ra, state, data, labels, seen_signatures):
        """
        Propose new RAs by merging pairs of states.
        
        Parameters
        ----------
        ra : RegisterAutomaton
            The RA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen RA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new RAs created by merging states
        """
        import itertools
        print(f"\n[DEBUG _propose_merge] Starting with {len(ra.states)} states")
        new_ras = []
        # states = list(ra.states)
        # pairs = list(itertools.combinations(states, 2))

        for s1, s2 in itertools.combinations(list(ra.states), 2):
            if s1 == ra.initial_state or s2 == ra.initial_state:
                continue

            new_ra = self.clone_automaton(ra)
            print(f"Merging state {s2} into {s1}")
            
            # Redirect all transitions pointing to s2 to point to s1
            for st in new_ra.states:
                trans = new_ra.transitions.get(st, [])
                regs = new_ra.register_transitions.get(st, [])

                new_trans = []
                new_regs = []

                for i, (cond, dst) in enumerate(trans):
                    new_dst = s1 if dst == s2 else dst
                    new_trans.append((cond, new_dst))
                    if i < len(regs):
                        a_id, _ = regs[i]
                        new_regs.append((a_id, new_dst))

                new_ra.transitions[st] = new_trans
                new_ra.register_transitions[st] = new_regs
                self._dedup_outgoing(new_ra, st)

                # semantic determinism check
                if self._has_guard_conflict(new_trans):
                    print(f"  [REJECT] State {st} has guard conflict after merging {s1},{s2}")
                    break
            else:
            # merge outgoing transitions of s2 into s1
                if s2 in new_ra.transitions:
                    for i, (cond, dst) in enumerate(new_ra.transitions[s2]):
                        new_ra.transitions.setdefault(s1, []).append((cond, dst))
                        if s2 in new_ra.register_transitions:
                            a_id, _ = new_ra.register_transitions[s2][i]
                            new_ra.register_transitions.setdefault(s1, []).append((a_id, dst))
                
                # IMPORTANT: Dedup s1's transitions after merging
                self._dedup_outgoing(new_ra, s1)
                
                # Check for guard conflicts on s1 after merge
                if self._has_guard_conflict(new_ra.transitions.get(s1, [])):
                    print(f"  [REJECT] State {s1} has guard conflict after merging s2 transitions")
                    continue
                
                # merge final state
                if s2 in new_ra.final_states:
                    new_ra.final_states.add(s1)
                
                # Remove s2
                new_ra.states.remove(s2)
                new_ra.transitions.pop(s2, None)
                new_ra.register_transitions.pop(s2, None)
                new_ra.final_states.discard(s2)
                
                # Final dedup pass on all states to ensure no duplicates anywhere
                for st in new_ra.states:
                    if st in new_ra.transitions:
                        self._dedup_outgoing(new_ra, st)
                
                # 關鍵修復：移除不可達狀態（確保連通圖）
                self._remove_unreachable_states(new_ra, verbose=True)
                
                # 關鍵檢查：確保至少保留一個接受狀態和一個拒絕狀態
                if len(new_ra.final_states) == 0:
                    print(f"  [REJECT] Merging {s1},{s2} would remove all accepting states")
                    continue
                if len(new_ra.final_states) == len(new_ra.states):
                    print(f"  [REJECT] Merging {s1},{s2} would make all states accepting")
                    continue
                
                # 確保至少有2個狀態（合併+清理後）
                if len(new_ra.states) < 2:
                    print(f"  [REJECT] Merging {s1},{s2} would leave < 2 states after cleanup")
                    continue

                sig = self.serialize_automaton(new_ra)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    self.update_state_metrics(state, ra, new_ra, data, labels, "MERGE")
                    new_ras.append(new_ra)
                    print(f"  [SUCCESS] Merged {s2} into {s1}, new RA added")
                else:
                    print(f"  [REJECT] Duplicate signature after merging {s1},{s2}")
        
        print(f"[DEBUG _propose_merge] Generated {len(new_ras)} candidates")
        return new_ras
    
    def get_transition_trace(self, ra, path):
        """
        Execute the RA on a concrete input path and return the exact transition trace.

        Each element:
        (src_state, cond_id, assign_id, dst_state, symbol)
        """
        trace = []
        current = ra.initial_state
        ra.reset_registers()

        for symbol in path:
            taken = False

            # transitions: Dict[state, List[(cond_id, next_state)]]
            for idx, (cond_id, dst) in enumerate(ra.transitions.get(current, [])):

                if ra._evaluate_condition(cond_id, symbol):
                    assign_id = None

                    # register_transitions: Dict[state, List[(assign_id, next_state)]]
                    if current in ra.register_transitions:
                        for a_id, a_dst in ra.register_transitions[current]:
                            if a_dst == dst:
                                assign_id = a_id
                                break

                    trace.append((current, cond_id, assign_id, dst, symbol))

                    if assign_id is not None:
                        ra._execute_assignment(assign_id, symbol)

                    current = dst
                    taken = True
                    break

            if not taken:
                break

        return trace


    def replace_guard(self, ra, src, old_cond, new_cond, dst):
        """
        Replace ONE transition
        """
        new_trans = []
        new_reg_trans = []

        for i, (cid, nxt) in enumerate(ra.transitions[src]):
            reg = None
            if src in ra.register_transitions:
                reg = ra.register_transitions[src][i]

            if cid == old_cond and nxt == dst:
                new_trans.append((new_cond, nxt))
                if reg:
                    new_reg_trans.append(reg)
            else:
                new_trans.append((cid, nxt))
                if reg:
                    new_reg_trans.append(reg)

        ra.transitions[src] = new_trans
        if src in ra.register_transitions:
            ra.register_transitions[src] = new_reg_trans

    def neighbor_guards(self, cond_id):
        """
        Return semantic neighbors of a guard.
        """
        base = self.condition_map[cond_id]
        neighbors = []

        for cid, g in self.condition_map.items():
            if cid == cond_id:
                continue
            if type(g) == type(base):
                neighbors.append(cid)

        return neighbors[:2]

    def check_path_with_disabled_indices(self, ra, path, disabled_indices: set) -> bool:
        """
        Execute RA on path, but skip transitions whose trace index is in disabled_indices.
        This simulates what happens if we remove those transitions.
        
        IMPORTANT: disabled_indices refers to SYMBOL indices (0, 1, 2, ..., len(path)-1),
        NOT transition indices within a state.
        
        If a symbol's index is disabled, we skip the transition that would normally fire,
        causing the automaton to potentially get stuck or take an alternative path.
        """
        ra.reset_registers()
        current = ra.initial_state

        for symbol_idx, symbol in enumerate(path):
            taken = False

            for cond_id, dst in ra.transitions.get(current, []):
                if ra._evaluate_condition(cond_id, symbol):
                    # Skip this transition if this symbol position is disabled
                    if symbol_idx in disabled_indices:
                        # 禁用這條轉移：嘗試這個狀態的其他轉移（如果有的話）
                        continue
                    
                    # Apply assignment
                    if current in ra.register_transitions:
                        for a_id, a_dst in ra.register_transitions[current]:
                            if a_dst == dst:
                                ra._execute_assignment(a_id, symbol)
                                break

                    current = dst
                    taken = True
                    break

            if not taken:
                # 沒有可用的轉移（可能是被禁用了，或者本來就沒有匹配的）
                return False

        return current in ra.final_states

    def find_cxp_indices(self, ra, path, trace_len, original_accept):
        """
        Find minimal set of transition indices whose disabling flips the result.
        """
        # try size-1
        for i in range(trace_len):
            if self.check_path_with_disabled_indices(ra, path, {i}) != original_accept:
                return {i}

        # try size-2 (optional but recommended)
        for i in range(trace_len):
            for j in range(i + 1, trace_len):
                if self.check_path_with_disabled_indices(ra, path, {i, j}) != original_accept:
                    return {i, j}

        return None


    def _propose_delta(self, ra, state, data, labels, seen_signatures):
        """
        Propose new RAs by modifying transitions based on CXP (Contrastive Explanation).
        
        Following DFA learner pattern:
        1. Find counterexamples using automaton's accept/reject
        2. Select shortest counterexample path as test_word
        3. Use CXP-like analysis to find critical transition positions
        4. Redirect transitions at those positions to all possible target states
        
        Parameters
        ----------
        ra : RegisterAutomaton
            The RA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen RA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new RAs created by transition modification
        """
        print(f"\n[DEBUG _propose_delta] Starting")
        new_ras = [ra]
        ID_ASSIGN = self._find_identity_assignment_id(ra)
        
        # Step 1: Find misclassified paths (counterexamples)
        accepts = np.array([self.check_path_accepted(ra, p) for p in data])
        false_accept_indices = np.where((labels == 0) & (accepts == True))[0]
        true_reject_indices = np.where((labels == 1) & (accepts == False))[0]
        
        if len(false_accept_indices) == 0 and len(true_reject_indices) == 0:
            print("[DEBUG _propose_delta] No counterexamples found, skipping CXP-guided DELTA.")
            return new_ras
        
        # Step 2: Select shortest counterexample as test path
        false_accept_paths = [data[i] for i in false_accept_indices]
        true_reject_paths = [data[i] for i in true_reject_indices]
        candidate_paths = false_accept_paths + true_reject_paths
        test_path = min(candidate_paths, key=len)
        expected_label = 1 if test_path in true_reject_paths else 0
        
        print(f"[DEBUG _propose_delta] Test path: {test_path[:5]}... (len={len(test_path)}, expected={expected_label})")
        
        # Step 3: Get execution trace
        trace = self.get_transition_trace(ra, test_path)
        if not trace:
            print(f"[DEBUG _propose_delta] No trace found for test path")
            return new_ras
        
        print(f"[DEBUG _propose_delta] Trace length: {len(trace)}")
        
        # Step 4: Get CXP indices - find critical transition positions
        # Following Explaining-FA pattern: find groups of indices that cause prediction flip
        cxp_groups = self._get_cxp_indices_explaining_fa_style(ra, test_path, expected_label)
        
        # FALLBACK: If no CXP found, use all trace positions
        # This ensures exhaustive search happens even when CXP analysis fails
        if not cxp_groups:
            print("[DEBUG _propose_delta] No CXP indices found, using all trace positions as fallback")
            # Use all positions in the trace
            cxp_groups = [[i] for i in range(len(trace))]
        
        print(f"[DEBUG _propose_delta] Found {len(cxp_groups)} CXP groups")
        
        # Step 5: 針對 CXP 指向的具體轉移進行修改
        # 關鍵：只修改 CXP 識別出的那條轉移，而非窮舉狀態的所有轉移
        
        for cxp_group in cxp_groups:
            for idx in cxp_group:
                if idx >= len(trace):
                    continue
                
                # 從 trace 獲取這條轉移的精確信息
                src, old_cond_id, old_assign_id, old_dst, sym = trace[idx]
                print(f"\n  [CXP-DELTA] Position {idx}: state {src} --[guard={old_cond_id}]--> {old_dst}")
                
                # 找到這條轉移在 transitions 列表中的索引
                trans_list = ra.transitions.get(src, [])
                regs_list = ra.register_transitions.get(src, [])
                
                trans_idx = None
                for i, (g, d) in enumerate(trans_list):
                    if g == old_cond_id and d == old_dst:
                        trans_idx = i
                        break
                
                if trans_idx is None:
                    print(f"    [WARN] Could not find transition in state {src}")
                    continue
                
                old_assign = regs_list[trans_idx][0] if trans_idx < len(regs_list) else ID_ASSIGN
                
                # 策略 A: 改變目標狀態
                for new_dst in ra.states:
                    if new_dst == old_dst:
                        continue
                    new_ra = self.clone_automaton(ra)
                    new_trans = list(new_ra.transitions[src])
                    new_regs = list(new_ra.register_transitions[src])
                    new_trans[trans_idx] = (old_cond_id, new_dst)
                    new_regs[trans_idx] = (old_assign, new_dst)
                    new_ra.transitions[src] = new_trans
                    new_ra.register_transitions[src] = new_regs
                    self._dedup_outgoing(new_ra, src)
                    
                    if self._has_guard_conflict(new_ra.transitions.get(src, [])):
                        continue
                    sig = self.serialize_automaton(new_ra)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        self.update_state_metrics(state, ra, new_ra, data, labels, f"DELTA-DST(p{idx})")
                        new_ras.append(new_ra)
                
                # 策略 B: 改變 guard（只嘗試語義相近的）
                from external_modules.interpretera.src.local_serach_synthesis.register_automata import Operator
                
                related_guards = []
                old_guard_def = ra.condition_map.get(old_cond_id)
                for gid, guard_def in ra.condition_map.items():
                    if gid == old_cond_id:
                        continue
                    # 排除 FF (False) 條件
                    if len(guard_def) == 2 and guard_def[1] == Operator.FF:
                        continue
                    related_guards.append(gid)
                
                # 限制數量避免爆炸
                for new_guard in related_guards[:8]:
                    new_ra = self.clone_automaton(ra)
                    new_trans = list(new_ra.transitions[src])
                    new_regs = list(new_ra.register_transitions[src])
                    new_trans[trans_idx] = (new_guard, old_dst)
                    new_ra.transitions[src] = new_trans
                    new_ra.register_transitions[src] = new_regs
                    self._dedup_outgoing(new_ra, src)
                    
                    if self._has_guard_conflict(new_ra.transitions.get(src, [])):
                        continue
                    sig = self.serialize_automaton(new_ra)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        self.update_state_metrics(state, ra, new_ra, data, labels, f"DELTA-GUARD(p{idx})")
                        new_ras.append(new_ra)
                
                # 策略 C: 改變 assignment
                useful_assignments = self._get_useful_assignments(ra, max_assignments=5)
                for new_assign in useful_assignments:
                    if new_assign == old_assign:
                        continue
                    new_ra = self.clone_automaton(ra)
                    new_trans = list(new_ra.transitions[src])
                    new_regs = list(new_ra.register_transitions[src])
                    new_regs[trans_idx] = (new_assign, old_dst)
                    new_ra.transitions[src] = new_trans
                    new_ra.register_transitions[src] = new_regs
                    
                    sig = self.serialize_automaton(new_ra)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        self.update_state_metrics(state, ra, new_ra, data, labels, f"DELTA-ASSIGN(p{idx})")
                        new_ras.append(new_ra)
        
        print(f"  [CXP-DELTA] Generated {len(new_ras)-1} candidates from CXP positions")
        
        # CXP 導向的狀態屬性修正：直接調整終點狀態的 final 屬性
        original_accept = self.check_path_accepted(ra, test_path)
        
        # 情況 1：應該接受卻拒絕了 → 嘗試將終點狀態加入 final_states
        if expected_label == 1 and not original_accept:
            if trace:
                end_state = trace[-1][3]  # dst_state of last transition
                
                if end_state not in ra.final_states:
                    new_ra = self.clone_automaton(ra)
                    new_ra.final_states.add(end_state)
                    
                    sig = self.serialize_automaton(new_ra)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        self.update_state_metrics(state, ra, new_ra, data, labels, f"DELTA-ADD-FINAL(s{end_state})")
                        new_ras.append(new_ra)
                        print(f"\n[CXP STATE MOD] Added state {end_state} to final_states (should accept but rejected)")
        
        # 情況 2：應該拒絕卻接受了 → 嘗試將終點狀態移出 final_states
        elif expected_label == 0 and original_accept:
            if trace:
                end_state = trace[-1][3]
                
                if end_state in ra.final_states and len(ra.final_states) > 1:
                    new_ra = self.clone_automaton(ra)
                    new_ra.final_states.remove(end_state)
                    
                    sig = self.serialize_automaton(new_ra)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        self.update_state_metrics(state, ra, new_ra, data, labels, f"DELTA-REMOVE-FINAL(s{end_state})")
                        new_ras.append(new_ra)
                        print(f"\n[CXP STATE MOD] Removed state {end_state} from final_states (should reject but accepted)")
        
        # Final pass: Filter out disconnected RAs
        valid_ras = []
        for candidate_ra in new_ras:
            reachable = self._get_reachable_states(candidate_ra)
            # Ensure all states are reachable from initial state
            if reachable == candidate_ra.states:
                valid_ras.append(candidate_ra)
            else:
                # Log disconnected RA
                unreachable = candidate_ra.states - reachable
                print(f"    [FILTER] Removed RA with unreachable states: {unreachable}")
        
        print(f"[DEBUG _propose_delta] Total generated: {len(valid_ras)} valid candidates (filtered from {len(new_ras)})")
        
        return valid_ras
    
    def _get_cxp_indices_explaining_fa_style(self, ra, path, expected_label):
        """
        Find CXP (Contrastive Explanation) indices for a path - positions where
        redirecting transitions could flip the automaton's decision.
        
        This follows the Explaining-FA pattern adapted for RA:
        - For false accepts (expected=0, actual=1): Find transitions that lead to accept
        - For false rejects (expected=1, actual=0): Find transitions that prevent accept
        
        Returns
        -------
        list of list
            Groups of indices, e.g., [[0], [2, 5], [7]] where each group represents
            a set of transition positions that together could flip the decision
        """
        trace = self.get_transition_trace(ra, path)
        if not trace:
            print("    [CXP] No trace found")
            return []
        
        original_accept = self.check_path_accepted(ra, path)
        print(f"    [CXP] original_accept={original_accept}, expected_label={expected_label}")
        print(f"    [CXP] Trace: {[(src, dst) for src, cond_id, assign_id, dst, sym in trace[:5]]}...")
        
        cxp_groups = []
        
        # Strategy 1: Try single positions first (size-1 CXPs)
        print(f"    [CXP] Testing size-1 CXPs (trace_len={len(trace)})")
        for i in range(len(trace)):
            # Check if disabling this transition flips the result
            result_with_disabled = self.check_path_with_disabled_indices(ra, path, {i})
            if result_with_disabled != original_accept:
                print(f"      [CXP] Found size-1: index {i} flips {original_accept} -> {result_with_disabled}")
                cxp_groups.append([i])
                if len(cxp_groups) >= 5:  # Early exit if we have enough
                    break
        
        # Strategy 2: If no size-1 CXPs, try pairs (size-2 CXPs)
        if not cxp_groups:
            print(f"    [CXP] No size-1 found, testing size-2 CXPs")
            for i in range(len(trace)):
                for j in range(i + 1, min(i + 3, len(trace))):  # Only try nearby pairs
                    result_with_disabled = self.check_path_with_disabled_indices(ra, path, {i, j})
                    if result_with_disabled != original_accept:
                        print(f"      [CXP] Found size-2: indices ({i},{j}) flip {original_accept} -> {result_with_disabled}")
                        cxp_groups.append([i, j])
                        if len(cxp_groups) >= 3:  # Limit size-2 groups
                            break
                if len(cxp_groups) >= 3:
                    break
        
        print(f"    [CXP] Total groups found: {len(cxp_groups)}")
        # Limit total groups to avoid explosion
        return cxp_groups[:5]

    # def _propose_random_delta(self, ra, state, data, labels, seen_signatures):
    #     """
    #     Local search style delta operator: for each counterexample, randomly select a transition and randomly mutate it (destination or guard/assignment).
    #     No score computation, just random mutation as in local search.
    #     """
    #     import random
    #     new_ras = []
    #     ID_ASSIGN = self._find_identity_assignment_id(ra)

    #     for src in ra.states:
    #         trans = ra.transitions.get(src, [])

    #         for i, (cond, dst) in enumerate(trans):
    #             new_ra = self.clone_automaton(ra)

    #             mutate = random.choice(["dst", "guard"])

    #             if mutate == "dst":
    #                 candidates = [s for s in ra.states if s != dst]
    #                 if not candidates:
    #                     continue
    #                 new_dst = random.choice(candidates)
    #                 new_cond = cond
    #             else:
    #                 all_conds = list(ra.condition_map.keys())
    #                 new_cond = random.choice([c for c in all_conds if c != cond])
    #                 new_dst = dst

    #             # rebuild atomically
    #             new_trans = []
    #             new_regs = []

    #             for j, (c, d) in enumerate(new_ra.transitions[src]):
    #                 if j == i:
    #                     new_trans.append((new_cond, new_dst))
    #                     new_regs.append((ID_ASSIGN, new_dst))
    #                 else:
    #                     new_trans.append((c, d))
    #                     if src in new_ra.register_transitions:
    #                         a_id, _ = new_ra.register_transitions[src][j]
    #                         new_regs.append((a_id, d))

    #             new_ra.transitions[src] = new_trans
    #             new_ra.register_transitions[src] = new_regs
    #             self._dedup_outgoing(new_ra, src)

    #             sig = self.serialize_automaton(new_ra)
    #             if sig not in seen_signatures:
    #                 seen_signatures.add(sig)
    #                 self.update_state_metrics(state, ra, new_ra, data, labels, "DELTA")
    #                 new_ras.append(new_ra)

    #     return new_ras
    
    
    def propose_automata(self, ra_list, state, sample_fcn, iteration, previous_best: list, data_type: str = 'Tabular'):
        """
        Propose new Register Automata candidates by expanding existing RAs.
        
        This method coordinates three strategies:
        - DELETE: Remove states from RA (even iterations)
        - MERGE: Combine pairs of states (even iterations)
        - DELTA: Modify transitions (odd iterations)
        
        Parameters
        ----------
        ra_list : list
            List of current RAs
        state : dict
            State dictionary for tracking metrics
        sample_fcn : object
            Sampling function with feature values
        iteration : int
            Current iteration number
        previous_best : list
            List of best RAs from previous iteration
        data_type : str
            Type of data ('Tabular', 'Text', etc.)
            
        Returns
        -------
        list
            List of proposed new RAs
        """
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]
        
        # Initialize metrics for first iteration
        if iteration == 0:
            for ra in ra_list:
                accepts = np.array([self.check_path_accepted(ra, p) for p in data])
                true_accept = np.sum((labels == 1) & (accepts == True))
                false_reject = np.sum((labels == 0) & (accepts == False))
                
                ra_id = id(ra)
                state['t_nsamples'][ra_id] = float(len(data))
                state['t_accepted'][ra_id] = float(np.sum(accepts))
                state['t_positives'][ra_id] = float(true_accept)
                state['t_negatives'][ra_id] = float(false_reject)
                state['t_order'][ra_id].append(ra_id)
                
                print("--------------------------------------------")
                print(f"Proposed RA ID: {ra_id}")
                # print(self.automaton_to_graphviz(ra))
            
            return ra_list
        
        seen_signatures = set()
        new_ras = []
        
        print(f"\n{'='*60}")
        print(f"[propose_automata] Iteration {iteration}, processing {len(previous_best)} RAs")
        print(f"{'='*60}")
        
        for idx, ra in enumerate(previous_best):
            print(f"\n--- Processing RA {idx+1}/{len(previous_best)} ---")
            if iteration % 2 == 0:
                # Even iterations: DELETE and MERGE
                delete_candidates = self._propose_delete(ra, state, data, labels, seen_signatures)
                merge_candidates = self._propose_merge(ra, state, data, labels, seen_signatures)
                new_ras.extend(delete_candidates)
                new_ras.extend(merge_candidates)
                print(f"Summary: DELETE={len(delete_candidates)}, MERGE={len(merge_candidates)}")
            else:
                # Odd iterations: DELTA
                delta_candidates = self._propose_delta(ra, state, data, labels, seen_signatures)
                new_ras.extend(delta_candidates)
                print(f"Summary: DELTA={len(delta_candidates)}")
        
        print(f"\n{'='*60}")
        print(f"[propose_automata] Total candidates generated: {len(new_ras)}")
        print(f"{'='*60}\n")
        return new_ras


# ==============================================================
# Module-level exports
# ==============================================================

__all__ = [
    # Learner class
    'RASampler',
    'RegisterAutomataLearner',
    # Base classes (for external library wrappers)
    'BaseState',
    'BaseAutomaton',
    # Operations
    'clone_ra',
    'serialize_ra',
    'ra_to_graphviz',
    'check_ra_path_accepted',
    'check_ra_path_exist',
    'merge_ra_states',
    'delete_ra_state',
    # Flag
    'USING_EXTERNAL_LIBRARY',
]
