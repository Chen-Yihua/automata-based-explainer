"""
DFA and sequence utility functions.
"""

import re
import matplotlib.pyplot as plt
from typing import Dict, Set, Tuple
from aalpy.automata.Dfa import Dfa, DfaState

# ==============================================================
# DFA Operations
# ==============================================================
def get_alphabet(dfa) -> Set[str]:
    """Get the alphabet of the dfa"""
    if hasattr(dfa, "alphabet"):
        return set(dfa.alphabet)
    elif hasattr(dfa, "get_input_alphabet"):
        return set(dfa.get_input_alphabet())
    else:
        syms = set()
        for s in getattr(dfa, "states", []):
            syms.update(getattr(s, "transitions", {}).keys())
        return syms

def is_valid_dfa(dfa) -> bool:
        """
        Check if DFA is valid (has >= 2 states and at least one accepting state).
        
        Parameters
        ----------
        dfa : Dfa
            DFA to validate
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if not hasattr(dfa, 'states'):
            return False
        states = list(dfa.states)
        return len(states) >= 2 and any(s.is_accepting for s in states)


def check_dfa_path_accepted(dfa, path) -> bool:
        """Check if path is accepted by DFA"""
        dfa = dfa[0] if isinstance(dfa, (list, tuple)) else dfa
        current = dfa.initial_state
        for symbol in path:
            next_state = current.transitions.get(symbol)
            if next_state is None:
                return False
            current = next_state
        return current.is_accepting


def dfa_product(dfa1: Dfa, dfa2: Dfa, final_func) -> Dfa:
    """
    Generic DFA product construction (for intersection/union).
    final_func: function that takes (accept1, accept2) -> bool (accepting)
    """
    alphabet = list(get_alphabet(dfa1) | get_alphabet(dfa2))
    state_map: Dict[Tuple[str, str], DfaState] = {}
    queue: list = []

    def get_id(s1, s2):
        return f"({s1.state_id},{s2.state_id})"

    initial_tuple = (dfa1.initial_state, dfa2.initial_state)
    initial_id = get_id(*initial_tuple)
    initial_accept = final_func(initial_tuple[0].is_accepting, initial_tuple[1].is_accepting)
    initial = DfaState(initial_id, initial_accept)
    state_map[(initial_tuple[0], initial_tuple[1])] = initial
    queue.append(initial_tuple)

    while queue:
        curr1, curr2 = queue.pop(0)
        curr = state_map[(curr1, curr2)]
        for a in alphabet:
            if a not in curr1.transitions or a not in curr2.transitions:
                continue
            next1 = curr1.transitions[a]
            next2 = curr2.transitions[a]
            next_tuple = (next1, next2)
            if next_tuple not in state_map:
                accept = final_func(next1.is_accepting, next2.is_accepting)
                node = DfaState(get_id(next1, next2), accept)
                state_map[next_tuple] = node
                queue.append(next_tuple)
            curr.transitions[a] = state_map[next_tuple]
    all_states = list(state_map.values())
    return Dfa(initial, all_states)


def dfa_intersection(dfa1: Dfa, dfa2: Dfa) -> Dfa:
    return dfa_product(dfa1, dfa2, lambda a1, a2: a1 and a2)


def dfa_union(dfa1: Dfa, dfa2: Dfa) -> Dfa:
    return dfa_product(dfa1, dfa2, lambda a1, a2: a1 or a2)


def remove_unreachable_states(dfa):
    """Remove unreachable states from DFA"""
    reachable = set()
    queue = [dfa.initial_state]
    while queue:
        s = queue.pop()
        if s not in reachable:
            reachable.add(s)
            for nxt in s.transitions.values():
                queue.append(nxt)
    dfa.states = [s for s in dfa.states if s in reachable]
    return dfa


def serialize_dfa(dfa) -> int:
    """Serialize DFA to hashable signature for deduplication"""
    items = []
    for s in sorted(dfa.states, key=lambda x: str(x.state_id)):
        trans = sorted([(sym, str(dst.state_id)) for sym, dst in s.transitions.items()])
        items.append((str(s.state_id), s.is_accepting, tuple(trans)))
    return hash(tuple(items))


def make_dfa_complete(dfa: Dfa, alphabet: list) -> Dfa:
    """Convert Partial DFA to Complete DFA by adding sink state"""
    sink_id = "sink_state"
    existing_ids = set(s.state_id for s in dfa.states)
    while sink_id in existing_ids:
        sink_id += "_"

    sink_state = DfaState(sink_id, is_accepting=False)
    
    for sym in alphabet:
        sink_state.transitions[sym] = sink_state

    added_sink = False
    for s in dfa.states:
        for sym in alphabet:
            if sym not in s.transitions:
                s.transitions[sym] = sink_state
                added_sink = True
    
    if added_sink:
        dfa.states.append(sink_state)
        
    return dfa


def trim_dfa(dfa):
    """Trim unreachable states from DFA"""
    alphabet = get_alphabet(dfa)
    start = dfa.initial_state

    reachable = {start}
    queue = [start]

    while queue:
        cur = queue.pop(0)
        for a in alphabet:
            if a in cur.transitions:
                nxt = cur.transitions[a]
                if nxt not in reachable:
                    reachable.add(nxt)
                    queue.append(nxt)

    state_map = {}
    for old in reachable:
        new_state = DfaState(old.state_id)
        new_state.is_accepting = old.is_accepting
        new_state.transitions = {}
        state_map[old] = new_state
    
    for old, new in state_map.items():
        for a in alphabet:
            if a in old.transitions:
                nxt = old.transitions[a]
                if nxt in state_map:
                    new.transitions[a] = state_map[nxt]

    return Dfa(
        states=set(state_map.values()),
        initial_state=state_map[start],
    )


def merge_linear_edges(dfa, learn_type=None):
    """
    Merge consecutive edges in the dfa.
    For TEXT type: if two consecutive edges are both wildcards (*), merge them.
    """
    def _is_wildcard(sym):
        if sym == "*":
            return True
        if isinstance(sym, str) and re.match(r"p\d+_\(\*\)", sym):
            return True
        return False
    
    def _get_position(sym):
        if not isinstance(sym, str):
            return None
        match = re.match(r"p(\d+)_", sym)
        return int(match.group(1)) if match else None
    
    states = set(dfa.states)
    changed = True
    while changed:
        changed = False
        for s in list(states):
            if s is dfa.initial_state or s.is_accepting:
                continue

            in_edges = []
            for p in states:
                for sym, q in p.transitions.items():
                    if q is s:
                        in_edges.append((p, sym))
                        if len(in_edges) > 1:
                            break
                if len(in_edges) > 1:
                    break

            out_edges = list(s.transitions.items())
            if len(in_edges) == 1 and len(out_edges) == 1:
                p, sym_in = in_edges[0] 
                sym_out, q = out_edges[0]

                if sym_in == sym_out:
                    p.transitions[sym_in] = q
                    states.remove(s)
                    dfa.states.remove(s)
                    changed = True
                    break
                
                if learn_type == "Text" and _is_wildcard(sym_in) and _is_wildcard(sym_out):
                    pos_in = _get_position(sym_in)
                    pos_out = _get_position(sym_out)
                    
                    if pos_in is not None and pos_out is not None and pos_out == pos_in + 1:
                        merged_label = f"p{pos_in}-{pos_out}_(*)"
                        del p.transitions[sym_in]
                        p.transitions[merged_label] = q
                        states.remove(s)
                        dfa.states.remove(s)
                        changed = True
                        break
                    
                    range_match = re.match(r"p(\d+)-(\d+)_\(\*\)", sym_in)
                    if range_match and pos_out is not None:
                        start_pos = int(range_match.group(1))
                        end_pos = int(range_match.group(2))
                        if pos_out == end_pos + 1:
                            merged_label = f"p{start_pos}-{pos_out}_(*)"
                            del p.transitions[sym_in]
                            p.transitions[merged_label] = q
                            states.remove(s)
                            dfa.states.remove(s)
                            changed = True
                            break
    return dfa


def merge_parallel_edges(dfa, learn_type):
    """Merge parallel edges in the dfa."""
    def _clean_symbol(sym: str) -> str:
        if learn_type in ("Text", "Image"):
            match = re.search(r"\(([\d\.\-]+)\)", sym)
            if match:
                return match.group(1)
            return sym
        return sym

    def _get_prefix(sym):
        if not isinstance(sym, str):
            return ""
        match = re.match(r"(p\d+)_\([\d\.\-\w]+\)", sym)
        return match.group(1) + "_" if match else ""

    def _get_position(sym):
        if not isinstance(sym, str):
            return None
        match = re.match(r"p(\d+)_", sym)
        return match.group(1) if match else None

    alphabet = set(_clean_symbol(sym) for sym in get_alphabet(dfa))

    for s in dfa.states:
        target_map = {}
        new_transitions = {}
        prefix_map = {}
        position_map = {}

        for sym, t in s.transitions.items():
            clean_sym = _clean_symbol(sym)
            new_transitions[clean_sym] = t
            if t != s:
                target_map.setdefault(t, set()).add(clean_sym)
                prefix = _get_prefix(sym)
                if prefix:
                    prefix_map[t] = prefix
                
                if learn_type == "Text":
                    pos = _get_position(sym)
                    if pos:
                        key = (t, pos)
                        position_map.setdefault(key, set()).add(sym)

        if learn_type == "Text":
            for (t, pos), symbols in position_map.items():
                if len(symbols) >= 2:
                    has_unk = any("UNK" in sym for sym in symbols)
                    has_word = any("UNK" not in sym for sym in symbols)
                    
                    if has_unk and has_word:
                        prefix = prefix_map.get(t, "")
                        wildcard = f"{prefix}(*)" if prefix else "*"
                        
                        for sym in symbols:
                            if sym in new_transitions:
                                del new_transitions[sym]
                        
                        new_transitions[wildcard] = t
            
            s.transitions = new_transitions
        else:
            for t, symbols in target_map.items():
                if symbols == alphabet:
                    s.transitions = new_transitions.copy()
                    for sym in list(symbols):
                        if sym in s.transitions and s.transitions[sym] is t:
                            del s.transitions[sym]

                    prefix = prefix_map.get(t, "")
                    if prefix:
                        s.transitions[f"{prefix}(*)"] = t
                    else:
                        s.transitions[f"*"] = t
    
    return dfa


def simplify_dfa(dfa, learn_type):
    """Simplify dfa by merging edges"""
    dfa = merge_parallel_edges(dfa, learn_type)
    dfa = merge_linear_edges(dfa, learn_type)
    return dfa


# ==============================================================
# DFA Visualization and Export
# ==============================================================
def dfa_to_mata(dfa, file_path):
    """Export DFA to Mata format for libMata - produces valid NFA-explicit format"""
    def _clean_state_name(name, index):
        """Create a valid mata identifier: q + index (stable mapping)"""
            # Use indexed names: q0, q1, q2, etc. for stability and libMata compatibility
        return f"q{index}"

    try:
        raw_states = list(getattr(dfa, 'states', []))
        init_state = getattr(dfa, 'initial_state', None)
        if isinstance(init_state, (list, tuple)):
            init_state = init_state[0] if init_state else None

        # Build a robust state pool using states + transition targets + initial state.
        state_by_id = {}
        for st in raw_states:
            sid = getattr(st, 'state_id', None)
            if sid is not None and sid not in state_by_id:
                state_by_id[sid] = st

        if init_state is not None:
            init_id = getattr(init_state, 'state_id', None)
            if init_id is not None and init_id not in state_by_id:
                state_by_id[init_id] = init_state

        for st in list(state_by_id.values()):
            for _, tgt in getattr(st, 'transitions', {}).items():
                tid = getattr(tgt, 'state_id', None)
                if tid is not None and tid not in state_by_id:
                    state_by_id[tid] = tgt

        # Last-resort synthetic single state if DFA object is severely malformed.
        if not state_by_id:
            synthetic = DfaState('q_synth', is_accepting=True)
            synthetic.transitions = {}
            state_by_id['q_synth'] = synthetic
            init_state = synthetic

        # Ensure we have a valid initial state that exists in state_by_id.
        if init_state is None or getattr(init_state, 'state_id', None) not in state_by_id:
            init_state = next(iter(state_by_id.values()))

        states = list(state_by_id.values())
        finals = [s for s in states if getattr(s, 'is_accepting', False)]
        if not finals:
            # Keep export valid even for intermediate malformed DFAs.
            init_state.is_accepting = True
            finals = [init_state]

        all_syms = sorted({sym for s in states for sym in getattr(s, 'transitions', {}).keys()})
        symbol_map = {sym: i for i, sym in enumerate(all_syms)}

        # Map each state to q0, q1, q2, etc.
        state_list = sorted(states, key=lambda s: str(getattr(s, 'state_id', '')))
        state_map = {s.state_id: _clean_state_name(s.state_id, i) for i, s in enumerate(state_list)}
        transition_map = {}

        # Build mata content
        mata_lines = ["@NFA-explicit"]
        init_name = state_map[init_state.state_id]
        mata_lines.append(f"%Initial {init_name}")
        finals_str = " ".join(state_map[s.state_id] for s in finals)
        mata_lines.append(f"%Final {finals_str}")

        transition_count = 0
        for s in states:
            src = state_map[s.state_id]
            for sym, tgt in sorted(getattr(s, 'transitions', {}).items(), key=lambda x: str(x[0])):  # Sort for consistency
                tgt_id = getattr(tgt, 'state_id', None)
                if tgt_id is None or tgt_id not in state_map:
                    continue
                tgt_name = state_map[tgt_id]
                sym_id = symbol_map[sym]
                mata_lines.append(f"{src} {sym_id} {tgt_name}")
                transition_map[transition_count] = sym
                transition_count += 1
        
        # Write mata file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mata_lines) + "\n")  # Ensure final newline
        
        print(f"  [MATA] Wrote {transition_count} transitions to {file_path}")
        
        # Create reverse mapping: cleaned_name -> original_id
        reverse_state_map = {v: k for k, v in state_map.items()}
        return state_map, symbol_map, transition_map, reverse_state_map
    
    except Exception as e:
        # Last-resort fallback: always write a minimal valid automaton file.
        # This prevents DELTA from crashing when intermediate DFAs are malformed.
        try:
            fallback_lines = [
                "@NFA-explicit",
                "%Initial q0",
                "%Final q0",
            ]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(fallback_lines) + "\n")
            print(f"  [MATA] WARNING: export fallback used due to error: {e}")
            return ({'q0': 'q0'}, {}, {}, {'q0': 'q0'})
        except Exception as inner_e:
            raise RuntimeError(f"Failed to export DFA to mata format: {e}; fallback failed: {inner_e}")
        
def dfa_to_graphviz(dfa, filename=None, show_sink=False, instance=None, output_dir="output") -> str:
        """
        Convert DFA to Graphviz DOT string and save visualization.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to visualize
        filename : str
            Output filename
        show_sink : bool
            Whether to show sink states
        instance : list or tuple, optional
            Original instance to highlight its path in the DFA with color
        output_dir : str
            Output directory
            
        Returns
        -------
        str
            DOT content string
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        def _clean_state_name(name):
            name = str(name)
            return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name)

        def _clean_label(label):
            return str(label).replace('"', "'")

        # draw path edges if instance is provided
        path_edges = set()
        if instance is not None:
            current = dfa.initial_state
            for i, symbol in enumerate(instance):
                if symbol in current.transitions:
                    next_state = current.transitions[symbol]
                    path_edges.add((_clean_state_name(current.state_id), _clean_state_name(next_state.state_id), _clean_label(symbol)))
                    current = next_state
                else:
                    break

        lines = ["digraph DFA {", "  rankdir=LR;", '  node [shape=circle];']
        start_name = _clean_state_name(dfa.initial_state.state_id)
        lines.append('  __start__ [shape=point];')
        lines.append(f'  __start__ -> "{start_name}";')

        for state in dfa.states:
            if not show_sink and hasattr(dfa, "sink") and state == dfa.sink:
                continue
            shape = "doublecircle" if state.is_accepting else "circle"
            lines.append(f'  "{_clean_state_name(state.state_id)}" [shape={shape}];')

        for state in dfa.states:
            for symbol, next_state in state.transitions.items():
                if not show_sink and hasattr(dfa, "sink") and (
                    state == dfa.sink or next_state == dfa.sink
                ):
                    continue
                src_name = _clean_state_name(state.state_id)
                dst_name = _clean_state_name(next_state.state_id)
                label_name = _clean_label(symbol)
                
                if (src_name, dst_name, label_name) in path_edges:
                    lines.append(f'  "{src_name}" -> "{dst_name}" [label="{label_name}", color=red, penwidth=2.5, fontcolor=red];')
                else:
                    lines.append(f'  "{src_name}" -> "{dst_name}" [label="{label_name}"];')

        lines.append("}")
        
        # Save to file
        dot_content = "\n".join(lines)
        if filename:
            dot_path = os.path.join(output_dir, filename)
            with open(dot_path, "w") as f:
                f.write(dot_content)

        return dot_content


def plot_beam_stats(iteration_stats, beam_size, output_dir="test_result/explain", show=False):
    """Save beam-search agreement and state-count plots."""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    if not iteration_stats:
        return

    def _values(step, *keys):
        for key in keys:
            if key in step:
                vals = step[key]
                if vals is None:
                    return []
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                out = []
                for val in vals:
                    while hasattr(val, "__len__") and not isinstance(val, str) and len(val) > 0:
                        val = val[0]
                    try:
                        out.append(float(val))
                    except Exception:
                        pass
                return out
        return []

    iterations = [int(step.get("iteration", i)) for i, step in enumerate(iteration_stats)]

    # Agreement plot
    train_best, val_best = [], []
    train_avg, val_avg = [], []
    valid_iters = []
    for i, step in zip(iterations, iteration_stats):
        train_vals = _values(step, "training_agreements", "training_agreement")
        val_vals = _values(step, "validation_agreements", "validation_agreement")
        if not train_vals:
            continue
        valid_iters.append(i)
        train_best.append(max(train_vals))
        train_avg.append(sum(train_vals) / len(train_vals))
        if val_vals:
            val_best.append(max(val_vals))
            val_avg.append(sum(val_vals) / len(val_vals))
        else:
            val_best.append(float("nan"))
            val_avg.append(float("nan"))

    if valid_iters:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(valid_iters, train_best, marker="o", label="Training best", linewidth=2)
        plt.plot(valid_iters, train_avg, marker="o", linestyle=":", label="Training avg", linewidth=1.5)
        if not all(v != v for v in val_best):
            plt.plot(valid_iters, val_best, marker="x", linestyle="--", label="Validation best", linewidth=2)
            plt.plot(valid_iters, val_avg, marker="x", linestyle=":", label="Validation avg", linewidth=1.5)
        plt.title("Agreement over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Agreement")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "agreement_over_iterations.png"), dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # State-count plot
    state_iters, state_min, state_avg = [], [], []
    for i, step in zip(iterations, iteration_stats):
        states = _values(step, "states")
        if not states:
            continue
        state_iters.append(i)
        state_min.append(min(states))
        state_avg.append(sum(states) / len(states))

    if state_iters:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(state_iters, state_min, marker="o", label="Min states", linewidth=2)
        plt.plot(state_iters, state_avg, marker="o", linestyle=":", label="Avg states", linewidth=1.5)
        plt.title("State count over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("States")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "states_over_iterations.png"), dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

