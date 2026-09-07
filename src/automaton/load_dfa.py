"""
Load DFA from DOT files in the automata/ folder as a teacher(black box model).
Provides utilities to parse Graphviz DOT format and convert to aalpy DFA objects.
"""

import os
import re
from typing import Dict, List, Tuple
import numpy as np
from aalpy import Dfa, DfaState

from automaton.dfa_utils import get_alphabet

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
AUTOMATA_DIR = os.path.join(PROJECT_ROOT, 'automata')

# Mapping of dot filenames to language names and metadata. `states` is
# intentionally not stored here -- it drifted out of sync with the actual
# .dot files (hand-counted once, never updated as the files were edited) and
# nothing in the codebase reads a static count for anything that matters.
# list_available_automata() below computes it fresh from each file instead.
AUTOMATA_MAPPING = {
    'secure_handshake.dot': {
        'name': 'SecureHandshake',
        'description': 'Secure handshake protocol DFA',
    },
    'multi_obligation_color_order.dot': {
        'name': 'MultiObligationOrder',
        'description': 'Multi-obligation color ordering',
    },
    'document_release_workflow.dot': {
        'name': 'DocumentReleaseWorkflow',
        'description': 'Document release workflow',
    },
    'lexer_tokenization.dot': {
        'name': 'LexerTokenization',
        'description': 'Lexer tokenization DFA',
    },
    'embedded_controller_workflow.dot': {
        'name': 'EmbeddedControllerWorkflow',
        'description': 'Embedded controller workflow DFA',
    }
}

# Reverse mapping: name -> filename
NAME_TO_FILE = {v['name']: k for k, v in AUTOMATA_MAPPING.items()}


def parse_dot_file(dot_path: str) -> Tuple[Dict[str, DfaState], str, set]:
    """
    Parse a Graphviz DOT file and convert to aalpy DFA format.
    Handles both quoted and unquoted state names.
    
    Parameters
    ----------
    dot_path : str
        Path to the .dot file
    
    Returns
    -------
    tuple: (states_dict, initial_state_name, alphabet)
        - states_dict: Dict[state_id, DfaState]
        - initial_state_name: name of initial state
        - alphabet: set of all symbols used in transitions
    
    Raises
    ------
    FileNotFoundError
        If dot_path does not exist
    ValueError
        If dot file cannot be parsed or is malformed
    """
    if not os.path.exists(dot_path):
        raise FileNotFoundError(f"DOT file not found: {dot_path}")
    
    with open(dot_path, 'r') as f:
        content = f.read()
    
    states_dict = {}
    initial_state_name = None
    alphabet = set()
    
    # Helper to extract state name from potentially quoted format
    def extract_state_name(s):
        """Extract unquoted state name from 'state' or state format."""
        return s.strip('"')
    
    # Extract all explicitly defined states (with shape attribute).
    # `[^\]]*` after the shape lets this also match nodes with extra
    # Graphviz attributes, e.g. [shape=doublecircle, style=filled, ...] --
    # matching only [shape=xxx] silently dropped any such node down to the
    # non-accepting default below, whatever its real shape said.
    explicit_states = {}  # state_name -> is_accepting
    state_pattern = r'(")?(\w+)\1?\s*\[shape=([a-z]+)[^\]]*\]'

    for match in re.finditer(state_pattern, content):
        quoted = match.group(1) is not None
        state_name = extract_state_name(match.group(2))
        shape = match.group(3)

        # Skip special nodes
        if state_name.startswith('__'):
            continue

        # `node [shape=...];` (unquoted) is Graphviz syntax for setting the
        # default shape of every node declared after it, not a declaration
        # of an actual state named "node" -- same for `edge`/`graph`. A real
        # state named "node" would be written quoted ("node" [shape=...]).
        if not quoted and state_name in ('node', 'edge', 'graph'):
            continue

        # shape=point marks an invisible anchor for the incoming arrow that
        # visually indicates the initial state (e.g. `qi [shape=point];
        # qi -> q0;`) -- not a real automaton state. The actual initial
        # state is resolved separately below via the qi/__start__ patterns.
        if shape == 'point':
            continue

        is_accepting = (shape == 'doublecircle')
        explicit_states[state_name] = is_accepting
    
    # Extract all transitions to find all states (including implicit ones)
    transition_pattern = r'"?(\w+)"?\s*->\s*"?(\w+)"?\s*\[label="([^"]+)"\]'
    
    transition_list = []
    for match in re.finditer(transition_pattern, content):
        from_state = extract_state_name(match.group(1))
        to_state = extract_state_name(match.group(2))
        symbol = match.group(3)
        
        # Skip transitions from special nodes
        if from_state.startswith('__') or to_state.startswith('__'):
            continue
        
        transition_list.append((from_state, to_state, symbol))
        alphabet.add(symbol)
    
    # Create states - combine explicit definitions with states found in transitions
    all_state_names = set(explicit_states.keys())
    for from_state, to_state, _ in transition_list:
        all_state_names.add(from_state)
        all_state_names.add(to_state)
    
    for state_name in all_state_names:
        # Use explicit definition if available, otherwise assume non-accepting
        is_accepting = explicit_states.get(state_name, False)
        state = DfaState(state_id=state_name, is_accepting=is_accepting)
        states_dict[state_name] = state
    
    # Add all transitions to states
    for from_state, to_state, symbol in transition_list:
        if from_state in states_dict and to_state in states_dict:
            states_dict[from_state].transitions[symbol] = states_dict[to_state]
    
    # Find initial state: try both __start__ and qi formats
    initial_state_name = None
    
    # Try __start__ format (modern)
    initial_pattern = r'__start__\s*->\s*"?(\w+)"?'
    initial_match = re.search(initial_pattern, content)
    
    if initial_match:
        initial_state_name = extract_state_name(initial_match.group(1))
    else:
        # Try qi format (legacy)
        qi_pattern = r'qi\s*->\s*"?(\w+)"?'
        qi_match = re.search(qi_pattern, content)
        if qi_match:
            initial_state_name = extract_state_name(qi_match.group(1))
    
    # If no explicit initial marker, use first state from transitions
    if not initial_state_name and transition_list:
        initial_state_name = transition_list[0][0]
    
    # If still no initial state, use first state in dictionary
    if not initial_state_name and states_dict:
        initial_state_name = next(iter(states_dict.keys()))
    
    if not initial_state_name:
        raise ValueError("Could not find initial state in DOT file")
    
    if initial_state_name not in states_dict:
        raise ValueError(f"Initial state {initial_state_name} not found in states dictionary")
    
    if not alphabet:
        raise ValueError("Could not extract any transitions from DOT file")
    
    return states_dict, initial_state_name, alphabet


def load_dfa_from_dot(dot_filename: str) -> Dfa:
    """
    Load a complete DFA from a .dot file in the automata/ folder.
    
    Parameters
    ----------
    dot_filename : str
        Filename (e.g., 'secure_handshake.dot') or language name
        (e.g., 'SecureHandshake')
    
    Returns
    -------
    Dfa
        aalpy DFA object with all states and transitions
    
    Raises
    ------
    FileNotFoundError
        If file not found in automata/ folder
    ValueError
        If DFA cannot be parsed
    """
    # Resolve filename - support both exact filename and language name
    if dot_filename in NAME_TO_FILE:
        actual_filename = NAME_TO_FILE[dot_filename]
    else:
        actual_filename = dot_filename
    
    dot_path = os.path.join(AUTOMATA_DIR, actual_filename)
    
    # Parse DOT file
    try:
        states_dict, initial_state_name, alphabet = parse_dot_file(dot_path)
    except FileNotFoundError:
        # List AUTOMATA_MAPPING's static keys here, not
        # list_available_automata()'s -- that function now calls
        # load_dfa_from_dot() on every mapped file to compute `states`
        # dynamically, so calling it from inside load_dfa_from_dot's own
        # not-found handler would recurse into itself for this exact file.
        raise FileNotFoundError(
            f"DOT file not found: {actual_filename}\n"
            f"Available files: {list(AUTOMATA_MAPPING.keys())}"
        )
    except ValueError as e:
        raise ValueError(f"Failed to parse {actual_filename}: {e}")
    
    # Get initial state and create DFA
    initial_state = states_dict[initial_state_name]
    dfa = Dfa(initial_state=initial_state, states=list(states_dict.values()))
    
    return dfa


def create_automata_dfa_predictor(dfa: Dfa):
    """
    Create a predictor function from automata DFA.
    
    Parameters
    ----------
    dfa : Dfa
        aalpy DFA object
    
    Returns
    -------
    callable
        Function that takes list of sequences and returns numpy array of predictions [0/1]
    """
    def predictor(sequences: List[List[str]]) -> np.ndarray:
        """
        Predict class labels for sequences.
        
        Parameters
        ----------
        sequences : List[List[str]]
            List of sequences, each sequence is a list of symbols
        
        Returns
        -------
        np.ndarray
            Binary predictions [0/1] for each sequence
        """
        # Walk transitions with a local variable instead of dfa.step()/
        # reset_to_initial(), which mutate dfa.current_state -- an attribute
        # on the single shared Dfa instance this closure captures. Called
        # concurrently (draw_automata_samples_parallel runs this predictor
        # from several ThreadPoolExecutor threads at once, one per candidate
        # automaton, under the default parallel=True), those threads would
        # all reset/step the same dfa.current_state and interleave, so the
        # accepting check for one thread's sequence could reflect a walk
        # partly stepped by another thread's sequence. Reading
        # dfa.initial_state and each state's .transitions dict is safe
        # concurrently since prediction never modifies the DFA structure.
        predictions = []
        for seq in sequences:
            current_state = dfa.initial_state
            try:
                for symbol in seq:
                    current_state = current_state.transitions[symbol]
                predictions.append(1 if current_state.is_accepting else 0)
            except (KeyError, AttributeError):
                # Symbol not in alphabet or transition not defined
                predictions.append(0)

        return np.array(predictions, dtype=int)
    
    return predictor


def list_available_automata() -> Dict[str, Dict]:
    """
    List all available automata in the automata/ folder.

    `states` is computed by actually loading each .dot file rather than
    stored statically, so it can't drift out of sync the way the old
    hardcoded counts did. A file missing from automata/ (declared in
    AUTOMATA_MAPPING but not on disk) gets states=None instead of failing
    the whole listing.

    Returns
    -------
    dict
        Mapping of dot filename to metadata (name, description, states)
    """
    result = {}
    for filename, meta in AUTOMATA_MAPPING.items():
        entry = dict(meta)
        try:
            entry['states'] = len(load_dfa_from_dot(filename).states)
        except Exception:
            entry['states'] = None
        result[filename] = entry
    return result


# def get_automata_alphabet(dfa: Dfa) -> List[str]:
#     """
#     Extract alphabet from DFA by collecting all unique symbols used in transitions.
    
#     Parameters
#     ----------
#     dfa : Dfa
#         aalpy DFA object
    
#     Returns
#     -------
#     list
#         Sorted list of symbols used in DFA
#     """
#     alphabet = set()
#     for state in dfa.states:
#         for symbol in state.transitions.keys():
#             alphabet.add(symbol)
#     return sorted(list(alphabet))


if __name__ == '__main__':
    print("Testing automata DFA loading...")
    print("\nAvailable automata:")
    for name, meta in list_available_automata().items():
        print(f"  {name:30s}: {meta['description']}")
    
    print("\n\nTesting load_dfa_from_dot()...")
    dfa = load_dfa_from_dot('SecureHandshake')
    print(f"Loaded DFA: {len(dfa.states)} states")
    
    alphabet = get_alphabet(dfa)
    print(f"Alphabet: {alphabet}")
    
    # Test predictor
    predictor = create_automata_dfa_predictor(dfa)
    test_sequences = [
        [],
        ['a'],
        ['a', 'b'],
        ['a', 'b', 'c'],
    ]
    predictions = predictor(test_sequences)
    print(f"\nTest predictions: {predictions}")
    print(f"Prediction shape: {predictions.shape}")
