from abc import ABC, abstractmethod


class BaseAutomataLearner(ABC):
    """
    Minimal interface for automata learners used by AutomataBeamSearch.

    A learner is responsible for:
    1. constructing an initial automaton,
    2. proposing neighboring automata,
    3. checking whether an automaton accepts a sequence.
    """

    @abstractmethod
    def get_sampler(self):
        """Return the sampler class associated with this learner."""
        pass

    @abstractmethod
    def create_init_automata(self, data_type, positive_samples, negative_samples):
        """Create an initial automaton from positive and negative traces."""
        pass

    @abstractmethod
    def propose_automata(
        self,
        automata_list,
        state,
        iteration,
        previous_best,
        output_dir,
        beam_size,
        batch_size,
    ):
        """Propose neighboring automata candidates."""
        pass

    @abstractmethod
    def check_path_accepted(self, automaton, path) -> bool:
        """Return True if the automaton accepts the path."""
        pass