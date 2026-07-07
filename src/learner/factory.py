"""
Learner Factory - Create learner instances based on type
"""
from .base import BaseAutomataLearner
from .dfa_learner import DFALearner
from .ra_learner import RegisterAutomataLearner


class LearnerFactory:
    """
    Learner factory: create corresponding learner instances based on type
    """
    _learners = {}
    _instances = {}  # Shared instances cache
    
    @classmethod
    def register(cls, learner_type: str, learner_class):
        """Register a new learner type"""
        cls._learners[learner_type.lower()] = learner_class
    
    @classmethod
    def create(cls, learner_type: str = 'dfa', new_instance: bool = True) -> BaseAutomataLearner:
        """
        Create a learner instance
        
        :param learner_type: Learner type ('dfa', 'ra', etc.)
        :param new_instance: If False, return shared instance
        :return: Corresponding learner instance
        """
        learner_type = learner_type.lower()
        if learner_type not in cls._learners:
            available = list(cls._learners.keys())
            raise ValueError(f"Unknown learner type: '{learner_type}'. Available: {available}")
        
        if not new_instance:
            if learner_type not in cls._instances:
                cls._instances[learner_type] = cls._learners[learner_type]()
            return cls._instances[learner_type]
        
        return cls._learners[learner_type]()
    
    @classmethod
    def available_types(cls) -> list:
        """Return all available learner types"""
        return list(cls._learners.keys())


# Register the learners
LearnerFactory.register('dfa', DFALearner)
LearnerFactory.register('ra', RegisterAutomataLearner)
LearnerFactory.register('register', RegisterAutomataLearner)  # Alias


# ============================================================
# Global instances and convenience functions
# ============================================================


def get_learner(learner_type: str = 'dfa', new_instance: bool = False) -> BaseAutomataLearner:
    """
    Get learner instance
    
    :param learner_type: Learner type ('dfa', 'ra', 'register')
    :param new_instance: Whether to create a new instance
    :return: Learner instance
    
    Usage:
        learner = get_learner('dfa')       # Shared DFA learner
        learner = get_learner('ra')        # Shared RA learner
        learner = get_learner('dfa', True) # New DFA instance
    """
    
    if new_instance:
        return LearnerFactory.create(learner_type, new_instance=True)
    
    learner_type = learner_type.lower()
    if learner_type == 'dfa':
        return DFALearner()
    elif learner_type in ('ra', 'register'):
        return RegisterAutomataLearner()
    
    return LearnerFactory.create(learner_type, new_instance=False)
