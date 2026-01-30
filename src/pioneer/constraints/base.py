
class Constraint():
    def is_fulfilled_by(self, algorithm, data=None):
        raise NotImplementedError("Method not implemented in base class.")

class ResourceConstraint(Constraint):
    """
    technical constraints, e.g. memory, compute time etc...
    """
    def __init__(self):
        pass

class ProblemConstraint(Constraint):
    """
    problem constraints, e.g. data, PDE, symmetries etc...
    """
    def __init__(self):
        state = ... #could be 'objective function', 'hard constraint' or 'tracking only'
    
    @property
    def input_config(self):
        raise NotImplementedError("Input configuration not implemented.")
    
    @property
    def output_config(self):
        raise NotImplementedError("Output configuration not implemented.")

    def is_fulfilled_by(self, algorithm, data=None):
        return algorithm.fulfills(self, data=data)