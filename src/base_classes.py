class Constraint():
    """
    problem constraints, e.g. data, PDE, symmetries etc...
    """
    def __init__(self):
        pass
    
    @property
    def input_config(self):
        raise NotImplementedError("Input configuration not implemented.")
    
    @property
    def output_config(self):
        raise NotImplementedError("Output configuration not implemented.")

class TechnicalConstraint():
    """
    technical constraints, e.g. memory, compute time etc...
    """
    def __init__(self):
        pass


class DatatypeConfiguration():
    """
    sets the basic type (numpy array, torch tensor etc) and shape of the data, and also collections of these
    will be used to check compatibility of the algorithms
    
    -> later implement several configuration conversion methods (and visualization), it should be possible to this during
    the execution of an algorithm as well as offline
    ->  also suggest automatic conversion methods between compatible configurations
    """
    def __init__(self):
        pass

class Algorithm():
    def __init__(self):
        pass
    
    def fulfills(self, constraint, data=None):
        # return True or an empirical measure on how well a constraint is fulfilled (if data is available)?
        raise NotImplementedError("Fulfills method not implemented.")
    
    @property
    def input_config(self):
        raise NotImplementedError("Input configuration not implemented.")
    
    @property
    def output_config(self):
        raise NotImplementedError("Output configuration not implemented.")
    
    def forward(self, input_data):
        raise NotImplementedError("Forward method not implemented.")

class DeepLearningModel(Algorithm):
    def __init__(self):
        super().__init__()
        # the model that is applied is the minimizer of a loss function, minimized before

# add option to run algorithms partially, e.g. only train