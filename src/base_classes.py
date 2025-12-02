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

class Variable():
    """
    represents a variable in the problem, e.g. input, output, parameter etc...
    """
    def __init__(self, name, datatype_config):
        self.name = name
        self.datatype_config = datatype_config


class DatatypeConfiguration():
    """
    sets the basic type (numpy array, torch tensor etc) and shape of the data, and also collections of these
    will be used to check compatibility of the algorithms
    also include variables and their names?
    
    -> later implement several configuration conversion methods (and visualization), it should be possible to this during
    the execution of an algorithm as well as offline
    ->  also suggest automatic conversion methods between compatible configurations
    
    dimemsions like (None, 3, None)
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
    def state(self):
        # whether the algorithm is trained, initialized, optimized etc... 
        pass
    
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
        # this is just the forward model
        
class Optimization():
    def __init__(self):
        super().__init__()
        # this should be visualized seperately from the solution approach (''algorithm''),
        # and only change the state of the solution approach once the optimization is done
        

# add option to run algorithms partially, e.g. only train