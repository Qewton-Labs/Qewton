
class HyperParameter():
    def __init__(self, dtype, state=None, range=None):
        """
        Docstring for __init__
        
        :param self: Description
        :param dtype: Description
        :param state: If both state and range are defined, the state is used for 
        all constraint evaluation purposes first, but the range is used for optimization
        :param range: Description
        """
        self.state = state
        self.dtype = dtype
        self.range = range

class ContinuousHyperparameter(HyperParameter):
    def __init__(self, state=None, range=None):
        super().__init__(dtype=float, state=state, range=range)


class Optimization():
    """Should we use pytorch lightning here?"""
    def __init__(self):
        super().__init__()
        # this should be visualized seperately from the solution approach (''algorithm''),
        # and only change the state of the solution approach once the optimization is done
        self.constraints = ... # state of constraints decides whether its an objective function or only tracked..
    
    def get_hyperparameters(self):
        return ...

class SingleOptimization(Optimization):
    def __init__(self):
        super().__init__()
        # optimization with a single set of hparameters

class GridSearchOptimization(Optimization):
    def __init__(self):
        super().__init__()
        # optimization over a grid of hparameters
        for param in self.get_hyperparameters():
            pass