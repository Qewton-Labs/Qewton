
class HyperParameter():
    def __init__(self, state, dtype, range=None):
        self.state = state
        self.dtype = dtype
        self.range = range


class Optimization():
    def __init__(self):
        super().__init__()
        # this should be visualized seperately from the solution approach (''algorithm''),
        # and only change the state of the solution approach once the optimization is done
    
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