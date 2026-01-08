
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