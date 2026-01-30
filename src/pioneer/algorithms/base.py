import torch

class Algorithm(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
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
    
    def get_hyperparameters(self):
        return ...

class SequentialAlgorithm(Algorithm):
    def __init__(self, algorithms):
        super().__init__()
        self.algorithms = torch.nn.ModuleList(algorithms)
    
    @property
    def input_config(self):
        return self.algorithms[0].input_config
    
    @property
    def output_config(self):
        return self.algorithms[-1].output_config
    
    def forward(self, input_data):
        data = input_data
        for alg in self.algorithms:
            data = alg(data)
        return data

class IterativeAlgorithm(Algorithm):
    def __init__(self, algorithm, n_iterations):
        super().__init__()
        ...

class DeepLearningModel(Algorithm):
    def __init__(self):
        super().__init__()
        # this is just the forward model        