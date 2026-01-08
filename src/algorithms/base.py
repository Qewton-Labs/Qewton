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

class DeepLearningModel(Algorithm):
    def __init__(self):
        super().__init__()
        # this is just the forward model
        