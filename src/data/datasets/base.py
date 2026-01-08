
class DataSet():
    def __init__(self, data_config, data=None):
        pass
    
    def from_data(data):
        data_config = ...
        return DataSet(data_config=data_config, data=data)
    
    def compute_pca(self, n_components):
        self.pca = ...
    
    def pca(self, n_components=None):
        if self.pca:
            if n_components is None or n_components == self.pca.n_components:
                return self.pca
            elif n_components < self.pca.n_components:
                return self.pca[:n_components]

        self.compute_pca(n_components)
        return self.pca
    
    def compute_mean(self):
        self.mean = ...
    
    def mean(self):
        if self.mean is None:
            self.compute_mean()
        return self.mean
    
    def compute_std(self):
        self.std = ...
    
    def std(self):
        if self.std is None:
            self.compute_std()
        return self.std