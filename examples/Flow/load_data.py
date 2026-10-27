import torch

from qewton.config import Variable
from qewton.data import DataLoader, GridDataSet
from qewton.optim import DiscreteHyperparameter

domain_var = Variable("domain", 1)
temperature_var = Variable("temperature", 1)
velocity_var = Variable("velocity", 3)
pressure_var = Variable("pressure", 1)


def load_data(
    batch_size: int | DiscreteHyperparameter = 10, path="/localdata/qewton/dataset.pt"
):
    data = torch.load(path, weights_only=True)

    # discard pressure for now
    dataset = GridDataSet(
        data=[
            data["domain_mask"].float().unsqueeze(-1),
            data["temperature"],
            data["velocity"],
            # data["pressure"],
        ],
        feature_variables=[
            domain_var,
            temperature_var,
            velocity_var,
            # pressure_var,
        ],
        point_grid=data["domain_coords"][0],
    )
    data_loader = DataLoader(
        data_set=dataset,
        batch_size=batch_size,
        splitting_ratio=(0.7, 0.15, 0.15),
        shuffle_data=True,
        shuffle_seed=314,
    )
    return dataset, data_loader
