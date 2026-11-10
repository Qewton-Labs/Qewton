import torch

from qewton.config import Variable
from qewton.data import DataLoader, GridDataSet
from qewton.optim import DiscreteHyperparameter

domain_var = Variable("domain", 1)
temperature_var = Variable("temperature", 1)
velocity_var = Variable("velocity", 3)
pressure_var = Variable("pressure", 1)
x_var = Variable("x", 3)


# Maximum allowed absolute values for each variable
THRESHOLDS = {
    "temperature": 100,
    "velocity": 100,
    "pressure": 1e6,
    "domain_coords": 10,
}


def filter_bad_samples(data: dict, thresholds: dict = THRESHOLDS):
    n_samples = data["domain_mask"].shape[0]

    keep = torch.ones(n_samples, dtype=torch.bool)
    reasons = {}

    for name, threshold in thresholds.items():
        values = data[name]

        for i in range(n_samples):
            if not keep[i]:
                continue

            sample = values[i]
            mask = data["domain_mask"][i].bool()
            if sample.ndim == mask.ndim + 1:
                mask = mask.unsqueeze(-1).expand_as(sample)

            sample = sample[mask]

            # Check NaN and infinity
            if not torch.isfinite(sample).all():
                keep[i] = False
                reasons.setdefault(i, []).append(f"{name}: NaN/Inf")
                continue

            # Check maximum absolute value
            max_abs = sample.abs().max().item()

            if max_abs > threshold:
                keep[i] = False
                reasons.setdefault(i, []).append(
                    f"{name}: max abs = {max_abs:.4e} (limit {threshold})"
                )

    # Print report
    print("\n--- Data quality report ---")
    print(f"Total samples:     {n_samples}")
    print(f"Valid samples:     {keep.sum().item()}")
    print(f"Discarded samples: {(~keep).sum().item()}")

    if reasons:
        print("\nDiscarded samples:")
        for i, reason in reasons.items():
            print(f"  Sample {i}: {'; '.join(reason)}")

    return keep


def load_data(
    batch_size: int | DiscreteHyperparameter = 10, path="/localdata/qewton/dataset.pt"
):
    data = torch.load(path, weights_only=True)

    # run filer
    keep = filter_bad_samples(data)

    # remove invalid samples
    data = {key: value[keep] for key, value in data.items()}

    # discard pressure for now
    dataset = GridDataSet(
        data=[
            data["domain_mask"].float().unsqueeze(-1),
            data["temperature"],
            data["velocity"],
            data["domain_coords"],
            # data["pressure"],
        ],
        feature_variables=[
            domain_var,
            temperature_var,
            velocity_var,
            x_var,
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
