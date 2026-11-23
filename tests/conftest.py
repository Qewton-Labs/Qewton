"""Settings for the whole test suite."""

import os

if "PYTEST_XDIST_WORKER" in os.environ:
    # Parallel workers share the cores; one thread each avoids oversubscription.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    import torch

    torch.set_num_threads(1)
