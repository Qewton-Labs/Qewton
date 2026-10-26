# Pipelines

Predefined graph templates for common tasks. A pipeline is a
`qewton.graphs.Graph` subclass that builds its nodes and connections in its
constructor and is then used like any other graph.

## Public API
- **`PINNPipeline`**: Connects a sampler (any `DataNode`), one or more models
  and a physics constraint for PINN training. It splits and joins the
  sampled variables as needed by the models and the constraint (matched by
  variable name), and adds gradient tracking for every sampled variable a
  model uses, so the constraint can differentiate the model outputs.

## Usage
```python
from qewton.constraints import PINNConstraint
from qewton.graphs.pipelines import PINNPipeline

def residual(u: U, f: F, x: X):
    return u.gradient(x) - f

constraint = PINNConstraint(residual)
pipeline = PINNPipeline(data_loader, [model], constraint)
pipeline.setup()
```

The pipeline and its constraint are then passed to a trainer, e.g.
`GraphBasedTrainer(graphs=[pipeline], training_objectives=[constraint], ...)`.
