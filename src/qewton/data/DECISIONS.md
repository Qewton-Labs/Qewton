# Data: Design Decisions

## Reference comparisons evaluate the model at the reference points
When a model is compared to reference data (`active_discretization`), the
samplers output the reference geometry's own points instead of sampling or
meshing. The model is thus evaluated exactly where the reference has data,
and no interpolation error enters the comparison.

## Reference geometries are validated before use
Setting `PointSampler.active_discretization` checks the dimension, the
bounding-box overlap with the sampler's domain, and that all points lie
inside the domain. A mismatch (typically units or normalization, or a
different coordinate order) then fails with a clear error instead of
producing an error plot that looks like a bad model.
