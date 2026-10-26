# Optim: Design Decisions

## Losses are identified by unique constraint names
Losses are recorded and looked up in the trainer state by name. The
`GraphBasedTrainer` therefore renames constraints that share a name
(`"PDE"`, `"PDE_2"`, ...), in the order of the training objectives, then of
creation. When losses are recorded, two different owners with the same name
raise a `ValueError` instead of silently overwriting each other.
