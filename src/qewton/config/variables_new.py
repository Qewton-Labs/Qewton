from __future__ import annotations


class Variable:
    def __init__(
        self,
        name: str,
        dim: int | tuple[int, ...] | None = None,
        children: list[Variable] | None = None,
        parent: Variable | None = None,
    ):
        self.name = name
        self.parent = parent
        if isinstance(dim, int):
            assert children is None, "Cannot specify both dim and children"
            if dim < 1:
                raise ValueError("dim must be >= 1")
            if dim == 1:
                self.children = []
                self.dim = 1
            else:
                self.children = [
                    Variable(self, f"{name}_{i}", dim=1, parent=self) for i in range(dim)
                ]
                self.dim = dim
        if isinstance(dim, tuple):
            assert children is None, "Cannot specify both dim and children"
            self.dim = dim
            self.children = []

        if children is not None:
            self.children = children
            self.dim = sum(child.dim for child in children)

        if children is None and dim is None:
            raise ValueError("Either dim or children must be specified.")

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def leaves(self):
        if self.is_leaf():
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves.extend(child.leaves())
            return leaves

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.leaves[key]
        if isinstance(key, str):
            for child in self.children:
                if child.name == key:
                    return child
            raise KeyError(f"No child with name {key}")

    def __iter__(self):
        return iter(self.leaves)

    def __repr__(self):
        if self.is_leaf:
            return f"Variable('{self.name}', dim={self.dim})"
        return f"Variable('{self.name}', dim={self.dim}, children={len(self.children)})"

    def __mul__(self, other: Variable) -> Variable: ...
