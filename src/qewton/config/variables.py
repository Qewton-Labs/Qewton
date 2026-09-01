from __future__ import annotations


class Variable:
    """Order of children is now important."""

    def __deepcopy__(self, memo):
        # Variables are objects with reference identity, not names (the
        # same principle that makes Scale an object rather than a string
        # key) - two ports/configs are "the same quantity" exactly when
        # they share one Variable instance. Every node's dynamic
        # DataConfiguration is a deepcopy of its static one
        # (Node.copy_data_config_of_port), so without this, a caller who
        # deliberately passes the same Variable to two different nodes
        # would still see two different (deep-copied) objects looking back
        # from each node's own dynamic config - silently defeating any
        # identity-based comparison. Exempted from copying instead, the
        # same way SampledGeometry.__deepcopy__ already exempts itself for
        # an analogous reason.
        return self

    def __init__(
        self,
        name: str | None = None,
        dim: int | tuple[int, ...] | None = None,
        children: list[Variable] | None = None,
        parent: Variable | None = None,
    ):
        self.name = name
        self.parent = parent

        if children is None:
            if isinstance(dim, int):
                if dim == 1 or dim == 0:
                    self.children = []
                else:
                    self.children = [
                        Variable(f"{name}_{i}", dim=1, parent=self) for i in range(dim)
                    ]
                self.dim = dim
            if isinstance(dim, tuple):
                assert (
                    children is None
                ), "Variables with multiple axes cannot have children."
                self.dim = dim
                self.children = []

        if children is not None:
            self.children = children
            overall_dim = 0
            c_dim_is_none = False
            for child in children:
                if child.parent is None:
                    child.parent = self
                if isinstance(child.dim, tuple):
                    raise ValueError("Children cannot have multiple axes.")
                if isinstance(child.dim, int):
                    overall_dim += child.dim
                elif child.dim is None:
                    c_dim_is_none = True

            if c_dim_is_none:
                if dim is None:
                    self.dim = None
                else:
                    raise ValueError("Cannot specify dim from parent to children.")
            else:
                self.dim = overall_dim
                if dim is not None:
                    assert (
                        dim == overall_dim
                    ), f"Computed dim {self.dim} does not agree with given dim {dim}"

        else:
            if dim is None:
                self.dim = dim
                self.children = []

    @classmethod
    def empty(cls):
        return cls()

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def leaves(self):
        if self.is_leaf:
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves.extend(child.leaves)
            return leaves

    @property
    def leaf_names(self) -> set[str]:
        """The names of every leaf in this Variable's tree."""
        return {leaf.name for leaf in self.leaves}

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
            return f"Variable({self.name}, dim={self.dim})"
        return f"Variable({self.name}, dim={self.dim}, children={len(self.children)})"

    def __mul__(self, other: Variable) -> "Variable":
        new_var = Variable(name=f"({self.name}, {other.name})", children=[self, other])
        if self.parent is None:
            self.parent = new_var
        if other.parent is None:
            other.parent = new_var
        return new_var

    __add__ = __mul__

    @staticmethod
    def compose(variables: list[Variable]) -> Variable:
        """Folds `variables` into one Variable via `*`, left to right.

        Returns the single variable itself, unchanged, when only one is
        given - `*`-ing a lone variable would otherwise wrap it in a
        pointless composite.
        """
        if len(variables) == 1:
            return variables[0]
        result = variables[0]
        for variable in variables[1:]:
            result = result * variable
        return result

    @classmethod
    def from_dict(cls, data: dict[str, int | tuple[int, ...]]) -> Variable:
        children = []
        name = "("
        if len(data) == 0:
            return cls()
        for k, v in data.items():
            if isinstance(v, (int, tuple)):
                children.append(Variable(name=k, dim=v))
                name += f"{k}, "
            else:
                raise ValueError(f"Invalid value type for key {k}: {type(v)}")
        if len(children) == 1:
            return children[0]
        else:
            return cls(name=name.rstrip(", ") + ")", children=children)

    def get_slice(self, variable: Variable) -> slice:
        """Computes a slice index for the variable provided in this main
        variable. The provided variable has to be included in this
        variable

        Args:
            variable (Variable): The variable for which to compute the slice.

        Raises:
            KeyError: If the provided variable contains keys that are not present
                in this variable.

        Returns:
            tuple[slice, ...] | list[int]: The slice indices.
        """
        if variable == self:
            return slice(None)
        if self.is_leaf:
            raise KeyError(f"Variable '{variable.name}' not found in '{self.name}'")
        running_idx = 0
        for child in self.children:
            if variable == child:
                return slice(running_idx, running_idx + child.dim)  # type: ignore
            if variable in child.leaves:
                child_slice = child.get_slice(variable)
                return slice(
                    running_idx + child_slice.start, running_idx + child_slice.stop
                )
            running_idx += child.dim  # type: ignore
        raise KeyError(f"Variable '{variable.name}' not found in '{self.name}'")

    @property
    def is_empty(self) -> bool:
        """Check if the variable has no children and dim is 0.

        Returns:
            bool: True if the variable is empty, False otherwise.
        """
        return self.dim is None and len(self.children) == 0 and self.name is None

    def _hash_name(self):
        hash_name = self.name if self.name is not None else ""
        hash_name += str(self.dim)
        for v in self.children:
            hash_name += v._hash_name()  # pylint: disable=W0212
            hash_name += ";"
        return hash_name

    def __hash__(self):
        return hash(self._hash_name())

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self._hash_name() == other._hash_name()

    def prune(self, keep_names: set[str]) -> "Variable | None":
        """Removes every leaf whose name isn't in `keep_names`, preserving
        as much of the original grouping as possible.

        Returns this Variable itself, unchanged, if none of its leaves
        needed dropping - so an auto-expanded multi-component variable used
        whole stays whole instead of being flattened into its individual
        components just because some other, unrelated variable needed
        pruning. Returns None if every leaf was dropped.
        """
        if self.is_leaf:
            return self if self.name in keep_names else None
        survivors = [child.prune(keep_names) for child in self.children]
        survivors = [child for child in survivors if child is not None]
        if survivors == self.children:
            return self
        if not survivors:
            return None
        return Variable.compose(survivors)

    def unify(self, other: Variable) -> Variable:
        """Unifies two variables, i.e. checks if they are compatible and returns
        a new variable containing the information from both original variables.

        Args:
            other (Variable): The other variable to unify with.

        Raises:
            ValueError: If the variables have multiple axes.
            ValueError: If the variable names do not agree for unification.
        """
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        if isinstance(self.dim, tuple) or isinstance(other.dim, tuple):
            raise ValueError("Can not combine variables with multiple axes.")

        # check name
        if self.name is None:
            out_name = other.name
        elif other.name is None:
            out_name = self.name
        elif self.name != other.name:
            raise ValueError("Variable names have to agree for unification.")
        else:
            out_name = self.name

        # check dim
        if self.dim is None:
            out_dim = other.dim
        elif other.dim is None:
            out_dim = self.dim
        elif self.dim != other.dim:
            raise ValueError("Variable dimensions have to agree for unification.")
        else:
            out_dim = self.dim

        # check children
        out_children = []
        for child1, child2 in zip(self.children, other.children):
            out_children.append(child1.unify(child2))

        return Variable(out_name, out_dim, out_children)

    @property
    def shape(self):
        if isinstance(self.dim, tuple):
            return self.dim
        return (self.dim,)

    def __contains__(self, other: Variable):
        if self.is_empty:
            return False
        if other == self:
            return True
        for child in self.children:
            if other in child:
                return True
        return False
