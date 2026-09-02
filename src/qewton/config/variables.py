from __future__ import annotations
from collections import OrderedDict
from math import prod
from typing import Optional

from qewton.config.saving.saving import Serializable


class Variable(OrderedDict, Serializable):
    """Creates a variable of the given problem. Helps for a natural
    implementation of the problem and internal tracking.

    Args:
        name (str | None, optional): The name of the variable. Defaults to None.
        dim (int | tuple[int, ...] | None, optional): The dimension of the variable.
        Defaults to None.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        dim: int | tuple[int, ...] | None = None,
    ):
        super().__init__()
        if name is not None:
            self[name] = dim
        self.has_multiple_axes = isinstance(dim, tuple)

    @property
    def name(self) -> str:
        """Returns the variable keys split by ", ".

        Returns:
            str: The variable keys as a string.
        """
        return ", ".join(str(key) for key in self.keys())

    @classmethod
    def from_dict(cls, var_dict: dict[str, int]) -> Variable:
        """Construct a variable from a given dictionary.

        Args:
            var_dict (dict[str, int]): The dictionary containing the variable
                information. The keys of the dictionary are used as the
                variable names and the values should denote the dimension.

        Returns:
            Variable: The variable object.
        """
        v = cls()
        for name, dim in var_dict.items():
            v[name] = dim
        return v

    def get_slice(self, variable: Variable) -> tuple[slice, ...] | list[int]:
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
        if self.has_multiple_axes:
            return tuple([slice(None)] * len(list(self.values())[0]))
        slc = []
        for variable_k, variable_v in variable.items():
            prev_dims = 0
            found = False
            for k, v in self.items():
                if k == variable_k:
                    slc.extend(list(range(prev_dims, prev_dims + variable_v)))
                    found = True
                    break
                prev_dims += v
            if not found:
                raise KeyError(f"Variable key '{variable_k}' not found in {self.keys()}")
        return slc

    def is_empty(self) -> bool:
        """Checks if the variable is empty, i.e. has no keys.

        Returns:
            bool: True if the variable is empty, False otherwise.
        """
        return len(self) == 0

    def unify(self, other: Variable) -> Variable:
        """Unifies two variables, i.e. checks if they are compatible and returns
        a new variable containing the information from both original variables.

        Args:
            other (Variable): The other variable to unify with.

        Raises:
            ValueError: If the variables have multiple axes.
            ValueError: If the variable names do not agree for unification.
        """
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        if self.has_multiple_axes or other.has_multiple_axes:
            raise ValueError("Can not combine variables with multiple axes.")
        key_diff = self.keys() ^ other.keys()
        out = {}
        if len(key_diff) != 0:
            raise ValueError("Variable names have to agree for unification.")
        for key in self.keys() & other.keys():
            out[key] = Variable.check(self, other, key, key)
        return Variable.from_dict(out)

    @classmethod
    def check(cls, variable_a: Variable, variable_b: Variable, a_key, b_key):
        if variable_a[a_key] is None:
            return variable_b[b_key]
        if variable_b[b_key] is None:
            return variable_a[a_key]
        if variable_a[a_key] != variable_b[b_key]:
            raise ValueError("Variable dimensions have to agree for unification.")
        return variable_a[a_key]

    def __mul__(self, other: Variable) -> Variable:
        """Combines two variables to a single object.

        Args:
            other (Variable): The other variable.

        Returns:
            Variable: The combined variable containing the information from
                both original variables (Cross-product)
        """
        if self.has_multiple_axes or other.has_multiple_axes:
            raise ValueError("Can not combine variables with multiple axes.")
        if len(self.keys() & other.keys()) > 0:
            raise ValueError("Variables with overlapping names cannot be combined.")
        result = Variable.from_dict(self)
        for k, v in other.items():
            result[k] = v
        return result

    def __add__(self, other: Variable) -> Variable:
        return self * other

    def __hash__(self):
        hash_name = ""
        for key, value in self.items():
            hash_name += key + str(value) + "_"
        return hash(hash_name)

    @property
    def dim(self):
        first_value = next(iter(self.values()), None)
        if isinstance(first_value, tuple):  # there can be no other keys
            return prod(first_value)
        return sum(self.values())

    @property
    def shape(self):
        if self.has_multiple_axes:
            return list(self.values())[0]
        return (self.dim,)

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self)})"

    def __contains__(self, variable) -> bool:
        if isinstance(variable, str):
            return super().__contains__(variable)

        if isinstance(variable, Variable):
            return all((k in self and self[k] == v) for k, v in variable.items())

        return False

    def __getitem__(self, val: str | slice | list[str] | tuple[str]):
        if isinstance(val, slice):
            keys = list(self.keys())
            new_slice = slice(
                keys.index(val.start) if val.start is not None else None,
                keys.index(val.stop) if val.stop is not None else None,
                val.step,
            )
            new_keys = keys[new_slice]
            return self.from_dict({k: int(self[k]) for k in new_keys})
        if isinstance(val, (list, tuple)):
            return self.from_dict({k: int(self[k]) for k in val})
        return super().__getitem__(val)
