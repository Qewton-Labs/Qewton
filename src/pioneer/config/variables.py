from __future__ import annotations
from collections import OrderedDict
from math import prod


class NO_NAME:
    pass


class Variable(OrderedDict):
    """Creates a variable of the given problem. Helps for a natural
    implementation of the problem and internal tracking.
    """

    def __init__(
        self, name: str | None = NO_NAME, dim: int | tuple[int, ...] | None = None
    ):
        """
        Args:
            name (str | None, optional): The name of the variable. Defaults to None.
            dim (int | tuple[int, ...] | None, optional): The dimension of the variable.
            Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self[name] = dim

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

    def is_empty(self) -> bool:
        """Checks if the variable is empty, i.e. has no keys.

        Returns:
            bool: True if the variable is empty, False otherwise.
        """
        return self.keys() == {NO_NAME} and self.values() == {None}

    def unify(self, other: Variable) -> Variable:
        """Unifies two variables, i.e. checks if they are compatible and returns
        a new variable containing the information from both original variables.

        Args:
            other (Variable): The other variable to unify with."""
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        key_diff = self.keys() ^ other.keys()
        out = {}
        if len(key_diff) == 2:
            if NO_NAME in key_diff:
                other_key = next(iter(key_diff - {NO_NAME}))
                if NO_NAME in self:
                    out[other_key] = Variable.check(self, other, NO_NAME, other_key)
                else:
                    out[other_key] = Variable.check(other, self, NO_NAME, other_key)
            else:
                raise ValueError("Variable names have to agree for unification.")
        elif len(key_diff) != 0:
            raise ValueError("Variable names have to agree for unification.")
        for key in self.keys() & other.keys():
            out[key] = Variable.check(self, other, key, key)
        return Variable.from_dict(out)

    @classmethod
    def check(cls, variable_a: Variable, variable_b: Variable, a_key, b_key):
        if variable_a[a_key] is None:
            return variable_b[b_key]
        elif variable_b[b_key] is None:
            return variable_a[a_key]
        elif variable_a[a_key] != variable_b[b_key]:
            raise ValueError("Variable dimensions have to agree for unification.")
        else:
            return variable_a[a_key]

    def __mul__(self, other: Variable) -> Variable:
        """Combines two variables to a single object.

        Args:
            other (Variable): The other variable.

        Returns:
            Variable: The combined variable containing the information from
                both original variables (Cross-product)
        """
        if len(self.keys() & other.keys()) > 0:
            raise ValueError("Variables with overlapping names cannot be combined.")
        result = Variable.from_dict(self)
        for k, v in other.items():
            result[k] = v
        for v in result.values():
            if isinstance(v, tuple):
                raise ValueError(
                    "Can not combine variables with tuple dimensions. \
                        Please flatten the dimensions."
                )
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
