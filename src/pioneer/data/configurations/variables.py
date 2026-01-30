from __future__ import annotations
from collections import OrderedDict


class Variable(OrderedDict):
    def __init__(self, name: str | None = None, dim: int | None = None):
        super().__init__()
        if name is not None:
            if dim is None:
                raise ValueError("Dimension must be provided if name is given.")

            self[name] = dim

    @classmethod
    def from_dict(cls, var_dict: dict[str, int]):
        v = cls()
        for name, dim in var_dict.items():
            v[name] = dim
        return v

    def __mul__(self, other: Variable) -> Variable:
        result = Variable.from_dict(self)
        for k, v in other.items():
            result[k] = result.get(k, 0) + v
        return result

    def __add__(self, other: Variable) -> Variable:
        return self * other

    @property
    def dim(self):
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
