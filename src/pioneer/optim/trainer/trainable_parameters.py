from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Iterable


class _TrainableParameterBase(ABC):

    @abstractmethod
    def __iter__(self) -> Iterator[TrainableParameters]:
        pass

    @property
    @abstractmethod
    def empty(self) -> bool:
        pass

    def combine(self, other: _TrainableParameterBase):
        return TrainableParametersCollection(self, other)

    def __add__(self, other):
        return self.combine(other)

    @abstractmethod
    def update_option(self, key, value):
        pass


class TrainableParameters(_TrainableParameterBase):
    """A class to represent trainable parameters of a node."""

    def __init__(self, name, parameters, **kwargs):
        self._name = name
        self._parameters = parameters
        self._options = kwargs

    @classmethod
    def create_empty(cls) -> TrainableParameters:
        return cls("", None)

    @property
    def empty(self) -> bool:
        return self._parameters is None

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self):
        return self._parameters

    @property
    def options(self):
        return self._options

    def __iter__(self) -> Iterator[TrainableParameters]:
        yield self

    def update_option(self, key, value):
        self.options[key] = value


class TrainableParametersCollection(_TrainableParameterBase):

    def __init__(self, *groups: _TrainableParameterBase):
        self._groups: list[TrainableParameters] = []
        # save names as an extra set for faster look up of duplicates
        self._names: set[str] = set()
        self.extend(groups)

    def add(self, param: TrainableParameters):
        if param.empty or param.name in self._names:
            return

        self._groups.append(param)
        self._names.add(param.name)

    def extend(self, groups: _TrainableParameterBase | Iterable[_TrainableParameterBase]):
        if isinstance(groups, _TrainableParameterBase):
            groups = (groups,)

        for g in groups:
            if isinstance(g, TrainableParametersCollection):
                for param in g:
                    self.extend(param)
            elif isinstance(g, TrainableParameters):
                self.add(g)

    def __iter__(self) -> Iterator[TrainableParameters]:
        return iter(self._groups)

    @property
    def empty(self) -> bool:
        return len(self._groups) == 0

    @property
    def parameters(self):
        return self._groups

    def update_option(self, key, value):
        for p in self.parameters:
            p.update_option(key, value)
