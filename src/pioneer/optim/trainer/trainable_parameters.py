from __future__ import annotations


class TrainableParameters:
    """A class to represent trainable parameters of a node."""

    def __init__(self, name, parameters, **kwargs):
        self._name = name
        self._parameters = parameters
        self._options = kwargs

    @classmethod
    def create_empty(cls) -> TrainableParameters:
        return cls("", None)

    @property
    def empty(self):
        return self._parameters is None

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        return self._parameters

    @property
    def options(self):
        return self._options
