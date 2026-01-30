from ..configurations.variables import Variable


# TODO: General domain classes to define underlying geometry, maybe later include again
# func. such as union, differences,....

# TODO: First idea for domains would be mainly in the discrete case to know what
# structure the data has, e.g. grid, mesh,....
# And also maybe be helpful if all of the batch data is defined on the same domain,
# The user does not have to copy the data by himself into each batch entry, but we
# can somehow handle this. Would mean the domain is somehow connected to the
# dataset class?


class Domain:

    def __init__(self, variable: Variable, bounding_box: list[tuple[float, float]]):
        assert variable.dim == len(
            bounding_box
        ), "Dimension of variable must match length of bounding box."
        self.bounding_box = bounding_box
        self.variable = variable

    @property
    def dim(self) -> int:
        return self.variable.dim
