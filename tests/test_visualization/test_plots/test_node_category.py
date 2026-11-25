"""`node_category`: what a node is, for its colors, read from its class."""

import pytest

from qewton.algorithms import FCN
from qewton.constraints import MSEConstraint
from qewton.data.dataloaders.base import DataLoader
from qewton.visualization import NodeCategory, node_category


@pytest.mark.parametrize(
    "node_class, category",
    [
        (MSEConstraint, NodeCategory.CONSTRAINT),
        (DataLoader, NodeCategory.DATA),
        (FCN, NodeCategory.MODEL),
    ],
)
def test_the_category_of_a_node_class(node_class, category):
    assert node_category(node_class) is category


def test_a_node_has_the_category_of_its_class():
    assert node_category(MSEConstraint()) is NodeCategory.CONSTRAINT
