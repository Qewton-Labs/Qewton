import numpy as np

from qewton.visualization.plots.result import Column


class TestColumn:
    def test_selecting_rows_keeps_values_labels_and_log_scale(self):
        column = Column(values=np.array([1.0, 10.0, 100.0]), labels=None, log_scale=True)
        selected = column[np.array([True, False, True])]
        assert np.array_equal(selected.values, [1.0, 100.0])
        assert selected.log_scale is True
        assert len(selected) == 2

    def test_selecting_rows_of_a_categorical_column_keeps_its_labels(self):
        column = Column(values=np.array([0, 1, 0]), labels=["a", "b"])
        selected = column[np.array([False, True, True])]
        assert selected.labels == ["a", "b"]
        assert np.array_equal(selected.values, [1, 0])
