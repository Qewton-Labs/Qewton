from enum import Enum


class EvaluationMode(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2
    TEST_AND_VALIDATION = 3
    ALWAYS = 4
    NEVER = 5


# class Optimization:
#     """Should we use pytorch lightning here?"""

#     def __init__(self):
#         super().__init__()
#         # this should be visualized seperately from the solution approach (''algorithm''),
#         # and only change the state of the solution approach once the optimization is done
#         self.constraints = (
#             ...
#         )  # state of constraints decides whether its an objective function or only tracked..

#     def get_hyperparameters(self):
#         return ...


# class SingleOptimization(Optimization):
#     def __init__(self):
#         super().__init__()
#         # optimization with a single set of hparameters


# class GridSearchOptimization(Optimization):
#     def __init__(self):
#         super().__init__()
#         # optimization over a grid of hparameters
#         for param in self.get_hyperparameters():
#             pass
