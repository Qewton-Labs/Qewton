from enum import Enum


class EvaluationMode(Enum):
    """A marker to distinguish between different behaviors in the training
    process.

    TRAIN
        The behavior while training. May also mean that this quantity is
        only checked/evaluated when run in the training phase.
    TEST
        The behavior while testing.
    VALIDATION
        The behavior while validating.
    TUNE
        Only evaluated for parameter tuning.
    ALWAYS
        Always evaluated independent of the current phase.
    NEVER
        Never automatically used. Such nodes are only evaluated when called by the
        user themselves.
    """

    TRAIN = 0
    TEST = 1
    VALIDATION = 2
    TUNE = 3
    ALWAYS = 4
    NEVER = 5
