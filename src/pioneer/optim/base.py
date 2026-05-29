from enum import Enum


class EvaluationPhase(Enum):
    """A marker to distinguish between different behaviors in the training
    process.

    TRAIN
        The behavior while training. May also mean that this quantity is
        only checked/evaluated when run in the training phase.
    TEST
        The behavior while testing.
    VALIDATION
        The behavior while validating.
    ALWAYS
        Always evaluated independent of the current phase.
    NEVER
        Never automatically used. Such nodes are only evaluated when called by the
        user themselves.
    """

    TRAIN = 0
    TEST = 1
    VALIDATION = 2
    ALWAYS = 3
    NEVER = 4
