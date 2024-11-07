import numpy as np


def onehot(white: int, black: int) -> np.ndarray:
    label = (white, black)
    mappings = {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 2,
        (1, 1): 3,
    }
    mapping = mappings[label]
    return np.eye(4)[mapping], mapping
