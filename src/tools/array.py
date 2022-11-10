import numpy as np


def ensure_strictly_increasing(first_array: np.ndarray, second_array: np.ndarray):
    pos = 0
    x = []
    y = []
    for i in first_array:
        if i not in x:
            x = np.append(x, i)
            y = np.append(y, second_array[pos])
        pos += 1
    return x, y
