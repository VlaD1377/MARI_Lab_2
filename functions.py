import numpy as np

def dixon_price(x):
    x = np.asarray(x)
    result = (x[0] - 1)**2 + sum((i + 1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, len(x)))
    return result