import numpy as np

class DixonPriceExact:
    def __init__(self, d):
        self.d = d

    def solve(self):
        x = np.zeros(self.d)
        x[0] = 1.0
        for i in range(1, self.d):
            x[i] = np.sqrt(x[i - 1] / 2.0)
        return x, 0.0
