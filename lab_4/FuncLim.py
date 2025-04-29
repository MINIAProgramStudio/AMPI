from collections.abc import Iterable
import numpy as np

class FuncLim:
    def __init__(self, base, constrains, negative_infinity = False):
        self.base = base
        self.ni = negative_infinity
        self.constrains = constrains # [func, value, -1 for <; 0 for =; 1 for >]

    def func(self, pos):
        output = self.base(pos)
        for c in self.constrains:
            if c[2] == 0:
                if self.ni:
                    output += float("-inf")**np.not_equal(c[0](pos), c[1])
                    break
                else:
                    output += float("inf") ** np.not_equal(c[0](pos), c[1])
                    break
            elif c[2] == -1:
                if self.ni:
                    output += float("-inf")**np.greater(c[0](pos), c[1])
                    break
                else:
                    output += float("inf") ** np.greater(c[0](pos), c[1])
                    break
            elif c[2] == -2:
                if self.ni:
                    output += float("-inf")**np.greater_equal(c[0](pos), c[1])
                    break
                else:
                    output += float("inf") ** np.greater_equal(c[0](pos), c[1])
                    break
            elif c[2] == 1:
                if self.ni:
                    output += float("-inf") ** np.greater(c[1], c[0](pos))
                    break
                else:
                    output += float("inf") ** np.greater(c[1], c[0](pos))
                    break
            elif c[2] == 2:
                if self.ni:
                    output += float("-inf") ** np.greater_equal(c[1], c[0](pos))
                    break
                else:
                    output += float("inf") ** np.greater_equal(c[1], c[0](pos))
                    break
        return output
