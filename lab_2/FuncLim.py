class FuncLim:
    def __init__(self, base, constrains, negative_infinity = False):
        self.base = base
        self.ni = negative_infinity
        self.constrains = [] # [func, value, -1 for <; 0 for =; 1 for >]

    def func(self, pos):
        for c in self.constrains:
            if c[2] == -1:
                if c[0](pos) < c[1]:
                    return self.base(pos)
                else:
                    if self.ni:
                        return -float("inf")
                    else:
                        return float("inf")
            if c[2] == 0:
                if c[0](pos) == c[1]:
                    return self.base(pos)
                else:
                    if self.ni:
                        return -float("inf")
                    else:
                        return float("inf")
            if c[2] == 1:
                if c[0](pos) > c[1]:
                    return self.base(pos)
                else:
                    if self.ni:
                        return -float("inf")
                    else:
                        return float("inf")