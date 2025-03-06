import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

    def __call__(self, ):
        return self.data*2

x = Variable("1")
print(x.data)

x.data = np.array(2.0)
print(x.data)

y = Variable(-1)
print(y.data)