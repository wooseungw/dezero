if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from dezero import Variable, Function
import numpy as np
import math

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx,
    
def sin(x):
    f = Sin()
    return f(x)

def my_sin(x,threshold = 1e-4):
    y = 0
    for i in range(10000):
        c = (-1)**i / math.factorial(2*i + 1)
        t = c * x**(2*i + 1)
        y += t
        if abs(t) < threshold:
            break
    return y

x = Variable(np.array(3*np.pi/4))
y = sin(x)
y.backward()
# print("sin(x):",y.data)
# print("sin(x) grad:",x.grad)
# print("my_sin(x):",my_sin(x.data))
# print("my_sin(x) grad:",x.grad)

def rosenbrock(x, y):
    z = (1 - x)**2 + 100 * (y - x**2)**2
    return z
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.0001
iterations = 1000

# for i in range(iterations):
#     print("x0:",x0.data,"x1:",x1.data)
#     z = rosenbrock(x0, x1)
    
#     x0.cleargrad()
#     x1.cleargrad()
#     z.backward()
#     x0.data -= lr * x0.grad
#     x1.data -= lr * x1.grad

def func(x):
    return (x**4) - (x**2)

def gx2(x):
    return 12*x**2 - 2

iterations = 10
x = Variable(np.array(2.0))
for i in range(iterations):
    print("x:",x.data)
    y = func(x)
    
    x.cleargrad()
    y.backward()
    x.data -=  x.grad / gx2(x.data)