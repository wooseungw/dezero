if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero.utils import plot_dot_graph
import numpy as np
from dezero import Variable


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(0.0))
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad)
x.name = 'x'
y.name = 'y'
z.name = 'z'
# Plot the computational graph
plot_dot_graph(z, verbose=True, to_file='goldstein.png')


