import numpy as np
from scipy.optimize import fminbound

def f(x, y):
  return (x - y - 1) ** 2

lb = np.array([0, 0])
ub = np.array([1, 1])
res = fminbound(f, lb, ub)
print(res)