import numpy as np
from scipy.stats import multivariate_normal

mean = np.zeros([10])
cov = np.ones([10])

x = np.random.rand(100, 10)
y = multivariate_normal.pdf(x, mean=mean, cov=cov)
print(y)