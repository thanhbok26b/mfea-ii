from mtsoo import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
D = 50
K = 2
N = 50

t = 100
data         = loadmat('data/dump/%d.mat' % t)
population   = data['population']
skill_factor = data['skill_factor'][0]

subpops = get_subpops(population, N, skill_factor)
models  = learn_models(subpops)

k = 0
j = 1
probmatrix = [np.ones([models[k].num_sample, 2]), 
              np.ones([models[j].num_sample, 2])]

probmatrix[0][:, 0] = models[k].density(subpops[k])
probmatrix[0][:, 1] = models[j].density(subpops[k])
probmatrix[1][:, 0] = models[k].density(subpops[j])
probmatrix[1][:, 1] = models[j].density(subpops[j])

x = []
y = []

for i in range(1000):
  x.append(0.001 * i)
  y.append(log_likelihood(0.001 * i, probmatrix, K))

plt.plot(x, y)
plt.title('Generation %d' % t)
plt.show()

# rmp = fminbound(lambda rmp: log_likelihood(rmp, probmatrix, K), 0, 1)
# print(rmp)