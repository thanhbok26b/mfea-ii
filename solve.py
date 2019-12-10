from mtsoo import *
import matplotlib.pyplot as plt

D = 50
K = 2
N = 50

population   = np.load('data/population.npy')
skill_factor = np.load('data/skill_factor.npy')

subpops = get_subpops(population, skill_factor)

# add noise and build probabilistic models
models = []
for k in range(K):
  subpop            = subpops[k]
  num_sample        = len(subpop)
  num_random_sample = int(np.floor(0.1 * num_sample))
  rand_pop          = np.random.rand(num_random_sample, D)
  mean              = np.mean(np.concatenate([subpop, rand_pop]), axis=0)
  std               = np.std(np.concatenate([subpop, rand_pop]), axis=0)
  models.append(Model(mean, std, num_sample))

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
plt.show()

# rmp = fminbound(lambda rmp: log_likelihood(rmp, probmatrix, K), 0, 1)

# print(rmp)