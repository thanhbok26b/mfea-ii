import numpy as np
from scipy.optimize import fminbound
from scipy.stats import multivariate_normal

def sbx_crossover(p1, p2, sbxdi):
  D = p1.shape[0]
  cf = np.empty([D])
  u = np.random.rand(D)        

  cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (sbxdi + 1)))
  cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (sbxdi + 1)))

  c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
  c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)

  c1 = np.clip(c1, 0, 1)
  c2 = np.clip(c2, 0, 1)

  return c1, c2

def mutate(p, pmdi):
  mp = float(1. / p.shape[0])
  u = np.random.uniform(size=[p.shape[0]])
  r = np.random.uniform(size=[p.shape[0]])
  tmp = np.copy(p)
  for i in range(p.shape[0]):
    if r[i] < mp:
      if u[i] < 0.5:
        delta = (2*u[i]) ** (1/(1+pmdi)) - 1
        tmp[i] = p[i] + delta * p[i]
      else:
        delta = 1 - (2 * (1 - u[i])) ** (1/(1+pmdi))
        tmp[i] = p[i] + delta * (1 - p[i])
  tmp = np.clip(tmp, 0, 1)
  return tmp

def variable_swap(p1, p2, probswap):
  D = p1.shape[0]
  swap_indicator = np.random.rand(D) <= probswap
  c1, c2 = p1.copy(), p2.copy()
  c1[np.where(swap_indicator)] = p2[np.where(swap_indicator)]
  c2[np.where(swap_indicator)] = p1[np.where(swap_indicator)]
  return c1, c2

def find_relative(population, skill_factor, sf, N):
  return population[np.random.choice(np.where(skill_factor[:N] == sf)[0])]

def calculate_scalar_fitness(factorial_cost):
  return 1 / np.min(np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1, axis=1)

def get_subpops(population, skill_factor):
  K = len(set(skill_factor))
  subpops = []
  for k in range(K):
    idx = np.where(skill_factor == k)[0]
    subpops.append(population[idx, :])
  return subpops

def learn_rmp_dummy(subpops):
  K = len(subpops)
  rmp_matrix = np.full([K, K], 0.3)
  return rmp_matrix

class Model:
  def __init__(self, mean, std, num_sample):
    self.mean        = mean
    self.std         = std
    self.num_sample  = num_sample

  def density(self, subpop):
    return multivariate_normal.pdf(subpop, mean=self.mean, cov=self.std)

def log_likelihood(rmp, probmatrix, K):
  f = 0
  for k in range(2):
    for j in range(2):
      if k == j:
        probmatrix[k][:, j] = probmatrix[k][:, j] * (1 - (0.5 * (K - 1) * rmp / K))
      else:
        probmatrix[k][:, j] = probmatrix[k][:, j] * 0.5  * (K - 1) * rmp / K
    f += np.sum(-np.log(np.sum(probmatrix[k], axis=1)))
  return f

def learn_rmp(subpops, D):
  K          = len(subpops)
  rmp_matrix = np.eye(K)

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

  for k in range(K - 1):
    for j in range(k + 1, K):
      probmatrix = [np.ones([models[k].num_sample, 2]), 
                    np.ones([models[j].num_sample, 2])]
      probmatrix[0][:, 0] = models[k].density(subpops[k])
      probmatrix[0][:, 1] = models[j].density(subpops[k])
      probmatrix[1][:, 0] = models[k].density(subpops[j])
      probmatrix[1][:, 1] = models[j].density(subpops[j])

      rmp = fminbound(lambda rmp: log_likelihood(rmp, probmatrix, K), 0, 1)
      rmp = rmp + np.random.randn() * 0.01
      rmp = np.clip(rmp, 0, 1)
      rmp_matrix[k, j] = rmp
      rmp_matrix[j, k] = rmp

  return rmp_matrix

def main():
  N = 30
  D = 50

  subpops = []
  for i in range(3):
    subpops.append(np.random.rand(N, D))

  learn_rmp(subpops, D)

if __name__ == '__main__':
  main()