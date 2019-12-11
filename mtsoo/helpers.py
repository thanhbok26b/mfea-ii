import yaml
from .operators import get_best_individual
from scipy.optimize import OptimizeResult

def load_config(path='config.yaml'):
  with open(path) as fp:
    config = yaml.load(fp)
  return config

def get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message):
  K = len(set(skill_factor))
  N = len(population) // 2
  results = []
  for k in range(K):
    result         = OptimizeResult()
    x, fun         = get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, k)
    result.x       = x
    result.fun     = fun
    result.message = message
    result.nit     = t
    result.nfev    = (t + 1) * N
    results.append(result)
  return results
