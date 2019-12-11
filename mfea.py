from mtsoo import *

config = load_config()

def mfea(functions, config, callback=None):
  # unpacking hyper-parameters
  K = len(functions)
  N = config['pop_size'] * K
  D = config['dimension']
  T = config['num_iter']
  sbxdi = config['sbxdi']
  pmdi  = config['pmdi']
  pswap = config['pswap']
  rmp   = config['rmp']

  # initialize
  population = np.random.rand(2 * N, D)
  skill_factor = np.array([i % K for i in range(2 * N)])
  factorial_cost = np.full([2 * N, K], np.inf)
  scalar_fitness = np.empty([2 * N])

  # evaluate
  for i in range(2 * N):
    sf = skill_factor[i]
    factorial_cost[i, sf] = functions[sf](population[i])
  scalar_fitness = calculate_scalar_fitness(factorial_cost)

  # sort 
  sort_index = np.argsort(scalar_fitness)[::-1]
  population = population[sort_index]
  skill_factor = skill_factor[sort_index]
  factorial_cost = factorial_cost[sort_index]

  # evolve
  iterator = trange(T)
  for t in iterator:
    # permute current population
    permutation_index = np.random.permutation(N)
    population[:N] = population[:N][permutation_index]
    skill_factor[:N] = skill_factor[:N][permutation_index]
    factorial_cost[:N] = factorial_cost[:N][permutation_index]
    factorial_cost[N:] = np.inf

    # select pair to crossover
    for i in range(0, N, 2):
      p1, p2 = population[i], population[i + 1]
      sf1, sf2 = skill_factor[i], skill_factor[i + 1]

      # crossover
      if sf1 == sf2:
        c1, c2 = sbx_crossover(p1, p2, sbxdi)
        c1 = mutate(c1, pmdi)
        c2 = mutate(c2, pmdi)
        c1, c2 = variable_swap(c1, c2, pswap)
        skill_factor[N + i] = sf1
        skill_factor[N + i + 1] = sf1
      elif sf1 != sf2 and np.random.rand() < rmp:
        c1, c2 = sbx_crossover(p1, p2, sbxdi)
        c1 = mutate(c1, pmdi)
        c2 = mutate(c2, pmdi)
        # c1, c2 = variable_swap(c1, c2, pswap)
        if np.random.rand() < 0.5: skill_factor[N + i] = sf1
        else: skill_factor[N + i] = sf2
        if np.random.rand() < 0.5: skill_factor[N + i + 1] = sf1
        else: skill_factor[N + i + 1] = sf2
      else:
        p2  = find_relative(population, skill_factor, sf1, N)
        c1, c2 = sbx_crossover(p1, p2, sbxdi)
        c1 = mutate(c1, pmdi)
        c2 = mutate(c2, pmdi)
        c1, c2 = variable_swap(c1, c2, pswap)
        skill_factor[N + i] = sf1
        skill_factor[N + i + 1] = sf1

      population[N + i, :], population[N + i + 1, :] = c1[:], c2[:]

    # evaluate
    for i in range(N, 2 * N):
      sf = skill_factor[i]
      factorial_cost[i, sf] = functions[sf](population[i])
    scalar_fitness = calculate_scalar_fitness(factorial_cost)

    # sort
    sort_index = np.argsort(scalar_fitness)[::-1]
    population = population[sort_index]
    skill_factor = skill_factor[sort_index]
    factorial_cost = factorial_cost[sort_index]
    scalar_fitness = scalar_fitness[sort_index]

    c1 = population[np.where(skill_factor == 0)][0]
    c2 = population[np.where(skill_factor == 1)][0]

    # optimization info
    message = {'algorithm': 'mfea', 'rmp':rmp}
    results = get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message)
    if callback:
      callback(results)

    desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join('{:0.6f}'.format(res.fun) for res in results), message)
    iterator.set_description(desc)
