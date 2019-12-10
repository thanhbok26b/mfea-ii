from mtsoo import *
from cea import cea
from mfea import mfea
from mfeaii import mfeaii

def callback(population, skill_factor):
  # np.save('data/population.npy', population)
  # np.save('data/skill_factor', skill_factor)
  pass

def main():
  config = load_config()
  functions = NI_MS().functions

  for exp_id in range(config['repeat']):
    # print('[+] EA - %d/%d' % (exp_id, config['repeat']))
    # cea(functions)
    # print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
    # mfea(functions)
    print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
    mfeaii(functions, callback)

if __name__ == '__main__':
  main()
