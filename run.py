from mtsoo import *
from cea import cea
from mfea import mfea
from mfeaii import mfeaii

def callback(t, population, skill_factor):
  if t % 100 == 0:
    np.save('data/dump/population_%d.npy' % t, population)
    np.save('data/dump/skill_factor_%d.npy' % t, skill_factor)

def main():
  config = load_config()
  functions = CI_HS().functions

  for exp_id in range(config['repeat']):
    print('[+] EA - %d/%d' % (exp_id, config['repeat']))
    cea(functions, callback)
    # print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
    # mfea(functions, callback)
    # print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
    # mfeaii(functions, callback)

if __name__ == '__main__':
  main()
