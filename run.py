from mtsoo import *
from cea import cea
from mfea import mfea
from mfeaii import mfeaii
from scipy.io import savemat

def callback(t, population, skill_factor):
  if t % 100 == 0:
    savemat('data/dump/%d.mat' % t, {'population':population, 'skill_factor':skill_factor})

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
