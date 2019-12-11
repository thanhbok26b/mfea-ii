from mtsoo import *
from cea import cea
from mfea import mfea
from mfeaii import mfeaii
from scipy.io import savemat

def callback(res):
  pass

def main():
  config = load_config()
  functions = CI_HS().functions

  for exp_id in range(config['repeat']):
    print('[+] EA - %d/%d' % (exp_id, config['repeat']))
    cea(functions, config, callback)
    print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
    mfea(functions, config, callback)
    print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
    mfeaii(functions, config, callback)

if __name__ == '__main__':
  main()
