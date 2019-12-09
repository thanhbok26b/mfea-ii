from mtsoo import *
from cea import cea
from mfea import mfea
from mfeaii import mfeaii

config = load_config()
functions = NI_HS().functions

for exp_id in range(config['repeat']):
  # print('[+] EA - %d/%d' % (exp_id, config['repeat']))
  # cea(functions)

  # print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
  # mfea(functions)

  print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
  mfeaii(functions)
