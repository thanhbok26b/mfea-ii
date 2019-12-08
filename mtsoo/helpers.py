import yaml

def load_config():
  with open('data/config.yaml') as fp:
    config = yaml.load(fp)
  return config