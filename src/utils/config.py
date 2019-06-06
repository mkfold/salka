from ruamel.yaml import YAML

def load_config(config):
    '''
    Convenience function for loading YAML configuration files.
    This exists mainly to keep the training/evaluation code clean.
    '''
    yaml = YAML()
    return yaml.load(open(config, 'r'))

