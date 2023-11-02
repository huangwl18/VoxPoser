"""load YAML config file"""
import os
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_config(env=None, config_path=None):
    assert env is None or config_path is None, 'env and config_path cannot be both specified'
    if config_path is None:
        assert env.lower() == 'rlbench'
        config_path = './configs/rlbench_config.yaml'
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    config = load_config(config_path)
    # wrap dict such that we can access config through attribute
    class ConfigDict(dict):
        def __init__(self, config):
            """recursively build config"""
            self.config = config
            for key, value in config.items():
                if isinstance(value, str) and value.lower() == 'none':
                    value = None
                if isinstance(value, dict):
                    self[key] = ConfigDict(value)
                else:
                    self[key] = value
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value
        def __delattr__(self, key):
            del self[key]
        def __getstate__(self):
            return self.config
        def __setstate__(self, state):
            self.config = state
            self.__init__(state)
    config = ConfigDict(config)
    return config

def main():
    config = get_config(config_path='./configs/rlbench_config.yaml')
    print(config)

