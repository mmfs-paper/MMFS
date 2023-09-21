import importlib
from configs.base_config import BaseConfig

def find_config_by_name(config_name):
    # load config lib by config name
    config_file = "configs." + config_name + '_config'
    config_lib = importlib.import_module(config_file)
    print(config_lib)

    # find the subclass of BaseConfig
    config = None
    target_config_name = config_name.replace('_', '') + 'config'
    target_config_name = target_config_name.lower()
    for name, cls in config_lib.__dict__.items():
        if name.lower() == target_config_name and issubclass(cls, BaseConfig):
            config = cls
    
    if config is None:
        raise Exception('No valid config found.')

    return config

def parse_config(cfg_file):
    # parse config using BaseConfig
    cfg = BaseConfig().parse_config(cfg_file)
    model_name = cfg['common']['model']

    # re-parse using specified Config class
    config = find_config_by_name(model_name)
    return config().parse_config(cfg_file)
