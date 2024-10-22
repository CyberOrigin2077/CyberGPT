from omegaconf import OmegaConf
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: OmegaConf):
    '''
    Instantiate an object from an OmegaConf configuration.

    Args:
        config (OmegaConf): the configuration to instantiate from

    Returns:
        object: the instantiated object
    '''
    if "class_path" not in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(config.get("init_args", dict()))  # noqa: C408


def get_function_from_str(string):
    module, func = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), func)
