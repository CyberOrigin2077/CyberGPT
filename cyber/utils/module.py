import torch
import safetensors.torch as safetorch


def load_statedict_from_file(path: str) -> dict:
    '''
    loads either a pytorch ckpt dict or a safetensor from a file
    '''
    if '.ckpt' in path:
        sd = torch.load(path, map_location="cpu")["state_dict"]
    elif '.safetensor' in path:
        sd = safetorch.load_file(path)
    return sd
