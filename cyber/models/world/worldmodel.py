from torch.nn import Module
from omegaconf import OmegaConf
from cyber.config import instantiate_from_config
from cyber.utils.module import load_statedict_from_file
import torch.nn as nn
import logging


class WorldModel(Module):
    '''
    Base class world models, provides some common functionality.
    All world models contains an encoder that comprehends the world state and a dynamics model that predicts the future state.
    '''
    def __init__(self, config: OmegaConf):
        '''
        Initialize the world model through a predefined configuration.
        '''
        super().__init__()
        self.encoder: nn.Module = instantiate_from_config(config.encoder)
        self.dynamic: nn.Module = instantiate_from_config(config.dynamic)

    def forward(self, x):
        '''
        Forward pass through the world model.
        '''
        encoding = self.encoder(x)
        prediction = self.dynamic(encoding)
        return prediction

    def load_weights(self, weights: dict):
        '''
        Load weights of each component.

        Args:
        weights(dict[str:str]): maps from encoder and dynamic to weight paths
        '''
        assert 'encoder' in weights, 'Missing encoder weights'
        assert 'dynamic' in weights, 'Missing dynamic weights'

        logging.info("loading weights for encoder", type(self.encoder))
        sd_encoder = load_statedict_from_file(weights['encoder'])
        missing_keys, unexpected_keys = self.encoder.load_state_dict(sd_encoder, strict=False)
        if missing_keys or unexpected_keys:
            logging.error(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        logging.info("loading weights for dynamic", type(self.dynamic))
        sd_dynamic = load_statedict_from_file(weights['dynamic'])
        missing_keys, unexpected_keys = self.dynamic.load_state_dict(sd_dynamic, strict=False)
        if missing_keys or unexpected_keys:
            logging.error(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        logging.info("all keys loaded")
