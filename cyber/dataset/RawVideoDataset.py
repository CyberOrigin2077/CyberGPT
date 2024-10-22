import torch
import pandas as pd
import numpy as np
from cyber.dataset.CyberDataset import BaseCyberDataset


class RawVideoDataset(BaseCyberDataset):
    '''
    Base class for all cyber datasets, provides some common functionality.
    '''
    def __init__(self, dataset_path, only_color=True):
        super().__init__(dataset_path)
        self.only_color = only_color

        # self.episodes_description = self.get_episodes_description(dataset_path)

    @classmethod
    def _load_all_modalities_data(cls, episode_description: pd.Series, only_color=True) -> dict:
        modalities = {}
        episode_id = episode_description['episode_id']
        modalities_description = episode_description['modalities']
        for modality_name, modality_description in modalities_description.items():
            if only_color and modality_name != 'color':
                continue
            modality = cls._load_modality_data(episode_description['path'], episode_id, modality_name, modality_description)
            modalities[modality_name] = modality
        return modalities

    def __getitem__(self, idx):
        data = self._load_all_modalities_data(self.episode_discription.loc[idx], self.only_color)
        matched_data = {}
        for modality_name, modality_data in data.items():
            if modality_data['data'].dtype == np.uint16:
                print(f"Warning: {modality_name} has dtype uint16, this is not supported by torch, skipping this modality")
                continue
            matched_data[modality_name] = torch.tensor(modality_data['data'])
        return matched_data
