import os
import logging
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import imageio
import ffmpeg
import tarfile
import jxlpy
import numpy as np
from glob import glob
from decord import VideoReader
from decord import cpu

from cyber.dataset.utils import match_timestamps


class BaseCyberDataset(Dataset):
    '''
    Base class for all cyber datasets, provides some common functionality.
    '''
    def __init__(self, dataset_path):
        '''
        Initialize the dataset.
        '''
        self.episode_discription = self.get_episodes_description(dataset_path)

    @staticmethod
    def get_episodes_description(dataset_path: str) -> pd.DataFrame:
        '''
        Get the description of all episodes in the dataset.

        Args:
            dataset_path(str): the path to the dataset

        Returns:
            pandas.DataFrame: the description of all episodes in the dataset
        '''
        episode_discription_path = os.path.join(dataset_path, 'episodes.csv')
        episodes_discription = pd.read_csv(episode_discription_path)

        def convert_json(data):
            return json.loads(data.replace(';', ',').replace('\'', '"'))
        episodes_discription['modalities'] = episodes_discription['modalities'].apply(convert_json)
        episodes_discription['metadata'] = episodes_discription['metadata'].apply(convert_json)
        episodes_discription['path'] = dataset_path
        return episodes_discription

    @classmethod
    def _load_all_modalities_data(cls, episode_description: pd.Series) -> dict:
        '''
        Load all modalities for a given episode.

        Args:
            episode_description (pandas.Series): the description of the episode to load

        Returns:
            dict: a dictionary mapping modality names to appropriately formatted data
        '''
        modalities = {}
        episode_id = episode_description['episode_id']
        modalities_description = episode_description['modalities']
        for modality_name, modality_description in modalities_description.items():
            modality = cls._load_modality_data(episode_description['path'],
                                               episode_id,
                                               modality_name,
                                               modality_description)
            modalities[modality_name] = modality
        return modalities

    @classmethod
    def _load_all_modalities_timestamps(cls, episode_description: pd.Series) -> dict[str, list]:
        '''
        Load all timestamps for a given episode. This is useful for matching up modalities.
        To save time and memory, we only load the timestamps for each modality without loading the actual data.

        Args:
            episode_description (pandas.Series): the description of the episode to load

        Returns:
            dict[str, list]: a dictionary mapping modality names to timestamps
        '''
        timestamps = {}
        episode_id = episode_description['episode_id']
        modalities_description = episode_description['modalities']
        for modality_name in modalities_description.keys():
            timestamps[modality_name] = cls._load_modality_timestamps(episode_description['path'],
                                                                      episode_id,
                                                                      modality_name)
        return timestamps

    @staticmethod
    def _load_modality_data(dataset_path: str, episode_id: str, modality_name: str, modality_description: dict) -> dict:
        '''
        Load a single modality for a given episode.

        Args:
            dataset_path(str): the path to the dataset
            episode_id(str): the id of the episode to load
            modality_name(str): the name of the modality to load
            modality_description(dict): the description of the modality to load

        Returns:
            dict: formatted data with timestamps
        '''
        # query file name for the given modality
        modality_file = glob(os.path.join(dataset_path, modality_name, f"{episode_id}*"))
        assert (len(modality_file) == 1), f"Expected 1 file for modality {modality_name} in episode {episode_id}, found {len(modality_file)}"
        modality_file = modality_file[0]
        if modality_file.split('.')[-1] == 'mp4':
            # load video
            vmetadata = ffmpeg.probe(modality_file)['format']['tags']['comment']
            video_ts = vmetadata.strip('[').strip('\'').strip(']').replace("'", "").split(',')
            video_ts = [float(v) for v in video_ts]
            vframes = VideoReader(modality_file, ctx=cpu(0))[:].asnumpy()
            return {'timestamps': video_ts, 'data': vframes}

        elif modality_file.split('.')[-1] == 'csv':
            # load csv
            pd_csv = pd.read_csv(modality_file)
            timestamps = list(pd_csv['timestamp'])
            pd_csv.drop(columns=['timestamp'], inplace=True)
            data = pd_csv.to_numpy(dtype=np.float32)
            column_tags = list(pd_csv.columns)
            return {'timestamps': timestamps, 'data': data, 'column_tags': column_tags}

        elif modality_file.split('.')[-1] == 'tar':
            # tar format: timestamp.[fileextension]
            data = []
            timestamps = []
            with tarfile.open(modality_file, 'r') as tar:
                members = tar.getmembers()
                for member in members:
                    if member.name.endswith('.jxl'):
                        jxl_file = tar.extractfile(member)
                        if jxl_file:
                            # Decode the JPEG XL image from the tar file
                            image = jxlpy.JxlDecoder(jxl_file)
                            image_data = image.get_image()
                            data.append(np.array(image_data))
                            timestamps.append(float(member.name.split('/')[-1][:-4]))
                    elif member.name.endswith('.png'):
                        png_file = tar.extractfile(member)
                        if png_file:
                            # Decode the PNG image from the tar file
                            image = imageio.imread(png_file)
                            data.append(np.array(image))
                            timestamps.append(float(member.name.split('/')[-1][:-4]))

            sorted_indices = np.argsort(timestamps)
            # timestamps = timestamps[sorted_indices]
            timestamps = sorted(timestamps)
            # data = data[sorted_indices]
            data = np.array(data)[sorted_indices]
            return {'timestamps': timestamps, 'data': data}

    @staticmethod
    def _load_modality_timestamps(dataset_path: str, episode_id: str, modality_name: str) -> list:
        '''
        Load the timestamps for a given modality.

        Args:
            datset_path(str): the path to the dataset
            episode_id(str): the id of the episode to load
            modality_name(str): the name of the modality to load

        Returns:
            list: timestamps for the given modality
        '''
        # query file name for the given modality
        modality_file = glob(os.path.join(dataset_path, modality_name, f"{episode_id}*"))
        assert (len(modality_file) == 1)
        modality_file = modality_file[0]
        if modality_file.split('.')[-1] == 'mp4':
            # load video
            vmetadata = ffmpeg.probe(modality_file)['format']['tags']['comment']
            video_ts = vmetadata.strip('[').strip('\'').strip(']').replace("'", "").split(',')
            video_ts = [float(v) for v in video_ts]
            return video_ts

        elif modality_file.split('.')[-1] == 'csv':
            # load csv
            return list(pd.read_csv(modality_file)['timestamp'])

        elif modality_file.split('.')[-1] == 'tar':
            # tar format: timestamp.[fileextension]
            with tarfile.open(modality_file, 'r') as tar:
                members = tar.getmembers()
                timestamps = [float(member.name.split('/')[-1][:-4]) for member in members
                                if member.name.endswith('.jxl') or member.name.endswith('.png')]
            return timestamps

    def __len__(self):
        return len(self.episode_discription)

    def __getitem__(self, idx):
        data = self._load_all_modalities_data(self.episode_discription.loc[idx])
        # match timestamps to color modality
        color_ts = data['color']['timestamps']
        matched_data = {}
        for modality_name, modality_data in data.items():
            if modality_data['data'].dtype == np.uint16:
                # uint16 is not supported by torch, raise a warning
                logging.warning(f"modality :{modality_name} has dtype uint16 not supported by torch, skipping")
                continue
            if modality_name == 'color':
                keep_indices = list(range(len(modality_data['timestamps'])))
            else:
                ts = modality_data['timestamps']
                keep_indices, _, _, _ = match_timestamps(color_ts, ts)
            matched_data[modality_name] = torch.tensor(modality_data['data'][keep_indices])
        return matched_data


def simple_stack_collate_fn(items: dict) -> dict[str, torch.Tensor]:
    '''
    A naive collate function that stacks all modalities into a single tensor.

    Args:
        items(dict): a dictionary mapping modality names to data

    Returns:
        dict[str, torch.Tensor]: a dictionary mapping modality names to stacked data tensors
    '''
    stacked_data = {}
    for modality_name in items[0].keys():
        stacked_data[modality_name] = torch.concat([d[modality_name] for d in items])
    return stacked_data
