import os
import torch
import pandas as pd
import json
import copy
import imageio
import ffmpeg
import tarfile
import jxlpy
import numpy as np
from glob import glob
from pathlib import Path
from decord import VideoReader
from decord import cpu

class BaseDataset():
    '''
    Base class for all cyber datasets, provides some common functionality.
    '''
    def __init__(self, path):
        self.dataset_path = path
        self.episodes_discription = self.get_episode_description(path)
    
    @staticmethod
    def get_episode_description(dataset_path):
        '''
        Load the episode description for a given dataset.

        Parameters
        ----------
        dataset_path: str
            The path to the dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the episode_id, duration, num_modalities, metadata and modalities for each episode.
        '''
        episode_discription_path = os.path.join(dataset_path, 'episodes.csv')
        if not os.path.exists(episode_discription_path):
            return pd.DataFrame(columns=['episode_id', 'duration', 'num_modalities', 'metadata', 'modalities'])
        episodes_discription = pd.read_csv(episode_discription_path)
        def convert_json(data):
            return json.loads(data.replace(';', ',').replace('\'', '"'))
        episodes_discription['modalities'] = episodes_discription['modalities'].apply(convert_json)
        episodes_discription['metadata'] = episodes_discription['metadata'].apply(convert_json)
        return episodes_discription

    @classmethod
    def load_all_modalities(cls, episode_description, dataset_path):
        '''
        Load timestamps and data of all modalities for a given episode.

        Parameters
        ----------
        episode_description: dict
            A dictionary containing the episode_id, modalities and metadata of the episode to load.
        data_path: str
            The path to the dataset.

        Returns
        -------
        modalities: dict
            A dictionary mapping modality names to appropriately formatted data.

        '''
        modalities = {}
        episode_id = episode_description['episode_id']
        modalities_description = episode_description['modalities']
        for modality_name, _ in modalities_description.items():
            modality = cls.load_modality_data(episode_id, modality_name, dataset_path)
            modalities[modality_name] = modality
        return modalities
    
    @classmethod
    def load_all_modalities_timestamps(cls, episode_description, dataset_path):
        """
        Load all timestamps for a given episode. This is useful for matching up modalities.
        To save time and memory, we only load the timestamps for each modality without loading the actual data.

        Parameters
        ----------
        episode_description : dict
            A dictionary containing the episode_id, modalities and metadata of the episode to load.
        dataset_path : str
            The path to the dataset.

        Returns
        -------
        dict
            A dictionary mapping modality names to timestamps.
        """
        timestamps = {}
        episode_id = episode_description['episode_id']
        modalities_description = episode_description['modalities']
        for modality_name, _ in modalities_description.items():
            timestamps[modality_name] = cls.load_modality_timestamps(episode_id, modality_name, dataset_path)
        return timestamps

    @staticmethod
    def load_modality_data(episode_id, modality_name, dataset_path):
        """
        Load a single modality for a given episode.

        Parameters
        ----------
        episode_id : str
            The id of the episode to load.
        modality_name : str
            The name of the modality to load.
        dataset_path: str
            The path to the dataset.

        Returns
        -------
        dict:
            A dictionary containing the timestamps and data for the modality. For column-based data, the dictionary will also contain the column names.
        """
        # query file name for the given modality
        modality_file = glob(os.path.join(dataset_path, modality_name, f"{episode_id}*"))[0]
        if modality_file.split('.')[-1] == 'mp4':
            # load video
            vmetadata = ffmpeg.probe(modality_file)['format']['tags']['comment']
            video_ts = vmetadata.strip('[').strip('\'').strip(']').replace("'", "").split(',')
            video_ts = [float(v) for v in video_ts]
            vframes = VideoReader(modality_file, ctx=cpu(0))[:].asnumpy()
            return {'timestamps': video_ts, 'frames': vframes}
        
        elif modality_file.split('.')[-1] == 'csv':
            # load csv
            raw_data = pd.read_csv(modality_file)
            ts = raw_data['timestamp']
            column_names = raw_data.columns
            data = raw_data.drop(columns=['timestamp']).to_numpy()
            return {'timestamps': ts, 'data': data, 'column_names': column_names}
        
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
                            timestamps.append(float(Path(member.name).stem))
                    elif member.name.endswith('.png'):
                        png_file = tar.extractfile(member)
                        if png_file:
                            # Decode the PNG image from the tar file
                            image = imageio.imread(png_file)
                            data.append(np.array(image))
                            timestamps.append(float(Path(member.name).stem))
            
            return {'timestamps': timestamps, 'data': np.array(data)}

    @staticmethod
    def load_modality_timestamps(episode_id, modality_name, dataset_path):
        """
        Load timestamps for a given modality for a given episode.

        Parameters
        ----------
        episode_id : str
            The id of the episode to load.
        modality_name : str
            The name of the modality to load.
        dataset_path : str
            The path to the dataset.

        Returns
        -------
        list
            A list of timestamps.
        """
        # query file name for the given modality
        modality_file = glob(os.path.join(dataset_path, modality_name, f"{episode_id}*"))[0]
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
            timestamps = []
            with tarfile.open(modality_file, 'r') as tar:
                members = tar.getmembers()
                for member in members:
                    if member.name.endswith('.jxl') or member.name.endswith('.png'):
                        timestamps.append(float(Path(member.name).stem))
            return timestamps

    @staticmethod
    def match_timestamps(times_a, times_b):
        """
        Match timestamps from two modalities.

        Parameters
        ----------
        times_a : list
            A list of timestamps.
        times_b : list
            A list of timestamps.

        Returns
        -------
        tuple
            A tuple containing the following:
            - matches_a: A list of indices into `times_b` for each timestamp in `times_a`.
            - matches_b: A list of indices into `times_a` for each timestamp in `times_b`.
            - diffs_a: A list of differences between each timestamp in `times_a` and its match in `times_b`.
            - diffs_b: A list of differences between each timestamp in `times_b` and its match in `times_a`.
        """
        i, j = 0, 0
        matches_a = [-1] * len(times_a)
        matches_b = [-1] * len(times_b)
        diffs_a = [float('inf')] * len(times_a)
        diffs_b = [float('inf')] * len(times_b)
        while i < len(times_a) and j < len(times_b):
            curdiff = abs(times_a[i] - times_b[j])
            if curdiff < diffs_a[i]:
                diffs_a[i] = curdiff
                matches_a[i] = j
            if curdiff < diffs_b[j]:
                diffs_b[j] = curdiff
                matches_b[j] = i
            if times_a[i] < times_b[j]:
                i += 1
            else:
                j += 1
        if i < len(times_a):
            matches_a[i:] = [j] * (len(times_a) - i)
            diffs_a[i:] = [abs(times_a[i] - times_b[-1])] * (len(times_a) - i)
        if j < len(times_b):
            matches_b[j:] = [i] * (len(times_b) - j)
            diffs_b[j:] = [abs(times_a[-1] - times_b[j])] * (len(times_b) - j)
        return matches_a, matches_b, diffs_a, diffs_b
        

    def __len__(self):
        return len(self.episodes_discription)

    def __getitem__(self, idx):
        raise NotImplementedError

class BaseMultifolderDataset(BaseDataset):
    '''
    Base class for all cyber datasets that are split into multiple folders, provides some common functionality.
    '''
    def __init__(self, path):
        self.path = path
        self.subpaths = os.listdir(path)
        self.episodes_description = []
        for subpath in self.subpaths:
            dataset_path = os.path.join(path, subpath)
            subepisodes_description = self.get_episode_description(dataset_path)
            subepisodes_description['dataset_path'] = dataset_path
            self.episodes_description.append(subepisodes_description)
        self.episodes_description = pd.concat(self.episodes_description, ignore_index=True)