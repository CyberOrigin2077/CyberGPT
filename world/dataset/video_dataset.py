from world.dataset.base_dataset import BaseMultifolderDataset

class VideoDataset(BaseMultifolderDataset):
    def __init__(self, path, transform=None):
        super().__init__(path)
        self.transform = transform
    
    def __getitem__(self, idx):
        # we only want the video frames (as a NxHxWxC numpy array)
        epid = self.episode_discription.loc[idx]['episode_id']
        dataset_path = self.episodes_description.loc[idx]['dataset_path']
        vframes = self.load_modality_data(epid, 'color', dataset_path)['data']
        # apply transform if any
        if self.transform:
            vframes = self.transform(vframes)
        return vframes