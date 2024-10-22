import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from .utils import ConcatMemmap


class RawTokenDataset(TorchDataset):
    """ Loads raw uint32 tokens as memmap-backed array """
    def __init__(
        self,
        data_dir,
        window_size,
        stride=1,
        filter_overlaps=False
    ):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g. frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
        """
        self.window_size, self.stride = window_size, stride
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride

        data_dir = Path(data_dir)
        # list all files in the data directory that ends with .json
        bin_files = list(data_dir.glob("*.json"))
        # remove "medatadata_" from the file stems:
        ids = [file.stem.replace("metadata_", "") for file in bin_files]

        # sanity check
        for idx in ids:
            for name in ["segment_ids_", "videos_"]:
                assert data_dir / f"{name}{idx}.bin", f"Missing {name}{idx}.bin"
        # the metadata of the tokens need to match
        with open(data_dir / f"metadata_{ids[0]}.json") as f:
            metadata = json.load(f)
            self.metadata = {}
            self.token_dtype = np.dtype(metadata.get("token_dtype", "uint32"))
            for key in ("s", "vocab_size", "hz"):
                self.metadata[key] = metadata[key]
        for idx in ids[1:]:
            with open(data_dir / f"metadata_{idx}.json") as f:
                metadata = json.load(f)
                assert np.dtype(metadata.get("token_dtype", "uint32")) == self.token_dtype, "Token dtypes must match"
                for key in ("s", "vocab_size", "hz"):
                    assert metadata[key] == self.metadata[key], f"metadata[{key}] must match"

        self.data = ConcatMemmap(dtype=self.token_dtype)
        self.segment_ids = ConcatMemmap(dtype=np.int32)
        for idx in ids:
            with open(data_dir / f"metadata_{idx}.json") as f:
                metadata = json.load(f)

            shape = (metadata["num_images"], metadata["s"], metadata["s"])
            video_tokens_path, segment_ids_path = [data_dir / f"{name}_{idx}.bin"
                                                        for name in ["videos", "segment_ids"]]
            self.data.append(np.memmap(video_tokens_path, dtype=self.token_dtype, mode="r", shape=shape))

            self.segment_ids.append(np.memmap(
                segment_ids_path,
                dtype=np.int32,
                mode="r",
                shape=(metadata["num_images"],)
            ))

        self.valid_start_inds = []
        for start_ind in range(len(self.data) - self.video_len):
            # ! NOTE that we need to filter interupts to avoid slicing across different memmaps
            # Assuming `segment_ids` is monotonically increasing, a sequence is interrupted
            # if the first and last frames have different segment ids.
            if not (self.segment_ids[start_ind] != self.segment_ids[start_ind + self.video_len]):
                self.valid_start_inds.append(start_ind)

        if filter_overlaps:
            # Instead of using a sliding window, use each frame at most once
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {start_ind - i * self.stride for i in range(1, self.window_size)}
                # all sequences from `overlapping_start_inds` will also contain `start_ind`,
                # so exclude sequence starting from `start_ind` if any of `overlapping_start_inds` is already being used
                for existing_start_ind in filtered_start_inds[-self.window_size * self.stride:]:
                    # Bound could be improved
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds

    def __len__(self):
        return len(self.valid_start_inds)

    @staticmethod
    def tokenize_action(action, action_min, action_max):
        '''
        this function tokenizes the action values
        '''
        # normalize the action values
        action = (action - action_min) / (action_max - action_min)
        # quantize the action values
        action = (action * 255.0).to(torch.uint8)
        # reshape the action tensor to be 1D
        action = action.flatten()
        return action

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        start_ind = self.valid_start_inds[idx]
        x = torch.from_numpy((self.data[start_ind : start_ind + self.video_len + 1 : self.stride]).astype(np.int64))
        x = x.flatten()

        attention_mask = torch.ones_like(x)
        return {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
        }
