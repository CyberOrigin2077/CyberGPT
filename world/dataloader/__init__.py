'''
Copyright (c) 2024 CyberOrigin
Author: Max
'''

from torch.utils.data import DataLoader

from .fusion_dataloader import FusionDataLoader
from .online_bioslam import Online_BioSLAM_Dataloader as online_bioslam_dataloader

from ..utils.log import log_print


def make_data_loader(cfg, gpu_ids, is_inference=False, is_train=True, is_reference=True, is_pair=True):
    '''

    Args:
        cfg: parameter configs
        logger: logger handle
        gpu_ids: current gpus
        is_inference: training or inference
        is_train: in training, the data is for training or validation
        is_reference: in inference, the data is reference trajectory or query trajectory

    Returns: dataloader
    '''
    if is_inference and is_train:
        log_print("Using dataset: %s" % cfg.DATA.DATASET_NAME, "g")
    
    dataLoader = FusionDataLoader
        
    data_loader = dataLoader(cfg, is_inference, is_train, is_reference, is_pair)
    
    loader = DataLoader(data_loader, 
                            batch_size=cfg.TRAIN.INPUT.BATCH_SIZE_PER_GPU*len(gpu_ids), # if is_train else len(cfg.GPU_INDEX), 
                            num_workers= 8 if is_train else 4, 
                            pin_memory=True, 
                            shuffle=True if ~is_inference else False)

    return loader
