import numpy as np
from skimage.io import imread


class GTMap():
    def __init__(self, gt_map_file):
        gt_map_filename = gt_map_file.split('/')[-1]
        i1, i2, j1, j2 = [int(x) for x in gt_map_filename[12:-4].split('_')]
        self.start_i = i1
        self.start_j = j1
        gt_map = imread(gt_map_file)
        self.gt_map = gt_map
        obstacle_map = (gt_map == 0)
        explored_map = (gt_map != 127)
        grid_map = np.concatenate([explored_map[:, :, np.newaxis], obstacle_map[:, :, np.newaxis]], axis=2)
