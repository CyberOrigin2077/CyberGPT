'''
Copyright (c) 2024 CyberOrigin
Author: Max
'''

import os
import pickle
import random
import numpy as np
import pandas as pd
from glob import glob
from sklearn.neighbors.kd_tree import KDTree
from tqdm import tqdm

from .base_dataloader import BaseDataLoader
from ..utils.log import log_print

random.seed(12345)

class FusionDataLoader(BaseDataLoader):
    def __init__(self, cfg, is_inference, is_train, is_reference, is_pair):
        super().__init__(cfg, is_inference, is_train, is_reference, is_pair)

        self.file_idxs = np.arange(0, len(self.queries.keys()))
        self.length = len(self.queries.keys())
        self.grid_edge = self.cfg.SPHERE.GRID_SIZE[1]
        self.shift_dis = int(self.grid_edge/self.cfg.ROTATE.DIV)
        self.is_pair = is_pair
        self.cfg = cfg

    def generate_pickles(self):
        #! Load pickles if exist
    
        data_dirs = sorted(glob("{}/*".format(self.dataset_dir)))
        data_dirs = [path for path in os.listdir(self.dataset_dir) if (os.path.isdir(os.path.join(self.dataset_dir, path)) and path != 'DATA' and 'traj_' in path)]        
        data_dirs = sorted([path for path in data_dirs if int(path.split('_')[-1]) in self.cfg.DATA.TRAIN_LIST])
        pcks = []
        start_index = 0
        for data_dir in data_dirs:
            pck_name = "{}.pkl".format(data_dir)
            pcks.append(pck_name)
            pck_file = os.path.join(self.dataset_dir, pck_name)
            if os.path.exists(pck_file):
                continue
            data_df = self.get_df(self.dataset_dir, data_dir, is_shuffle=False)
            start_index = self.construct_query_dict(data_df, pck_file, start_index)
            if self.is_train:
                log_print("Generated pickles {}".format(start_index), "g")
        return pcks

    def get_from_folder(self, data_path, index):

        all_file_id = []
        pose_data = glob(data_path+'/*_pose.npy')
        for file_name in pose_data:
            all_file_id.append(file_name.split('_pose')[-2])
        all_file_id.sort()
        all_data_df = pd.DataFrame(all_file_id, columns=["file"])
        all_data_df["pcd_position_x"] = all_data_df["file"].apply(
            lambda x: np.load(x + '_pose.npy')[0])
        all_data_df["pcd_position_y"] = all_data_df["file"].apply(
            lambda x: np.load(x + '_pose.npy')[1])
        all_data_df["pcd_position_z"] = all_data_df["file"].apply(
            lambda x: np.load(x + '_pose.npy')[2])
        # all_data_df["pcd_position_z"] = all_data_df["file"].apply(
        #     lambda x: np.load(x + '_pose.npy')[2]+2*index*self.cfg.DATA.TRAJ_RADIUS)
        all_data_df["date"] = all_data_df["file"].apply(
            lambda x: x.split('/')[-2])
        all_data_df.reset_index(drop=True, inplace=True)
        return all_data_df

    def get_df(self, dataset_dir, type, is_shuffle=False):
        file_df = pd.DataFrame()
        file_dirs = glob(dataset_dir+"/"+"{}*".format(type))
        file_dirs.sort()
        for index, folder in enumerate(file_dirs):
            data_df = self.get_from_folder(folder, index)
            file_df = file_df.append(data_df, ignore_index=True)
        if is_shuffle:
            file_df.sample(frac=1).reset_index(drop=True)

        return file_df
    
    def construct_query_dict(self, data_df, filename, start_index):
        data_df.reset_index(drop=True, inplace=True)

        tree = KDTree(
            data_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]])
        ind_nn = tree.query_radius(data_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]],
                                   r=self.cfg.DATA.POSITIVES_RADIUS)
        ind_r = tree.query_radius(data_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]],
                                  r=self.cfg.DATA.NEGATIVES_RADIUS)
        ind_traj = tree.query_radius(data_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]],
                                     r=self.cfg.DATA.TRAJ_RADIUS)

        queries = {}
        for i in tqdm(range(len(ind_nn)), total=len(ind_nn), desc='construct queries', leave=False):
            query = data_df.iloc[i]["file"]
            positives = np.setdiff1d(ind_nn[i], [i]).tolist()
            negatives = np.setdiff1d(ind_traj[i], ind_r[i]).tolist()

            random.shuffle(negatives)
            random.shuffle(positives)

            queries[i+start_index] = {"query": query, "positives": positives, "negatives": negatives}
        
        start_index += len(ind_nn)

        with open(os.path.join(filename), 'wb') as handle:
            pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return start_index

    def construct_evaluation_database_query_sets(self, database_df, query_data_df, database_file, query_file):
        if (len(database_df) == 0) or (len(query_data_df) == 0):
            raise ValueError('train data df is empty!')
        database_df.reset_index(drop=True, inplace=True)
        query_data_df.reset_index(drop=True, inplace=True)
        database_tree = KDTree(
            database_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]])
        database_dict = {}
        for index, row in tqdm(database_df.iterrows(), total=len(query_data_df), desc='constructing database sets',
                               leave=False):
            database_dict[index] = {
                "query":            row["file"],
                "pcd_position_x":   row["pcd_position_x"],
                "pcd_position_y":   row["pcd_position_y"],
                "pcd_position_z":   row["pcd_position_z"],
            }
        with open(os.path.join(self.dataset_dir, database_file), 'wb') as handle:
            pickle.dump(database_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        evaluate_query_dict = {}
        for index, row in tqdm(query_data_df.iterrows(), total=len(query_data_df), desc='constructing test sets',
                               leave=False):
            coor = row[["pcd_position_x",
                        "pcd_position_y", "pcd_position_z"]].values
            coor = coor.reshape(1, 3).astype("float64")
            query_index = database_tree.query_radius(
                coor, r=self.cfg.EVALUATE.RADIUS)

            gt_neighbors = query_index[0].tolist()

            evaluate_query_dict[index] = {
                "query":            row["file"],
                "pcd_position_x":   row["pcd_position_x"],
                "pcd_position_y":   row["pcd_position_y"],
                "pcd_position_z":   row["pcd_position_z"],
            }

            evaluate_query_dict[index][0] = gt_neighbors

        with open(os.path.join(self.dataset_dir, query_file), 'wb') as handle:
            pickle.dump(evaluate_query_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_file_func(self, filename):
        # TODO Need to modify here for fusion training
        return self.trans_data.get_data(filename, 
                                        rot_flag=self.rot_flag, 
                                        jitter_flag=self.jitter_flag, 
                                        branch=self.branch)
