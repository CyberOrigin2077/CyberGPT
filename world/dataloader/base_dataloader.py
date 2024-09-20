'''
Copyright (c) 2024 CyberOrigin
Author: Max
'''

import sys
import os
import copy
import pickle
import random
from abc import abstractmethod
from glob import glob
import torchvision.transforms as transforms
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import numpy as np
from ..utils.log import log_print
from ..utils.geometry.so3_rotate import get_projection_grid, rand_rotation_matrix, rotate_grid, project_2d_on_sphere

random.seed(12345)

class Trans_data(object):
    def __init__(self, cfg):
        self.cfg = cfg
        transforms_A = [
            transforms.Resize((self.cfg.SPHERE.GRID_SIZE[0], self.cfg.SPHERE.GRID_SIZE[1]), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]

        transforms_top = [
            transforms.Resize((self.cfg.SPHERE.TOP_SIZE[0], self.cfg.SPHERE.TOP_SIZE[1]), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
        
        transforms_top_fusion = [
            transforms.RandomRotation(10, resample=False, expand=False, center=None, fill=None),
            transforms.Resize((self.cfg.SPHERE.TOP_SIZE[0], self.cfg.SPHERE.TOP_SIZE[1]), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]

        transforms_B = [
            transforms.Resize((self.cfg.SPHERE.GRID_SIZE[0], self.cfg.SPHERE.GRID_SIZE[1]), Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transform_A = transforms.Compose(transforms_A)
        self.transform_B = transforms.Compose(transforms_B)
        self.transform_top = transforms.Compose(transforms_top)
        self.transform_top_fusion = transforms.Compose(transforms_top_fusion)

        self.grid = get_projection_grid(b=(int)(self.cfg.SPHERE.GRID_SIZE[0]/2.0))
        
    def rotate_point_cloud(self, data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
            Nx3 array, original batch of point clouds
            Return:
            Nx3 array, rotated batch of point clouds
        """
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        if self.cfg.MODEL.NAME in ["PointNetVLAD2", "FusionVLAD3"]:
            rotation_angle = np.pi if np.random.random(1)[0] <=0.5 else 0
            rotation_angle += np.pi/18 * (np.random.uniform()-0.5)
        else:
            rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        rotated_data = np.dot(data, rotation_matrix)
        return rotated_data


    def jitter_point_cloud(self, data, sigma=0.005, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
            Nx3 array, original batch of point clouds
            Return:
            Nx3 array, jittered batch of point clouds
        """
        N, C = data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        jittered_data += data
        return jittered_data

    def pc_normalize(self, pc):
        """Normalize point cloud"""
        centriod = np.mean(pc, axis=0)
        pc = pc - centriod
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def get_data(self, filename, rot_flag=True, jitter_flag=True, is_pair=True, branch=None):
        if self.cfg.DATA.DATASET_NAME != "ISIM":
            if branch == "sph":
                sph_img = Image.open(filename+"_sph.png")
                sph_img = self.transform_A(sph_img)
                if rot_flag:
                    rot = rand_rotation_matrix(deflection=self.cfg.DATA.ROT_DEFLECTION)
                    rotated_grid = rotate_grid(rot, self.grid)
                    out_data = project_2d_on_sphere(sph_img, rotated_grid, projection_origin=[0,0,0.00001])
                else:
                    out_data = sph_img.numpy()
            elif branch == "top":
                top_img = Image.open(filename+"_top.png")
                if rot_flag:
                    out_data = self.transform_top_fusion(top_img).numpy()
                else:
                    out_data = self.transform_top(top_img).numpy()
            elif branch == "points":
                out_data = o3d.io.read_point_cloud(filename + ".pcd")
                out_data = np.asarray(out_data.points)
                if(out_data.shape[0] != 4096):
                    raise ValueError("Input point cloud size should be 4096!")
                if rot_flag:
                    out_data = self.rotate_point_cloud(out_data)
                if jitter_flag:
                    out_data = self.jitter_point_cloud(out_data)
                out_data = self.pc_normalize(out_data)
                
            # out_data = np.zeros([4, self.cfg.SPHERE.GRID_SIZE[0], self.cfg.SPHERE.GRID_SIZE[1]])
            # sph_img = Image.open(filename+"_sph.png")
            # sph_img = self.transform_A(sph_img)
            # if rot_flag:
            #     rot = rand_rotation_matrix(deflection=self.cfg.DATA.ROT_DEFLECTION)
            #     rotated_grid = rotate_grid(rot, self.grid)
            #     sph_img = project_2d_on_sphere(sph_img, rotated_grid, projection_origin=[0,0,0.00001])
            # #* LiDAR Projection
            # out_data[:1,:,:] = sph_img
            # if is_pair and self.cfg.DATA.DATASET_NAME != "PITT":
            #     ori_img = Image.open(filename+"_raw.png")
            #     ori_img = self.transform_B(ori_img)
            #     if rot_flag:
            #         rot = rand_rotation_matrix(deflection=self.cfg.DATA.ROT_DEFLECTION)
            #         rotated_grid = rotate_grid(rot, self.grid)
            #         ori_img = project_2d_on_sphere(ori_img.numpy(), rotated_grid, projection_origin=[0,0,0.00001])
            #     out_data[1:,:,:] = ori_img
        else:
            #! For Dataset IMG, we won't use range projection, and use top-down and map instead
            out_data = np.zeros([6, self.cfg.SPHERE.GRID_SIZE[0], self.cfg.SPHERE.GRID_SIZE[1]])
            map_img = Image.open(filename+"_sph.png")
            raw_img = Image.open(filename+"_raw.png")
            map_img = self.transform_B(map_img)
            raw_img = self.transform_B(raw_img)
            out_data[:3,:,:] = map_img
            out_data[3:,:,:] = raw_img
        return out_data

    def get_topdown_data(self, filename, rot_flag=True, is_pair=True):
        out_data = np.zeros([4, self.cfg.SPHERE.TOP_SIZE[0], self.cfg.SPHERE.TOP_SIZE[1]])
        img = Image.open(filename+"_top.png")
        img = self.transform_top(img)
        # * LiDAR Projection
        out_data[0:1, :, :] = img
        return out_data

        
class BaseDataLoader(object):
    def __init__(self, cfg, is_inference, is_train, is_reference, is_pair=True):
        self.cfg = cfg
        # Generate pickle files
        self.dataset_dir = os.path.join(cfg.DATA.WORKSPACE_DIR, 'data/dataset', str(cfg.DATA.DATASET_NAME))

        self.is_inference = is_inference
        self.is_train = is_train
        self.is_reference = is_reference
        self.data_count = 0
        
        self.trans_data = Trans_data(cfg)
        self.jitter_flag = False
        self.rot_flag = False

        #* Rotation Flag
        if (self.cfg.DATA.ROT_TRAIN and is_train==True) or (self.cfg.DATA.ROT_TEST and is_train==False):
            self.rot_flag = True
        else:
            self.rot_flag = False

        if self.cfg.DATA.TRANSFORM:
            if is_reference or is_train:
                self.data_transform()

        #! Generate Pickles
        self.generate_pickles()
        self.queries = {}
        for pck in self.cfg.DATA.TRAIN_LIST:
            query = self.get_queries_dict("{}/traj_{:02d}.pkl".format(self.dataset_dir, pck))
            self.queries.update(query)
            log_print("Load query {} with {}".format(pck, len(self.queries)), "r")

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        if self.cfg.MODEL.NAME in ["SphVLAD", "SphVLAD2"]:
            self.branch = "sph"
            q_tuple = self.get_data()
        elif self.cfg.MODEL.NAME == "TopVLAD":
            self.branch = "top"
            self.rot_flag = False
            q_tuple = self.get_data()
        elif self.cfg.MODEL.NAME == "FusionVLAD":
            self.branch = "sph"
            q_tuple_sph = self.get_data()
            self.branch = "top"
            q_tuple_top = self.get_data()
            q_tuple = q_tuple_sph + q_tuple_top
        elif self.cfg.MODEL.NAME in ["PointNetVLAD", "PointNetVLAD2"]:
            self.rot_flag = True
            self.jitter_flag = True
            self.branch = "points"
            q_tuple = self.get_data()
        return q_tuple

    def get_data(self):
        q_tuple = []

        while(1):
            if self.data_count > (len(self.file_idxs)-self.cfg.TRAIN.INPUT.BATCH_SIZE):
                self.shuffle_query()
                continue
            batch_keys = self.file_idxs[self.data_count]
            if (len(self.queries[batch_keys]["positives"]) < self.cfg.TRAIN.INPUT.POSITIVES_PER_QUERY) or (len(self.queries[batch_keys]["negatives"]) < self.cfg.TRAIN.INPUT.NEGATIVES_PER_QUERY):
                self.data_count+=1
                continue
            q_tuple = self.get_tuple(self.queries[batch_keys], hard_neg=[], other_neg=True)
            self.data_count+=1
            break

        return q_tuple

    def shuffle_query(self):
        random.shuffle(self.file_idxs)
        self.data_count = 0

    def get_tuple(self, dict_value, hard_neg=None, other_neg=False):
        if hard_neg is None:
            hard_neg = []
        possible_negs = []

        num_pos = self.cfg.TRAIN.INPUT.POSITIVES_PER_QUERY
        num_neg = self.cfg.TRAIN.INPUT.NEGATIVES_PER_QUERY
        random.shuffle(dict_value["positives"])
        pos_files = []
        for i in range(num_pos):
            pos_files.append(self.queries[dict_value["positives"][i]]["query"])

        neg_files = []
        neg_indices = []
        if len(hard_neg) == 0:
            random.shuffle(dict_value["negatives"])
            for i in range(num_neg):
                neg_files.append(self.queries[dict_value["negatives"][i]]["query"])
                neg_indices.append(dict_value["negatives"][i])
        else:
            random.shuffle(dict_value["negatives"])
            for i in hard_neg:
                neg_files.append(self.queries[i]["query"])
                neg_indices.append(i)
            j = 0
            while len(neg_files) < num_neg:
                if not dict_value["negatives"][j] in hard_neg:
                    neg_files.append(self.queries[dict_value["negatives"][j]]["query"])
                    neg_indices.append(dict_value["negatives"][j])
                j += 1
        if other_neg:
            # get neighbors of negatives and query
            neighbors = []
            for pos in dict_value["positives"]:
                neighbors.append(pos)
            for neg in neg_indices:
                for pos in self.queries[neg]["positives"]:
                    neighbors.append(pos)
            possible_negs = list(set(self.queries.keys()) - set(neighbors))
            random.shuffle(possible_negs)

        query = self.load_file_func(dict_value["query"])  # Nx3
        query = np.expand_dims(query, axis=0)
        positives = self.load_files_func(pos_files)
        negatives = self.load_files_func(neg_files)

        if other_neg:
            neg2 = self.load_file_func(self.queries[possible_negs[0]]["query"])
            neg2 = np.expand_dims(neg2, axis=0)
            return [query, positives, negatives, neg2]
        else:
            return [query, positives, negatives]

    def data_transform(self):
        pass

    @abstractmethod
    def load_file_func(self, filename):
        pass

    def load_files_func(self, filenames):
        pcs = []
        for filename in zip(filenames):
            pc = self.load_file_func(filename[0])
            pcs.append(pc)
        pcs = np.array(pcs)
        return pcs

    @abstractmethod
    def generate_pickles(self):
        pass

    @staticmethod
    def get_queries_dict(filename):
        with open(filename, 'rb') as handle:
            queries = pickle.load(handle)
            return queries

    @staticmethod
    def get_sets_dict(filename):
        with open(filename, 'rb') as handle:
            trajectories = pickle.load(handle)
            return trajectories
