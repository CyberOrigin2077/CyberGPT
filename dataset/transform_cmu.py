import os
import numpy as np
import open3d as o3d
from glob import glob
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R

files = sorted(glob("dataset/CMU_*"))

for f in files:
    data_dir = "{}/DATA".format(f)
    os.system('mkdir -p {}'.format(data_dir))
    trjs = sorted(glob('{}/RAW/*'.format(f)))
    for trj_id, trj in enumerate(trjs):
        new_dir = "{}/Train{}".format(data_dir, trj_id+1)
        os.system('mkdir -p {}'.format(new_dir))

        os.system('cp {}/dense.pcd {}/cloudGlobal.pcd'.format(trj, new_dir))

        odom_files = sorted(glob("{}/*.jpg".format(trj)))
        if 'day_forward_1' in trj:
            odom_refes = sorted(glob("{}/*.odom".format(trj)))
        else:
            odom_refes = sorted(glob("{}/*_relative.odom".format(trj)))

        ref_poses = [] 
        for odom_ref in odom_refes:
            if 'day_forward_1' in trj:
                odom = np.loadtxt(odom_ref)
            else:
                odom = np.loadtxt(odom_ref, delimiter=",")
            rot = R.from_matrix(odom[:3,:3])
            rot = rot.as_rotvec()
            trans = odom[:3, 3]
            if 'day_forward_1' in trj:
                times = odom_ref.split('/')[-1].split('.odom')[0].split('_')
            else:
                times = odom_ref.split('/')[-1].split('_relative.odom')[0].split('_')
            time = float("{}.{}".format(times[0], times[1]))
            ref_poses.append([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],time])
        ref_poses = np.array(ref_poses)

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(ref_poses[:,-1].reshape(-1,1))
        poses = []
        gt_poses = []
        pose_txt =  open('{}/poses.txt'.format(new_dir), 'w')
        gt_txt =  open('{}/gts.txt'.format(new_dir), 'w')
        for odom_file in odom_files:
            odom = np.loadtxt('{}odom'.format(odom_file.split('jpg')[0]))
            rot = R.from_matrix(odom[:3,:3])
            rot = rot.as_rotvec()
            trans = odom[:3, 3]
            poses.append(trans)
            times = odom_file.split('/')[-1].split('.jpg')[0].split('_')
            time = float("{}.{}".format(times[0], times[1]))
            pose_txt.write('{} {} {} {} {} {} {}\n'.format(trans[0], trans[1], trans[2], \
                rot[0], rot[1], rot[2], time))

            distances, indices = nbrs.kneighbors(np.array(time).reshape(-1, 1))
            distances = distances.reshape(-1)
            indices = indices.reshape(-1)
            if np.any(distances<0.05):
                index = indices[np.where(distances<0.05)][0]
                gt_pose = ref_poses[index]
            else:
                factor = (ref_poses[indices[1]][-1] - time)/(ref_poses[indices[0]][-1]-time)
                gt_pose = (ref_poses[indices[1]] - factor*ref_poses[indices[0]])/(1-factor)
            gt_pose[-1] = time
            gt_poses.append(gt_pose)
            gt_txt.write('{} {} {} {} {} {} {}\n'.format(gt_pose[0], gt_pose[1], gt_pose[2], \
                gt_pose[3], gt_pose[4], gt_pose[5], time))
                
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(poses))
        o3d.io.write_point_cloud("{}/trj.pcd".format(new_dir), pcd)

        gt_poses = np.array(gt_poses)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_poses[:,:3])
        o3d.io.write_point_cloud("{}/gt.pcd".format(new_dir), gt_pcd)