# PRISM-TopoMap
The PRISM-TopoMap - online topological mapping method with place recognition and scan matching.

## Paper

The PRISM-TopoMap method is described in the [paper](https://arxiv.org/abs/2404.01674). If you use this code in your research, please cite this paper. A bibtex entry is provided below.

```
@article{muravyev2024prism,
  title={PRISM-TopoMap: Online Topological Mapping with Place Recognition and Scan Matching},
  author={Muravyev, Kirill and Melekhin, Alexander and Yudin, Dmitriy and Yakovlev, Konstantin},
  journal={arXiv preprint arXiv:2404.01674},
  year={2024}
}
```

## Prerequisites:
- [OpenPlaceRecognition](https://github.com/alexmelekhin/openplacerecognition)
- [ROS](https://ros.org) Melodic or Noetic
- For simulated demo: [habitat-sim v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7), [habitat-lab v0.1.7](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7), and [ROS tools for Habitat](https://github.com/cnndepth/habitat_ros)

## Installation
After installing ROS and OpenPlaceRecognition, build PRISM-TopoMap as ROS package:
```
cd your_ros_workspace/src
git clone https://github.com/KirillMouraviev/toposlam_msgs
git clone https://github.com/KirillMouraviev/PRISM-TopoMap
cd ../..
catkin_make
```

After that, download the actual [weights](https://drive.google.com/file/d/1r4Nw0YqHC9PKiZXDmUAWZkOTvgporPnS/view?usp=sharing) for the place recognition model] and set correct path to the weights in the config files `habitat_mp3d.yaml` and `husky_rosbag.yaml`.

## Launch
We provide two examples of launch scripts: for Habitat simulator and for rosbag from Husky robot.

### Habitat simulation

Terminal 1
```
roscore
```

Terminal 2
```
sudo -s
<source ROS and Habitat ros workspace>
roslaunch habitat_ros toposlam_experiment_mp3d_4x90.launch
```

Terminal 3
```
cd your_ros_workspace
source devel/setup.bash
roslaunch prism_topomap build_map_by_iou_habitat.launch path_to_gt_map:=<path to ground truth gt map in .png format> path_to_save_json=<desired path to save the graph in json format> config_file:=habitat_mp3d.yaml
```

### Husky robot

Terminal 1
```
roscore
```

Terminal 2
```
cd <path to rosbags from Husky robot>
for bag in $(ls); do rosbag play $bag --clock; done;
```

Terminal 3
```
cd your_ros_workspace
source devel/setup.bash
roslaunch prism_topomap build_map_by_iou_habitat.launch path_to_gt_map:=<path to ground truth gt map in .png format> path_to_save_json=<desired path to save the graph in json format> config_file:=husky_rosbag_realsense_and_zed.yaml
```

### Parameters in config

**Input**

  `pointcloud`:
  - `topic`: the ROS topic for input point cloud
  - `fields`: point cloud fields (may be `xyz` or `xyzrgb` for colored point clouds)
  - `rotation_matrix`: the matrix of point cloud rotation relative to the standard axes directions (x is forward, y is left, z is up)
  - `floor_height`: the floor level in the input point clouds (relative to the observation point). If the floor on the scene is uneven, the highest floor level should be set. May be set `auto` for dense point clouds.
  - `ceiling_height`: the ceiling level int the input point clouds (relative to the observation point). If the ceiling on the scene is uneven, the lowest ceiling level should be set. May be set `auto` for dense point clouds.

  `odometry`:
  - `topic`: the ROS topic for input odometry
  - `type`: ROS message type in the odometry topic (may be `PoseStamped` or `Odometry`)

  `image_front`:
  - `topic`: the ROS topic for input front-view RGB image
  - `color`: the color order (may be `rgb` or `bgr`)

  `image_back`:
  - `topic`: the ROS topic for input back-view RGB image
  - `color`: the color order (may be `rgb` or `bgr`)

**Topomap**

- `iou_threshold`: overlapping threshold for detachment from the current location
- `localization_frequency`: frequency of the localization module call in seconds

**Place recognition**

- `model`: the place recognition model type (supported types are `mssplace` (multimodal) or `minkloc3d` (point clouds only))
- `weights_path`: path to the place recognition model weights
- `model_config_path`: path to the place recognition model config

**Scan matching**

- `model`: the scan matching model type (supported types are `icp`, `geotransformer`, `feature2d`)
- `detector_type`: the type of feature detector (for `feature2d` model type only). Supported types are `ORB`, `SIFT`, `HarrisWithDistance`
- `score_threshold`: the scans are considered matched if the score is above this value. May be set in range (0, 1), the optimal values are in range (0.5, 0.8)
- `outlier_thresholds`: the thresholds for outlier removal from the matched keypoints (see Algorithm 1 in the paper). The length of the array sets the number of iterations
- `max_point_cloud_range`: the maximum distance from points in the input point cloud to the observation point. All the points further than this distance are removed from the point cloud
- `voxel_downsample_size`: the size of the grid for point cloud voxelization (for `feature2d` model type, the size of the 2D projection grid)
- `min_matches`: the minimal number of the matches for the scans be considered matched

**Scan matching along edge**

- `model`: the scan matching model type (supported types are `icp`, `geotransformer`, `feature2d`)
- `detector_type`: the type of feature detector (for `feature2d` model type only). Supported types are `ORB`, `SIFT`, `HarrisWithDistance`. `HarrisWithDistance` is highly recommended
- `score_threshold`: the scans are considered matched if the score is above this value. May be set in range (0, 1), the optimal values are in range (0.5, 0.8)
- `outlier_thresholds`: the thresholds for outlier removal from the matched keypoints (see Algorithm 1 in the paper). The length of the array sets the number of iterations
- `max_point_cloud_range`: the maximum distance from points in the input point cloud to the observation point. All the points further than this distance are removed from the point cloud
- `voxel_downsample_size`: the size of the grid for point cloud voxelization (for `feature2d` model type, the size of the 2D projection grid)
- `min_matches`: the minimal number of the matches for the scans be considered matched

**Visualization**
- `publish_gt_map`: the flag for the ground truth 2D grid map publication (should be set `true` only if `path_to_gt_map` parameter is correctly set)
- `map_frame`: the ROS frame for the ground truth map (if `publish_gt_map` is set `true`)
- `publish_tf_from_odom`: the flag for the ROS transformation publicatoin into topic `tf` from the input odometry
