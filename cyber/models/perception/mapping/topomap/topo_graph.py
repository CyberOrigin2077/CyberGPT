import json
import rospy
import heapq
import numpy as np
import time
import os
import ros_numpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from typing import Dict
from torch import Tensor

import torch
from scipy.spatial.transform import Rotation
from skimage.io import imread, imsave
from opr.datasets.augmentations import DefaultHM3DImageTransform
import MinkowskiEngine as ME
from toposlam_msgs.msg import Edge
from toposlam_msgs.msg import TopologicalGraph as TopologicalGraphMessage


class TopologicalGraph():
    def __init__(self,
                 place_recognition_model,
                 place_recognition_index,
                 registration_model,
                 inline_registration_model,
                 map_frame='map',
                 registration_score_threshold=0.5,
                 inline_registration_score_threshold=0.5,
                 floor_height=-1.0,
                 ceil_height=2.0):
        self.vertices = []
        self.adj_lists = []
        self.map_frame = map_frame
        self.graph_viz_pub = rospy.Publisher('topological_map', MarkerArray, latch=True, queue_size=100)
        self.graph_pub = rospy.Publisher('graph', TopologicalGraphMessage, latch=True, queue_size=100)
        self.place_recognition_model = place_recognition_model
        self.index = place_recognition_index
        self.registration_pipeline = registration_model
        self.registration_score_threshold = registration_score_threshold
        self.inline_registration_pipeline = inline_registration_model
        self.inline_registration_score_threshold = 0.5
        self.ref_cloud_pub = rospy.Publisher('/ref_cloud', PointCloud2, latch=True, queue_size=100)
        self._pointcloud_quantization_size = 0.2
        self.floor_height = floor_height
        self.ceil_height = ceil_height
        self.device = torch.device('cuda:0')
        self.image_transform = DefaultHM3DImageTransform(train=False)
        self.graph_save_path = '/home/kirill/TopoSLAM/toposlam_ws/src/simple_toposlam_model/test_husky_rosbag_minkloc3d_5/graph_data'
        if not os.path.exists(self.graph_save_path):
            os.mkdir(self.graph_save_path)
        self.pr_results_save_path = '/home/kirill/TopoSLAM/toposlam_ws/src/simple_toposlam_model/test_husky_rosbag_minkloc3d_5/place_recognition_data'
        if not os.path.exists(self.pr_results_save_path):
            os.mkdir(self.pr_results_save_path)
        self.global_pose_for_visualization = None

    def _preprocess_input(self, input_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Preprocess input data."""
        out_dict: Dict[str, Tensor] = {}
        for key in input_data:
            if key.startswith("image_"):
                out_dict[f"images_{key[6:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key.startswith("mask_"):
                out_dict[f"masks_{key[5:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key == "pointcloud_lidar_coords":
                quantized_coords, quantized_feats = ME.utils.sparse_quantize(
                    coordinates=input_data["pointcloud_lidar_coords"],
                    features=input_data["pointcloud_lidar_feats"],
                    quantization_size=self._pointcloud_quantization_size,
                )
                out_dict["pointclouds_lidar_coords"] = ME.utils.batched_coordinates([quantized_coords]).to(
                    self.device
                )
                out_dict["pointclouds_lidar_feats"] = quantized_feats.to(self.device)
        return out_dict

    def normalize(self, angle):
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def load_from_json(self, input_path):
        fin = open(os.path.join(input_path, 'graph.json'), 'r')
        j = json.load(fin)
        fin.close()
        self.vertices = j['vertices']
        self.adj_lists = j['edges']
        for i in range(len(self.vertices)):
            cloud = np.load(os.path.join(input_path, '{}_cloud.npz'.format(i)))['arr_0']
            self.vertices[i]['cloud'] = cloud
            img_front = imread(os.path.join(input_path, '{}_img_front.png'.format(i)))
            self.vertices[i]['img_front'] = img_front
            img_back = imread(os.path.join(input_path, '{}_img_back.png'.format(i)))
            self.vertices[i]['img_back'] = img_back
            self.index.add(np.array(self.vertices[i]['descriptor'])[np.newaxis, :])

    def add_vertex(self, global_pose_for_visualization, img_front, img_back, cloud=None):
        x, y, theta = global_pose_for_visualization
        print('Add new vertex ({}, {}, {}) with idx {}'.format(x, y, theta, len(self.vertices)))
        self.adj_lists.append([])
        if cloud is not None:
            img_front_transformed = self.image_transform(img_front)
            # print(img_front_transformed.shape, img_front_transformed.min(), img_front_transformed.mean(), img_front_transformed.max())
            img_back_transformed = self.image_transform(img_back)
            # print(img_back_transformed.shape, img_back_transformed.min(), img_back_transformed.mean(), img_back_transformed.max())
            # print('Transformed image shape:', img_front_transformed.shape)
            img_front_tensor = torch.Tensor(img_front).cuda()
            img_back_tensor = torch.Tensor(img_back).cuda()
            # print('Img front min and max:', img_front_transformed.min(), img_front_transformed.max())
            img_front_tensor = torch.permute(img_front_tensor, (2, 0, 1))
            img_back_tensor = torch.permute(img_back_tensor, (2, 0, 1))
            input_data = {
                     'pointcloud_lidar_coords': torch.Tensor(cloud[:, :3]).cuda(),
                     'pointcloud_lidar_feats': torch.ones((cloud.shape[0], 1)).cuda(),
                     'image_front': img_front_tensor,
                     'image_back': img_back_tensor
                     }
            batch = self._preprocess_input(input_data)
            descriptor = self.place_recognition_model(batch)["final_descriptor"].detach().cpu().numpy()
            if len(descriptor.shape) == 1:
                descriptor = descriptor[np.newaxis, :]
            # print('X y theta:', x, y, theta)
            vertex_dict = {
                'stamp': rospy.Time.now(),
                'pose_for_visualization': [x, y, theta],
                'img_front': img_front,
                'img_back': img_back,
                'cloud': cloud,
                'descriptor': descriptor
            }
            self.vertices.append(vertex_dict)
            # print('Descriptor shape:', descriptor.shape)
            self.index.add(descriptor)
        return len(self.vertices) - 1

    def save_vertex(self, vertex_id):
        save_dir = os.path.join(self.graph_save_path, str(vertex_id))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        vertex_dict = self.vertices[vertex_id]
        pose_stamped = np.array([vertex_dict['stamp'].to_sec()] + vertex_dict['pose_for_visualization'])
        # print('Pose stamped:', pose_stamped)
        np.savetxt(os.path.join(save_dir, 'pose_stamped.txt'), pose_stamped)
        imsave(os.path.join(save_dir, 'img_front.png'), vertex_dict['img_front'])
        imsave(os.path.join(save_dir, 'img_back.png'), vertex_dict['img_back'])
        np.savez(os.path.join(save_dir, 'cloud.npz'), vertex_dict['cloud'])
        np.savetxt(os.path.join(save_dir, 'descriptor.txt'), vertex_dict['descriptor'])
        edges = []
        for v, rel_pose in self.adj_lists[vertex_id]:
            edges.append([v] + rel_pose)
        edges = np.array(edges)
        np.savetxt(os.path.join(save_dir, 'edges.txt'), edges)

    def save_localization_results(self, state_dict, vertex_ids, transforms, pr_scores, reg_scores):
        save_dir = os.path.join(self.pr_results_save_path, str(state_dict['stamp']))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        imsave(os.path.join(save_dir, 'img_front.png'), state_dict['img_front'])
        imsave(os.path.join(save_dir, 'img_back.png'), state_dict['img_back'])
        np.savez(os.path.join(save_dir, 'cloud.npz'), state_dict['cloud'])
        np.savetxt(os.path.join(save_dir, 'descriptor.txt'), state_dict['descriptor'])
        np.savetxt(os.path.join(save_dir, 'vertex_ids.txt'), vertex_ids)
        gt_pose_data = [state_dict['pose_for_visualization']]
        tf_data = []
        for idx, tf in zip(vertex_ids, transforms, strict=False):
            if idx >= 0:
                vertex_dict = self.vertices[idx]
                # print('GT x, y, theta:', x, y, theta)
                # np.savetxt(os.path.join(save_dir, 'cand_cloud_{}.txt'.format(idx)), cloud)
                if tf is not None:
                    tf_data.append([idx] + list(tf))
                else:
                    tf_data.append([idx, 0, 0, 0, 0, 0, 0])
                gt_pose_data.append(vertex_dict['pose_for_visualization'])
        np.savetxt(os.path.join(save_dir, 'gt_poses.txt'), np.array(gt_pose_data))
        np.savetxt(os.path.join(save_dir, 'transforms.txt'), np.array(tf_data))
        np.savetxt(os.path.join(save_dir, 'pr_scores.txt'), np.array(pr_scores))
        np.savetxt(os.path.join(save_dir, 'reg_scores.txt'), np.array(reg_scores))

    def get_k_most_similar(self, img_front, img_back, cloud, stamp, k=1):
        t1 = time.time()
        img_front_tensor = torch.Tensor(img_front).cuda()
        img_back_tensor = torch.Tensor(img_back).cuda()
        img_front_tensor = torch.permute(img_front_tensor, (2, 0, 1))
        img_back_tensor = torch.permute(img_back_tensor, (2, 0, 1))
        input_data = {'pointcloud_lidar_coords': torch.Tensor(cloud[:, :3]).cuda(),
                    'pointcloud_lidar_feats': torch.ones((cloud.shape[0], 1)).cuda(),
                    'image_front': img_front_tensor,
                    'image_back': img_back_tensor}
        if cloud is not None:
            cloud_with_fields = np.zeros((cloud.shape[0]), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),  # ])
                ('r', np.uint8),
                ('g', np.uint8),
                ('b', np.uint8)])
            cloud_with_fields['x'] = cloud[:, 0]
            cloud_with_fields['y'] = cloud[:, 1]
            cloud_with_fields['z'] = cloud[:, 2]
            # cloud_with_fields['r'] = cloud[:, 3]
            # cloud_with_fields['g'] = cloud[:, 4]
            # cloud_with_fields['b'] = cloud[:, 5]
            # cloud_with_fields = ros_numpy.point_cloud2.merge_rgb_fields(cloud_with_fields)
            cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_with_fields)
            if stamp is not None:
                cloud_msg.header.stamp = stamp
            else:
                cloud_msg.header.stamp = rospy.Time.now()
            cloud_msg.header.frame_id = 'base_link'
            self.ref_cloud_pub.publish(cloud_msg)
        t2 = time.time()
        # print('Ref cloud publish time:', t2 - t1)
        batch = self._preprocess_input(input_data)
        t3 = time.time()
        # print('Preprocessing time:', t3 - t2)
        descriptor = self.place_recognition_model(batch)["final_descriptor"].detach().cpu().numpy()
        if len(descriptor.shape) == 1:
            descriptor = descriptor[np.newaxis, :]
        reg_scores = []
        dists, pred_i = self.index.search(descriptor, k)
        t4 = time.time()
        # print('Place recognition time:', t4 - t3)
        pr_scores = dists[0]
        pred_i = pred_i[0]
        pred_tf = []
        pred_i_filtered = []
        for idx in pred_i:
            # print('Stamp {}, vertex id {}'.format(stamp, idx))
            if idx < 0:
                continue
            t1 = time.time()
            cand_vertex_dict = self.vertices[idx]
            cand_cloud = cand_vertex_dict['cloud']
            cand_cloud_tensor = torch.Tensor(cand_cloud[:, :3]).to(self.device)
            ref_cloud_tensor = torch.Tensor(cloud[:, :3]).to(self.device)
            start_time = time.time()
            save_dir = os.path.join(self.pr_results_save_path, str(stamp))
            tf_matrix, score = self.registration_pipeline.infer(ref_cloud_tensor, cand_cloud_tensor, save_dir=save_dir)
            t2 = time.time()
            # print('Registration time:', t2 - t1)
            # t3 = time.time()
            # print('ICP time:', t3 - t2)
            # if score_icp < 0.8:
            reg_scores.append(score)
            # print('Registration score of vertex {} is {}'.format(idx, score))
            if score < self.registration_score_threshold:
                pred_i_filtered.append(-1)
                pred_tf.append([0, 0, 0, 0, 0, 0])
            else:
                pred_i_filtered.append(idx)
                tf_rotation = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
                tf_translation = tf_matrix[:3, 3]
                pred_tf.append(list(tf_rotation) + list(tf_translation))
                # print('Tf rotation:', tf_rotation)
                # print('Tf translation:', tf_translation)
        # print('Pred tf:', np.array(pred_tf))
        # print('Pred i filtered:', pred_i_filtered)
        state_dict = {
            'stamp': stamp,
            'pose_for_visualization': self.global_pose_for_visualization,
            'img_front': img_front,
            'img_back': img_back,
            'cloud': cloud,
            'descriptor': descriptor
        }
        self.save_localization_results(state_dict, pred_i, pred_tf, pr_scores, reg_scores)
        return pred_i, pred_i_filtered, np.array(pred_tf), pr_scores, reg_scores

    def get_transform_to_vertex(self, vertex_id, cloud):
        cand_cloud = self.vertices[vertex_id]['cloud']
        cand_cloud_tensor = torch.Tensor(cand_cloud[:, :3]).to(self.device)
        ref_cloud_tensor = torch.Tensor(cloud[:, :3]).to(self.device)
        tf_matrix, score = self.inline_registration_pipeline.infer(ref_cloud_tensor, cand_cloud_tensor)
        if score > self.inline_registration_score_threshold:
            tf_rotation = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
            tf_translation = tf_matrix[:3, 3]
            x, y, _ = tf_translation
            _, __, theta = tf_rotation
            return x, y, theta
        return None, None, None

    def inverse_transform(self, x, y, theta):
        x_inv = -x * np.cos(theta) - y * np.sin(theta)
        y_inv = x * np.sin(theta) - y * np.cos(theta)
        theta_inv = -theta
        return [x_inv, y_inv, theta_inv]

    def add_edge(self, i, j, x, y, theta):
        if i == j:
            return
        if j in [x[0] for x in self.adj_lists[i]]:
            return
        xi, yi, _ = self.vertices[i]['pose_for_visualization']
        xj, yj, _ = self.vertices[j]['pose_for_visualization']
        print('Add edge from ({}, {}) to ({}, {})'.format(xi, yi, xj, yj))
        self.adj_lists[i].append((j, [x, y, theta]))
        self.adj_lists[j].append((i, self.inverse_transform(x, y, theta)))

    def get_vertex(self, vertex_id):
        return self.vertices[vertex_id]

    def has_edge(self, u, v):
        for x, rel_pose in self.adj_lists[u]:
            if x == v:
                return True
        return False

    def get_edge(self, u, v):
        for x, rel_pose in self.adj_lists[u]:
            if x == v:
                return rel_pose
        return None

    def get_edges_from(self, u):
        return self.adj_lists[u]

    def get_path_with_length(self, u, v):
        # Initialize distances and previous nodes dictionaries
        distances = [float('inf')] * len(self.adj_lists)
        prev_nodes = [None] * len(self.adj_lists)
        # Set distance to start node as 0
        distances[u] = 0
        # Create priority queue with initial element (distance to start node, start node)
        heap = [(0, u)]
        # Run Dijkstra's algorithm
        while heap:
            # Pop node with lowest distance from heap
            current_distance, current_node = heapq.heappop(heap)
            if current_node == v:
                path = [current_node]
                cur = current_node
                while cur != u:
                    cur = prev_nodes[cur]
                    path.append(cur)
                path = path[::-1]
                return path, distances[v]
            # If current node has already been visited, skip it
            if current_distance > distances[current_node]:
                continue
            # For each neighbour of current node
            for neighbour, pose in self.adj_lists[current_node]:
                weight = np.sqrt(pose[0] ** 2 + pose[1] ** 2)
                # Calculate tentative distance to neighbour through current node
                tentative_distance = current_distance + weight
                # Update distance and previous node if tentative distance is better than current distance
                if tentative_distance < distances[neighbour]:
                    distances[neighbour] = tentative_distance
                    prev_nodes[neighbour] = current_node
                    # Add neighbour to heap with updated distance
                    heapq.heappush(heap, (tentative_distance, neighbour))
        return None, float('inf')

    def publish_graph(self):
        # Publish graph for visualization
        graph_msg = MarkerArray()
        vertices_marker = Marker()
        # vertices_marker = ns = 'points_and_lines'
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = rospy.Time.now()
        vertices_marker.scale.x = 0.2
        vertices_marker.scale.y = 0.2
        vertices_marker.scale.z = 0.2
        vertices_marker.color.r = 1
        vertices_marker.color.g = 0
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        for vertex_dict in self.vertices:
            x, y, _ = vertex_dict['pose_for_visualization']
            vertices_marker.points.append(Point(x, y, 0.05))
        graph_msg.markers.append(vertices_marker)

        edges_marker = Marker()
        edges_marker.id = 1
        edges_marker.type = Marker.LINE_LIST
        edges_marker.header.frame_id = self.map_frame
        edges_marker.header.stamp = rospy.Time.now()
        edges_marker.scale.x = 0.1
        edges_marker.color.r = 0
        edges_marker.color.g = 0
        edges_marker.color.b = 1
        edges_marker.color.a = 0.5
        edges_marker.pose.orientation.w = 1
        for u in range(len(self.vertices)):
            for v, pose in self.adj_lists[u]:
                ux, uy, _ = self.vertices[u]['pose_for_visualization']
                vx, vy, _ = self.vertices[v]['pose_for_visualization']
                edges_marker.points.append(Point(ux, uy, 0.05))
                edges_marker.points.append(Point(vx, vy, 0.05))
        graph_msg.markers.append(edges_marker)
        self.graph_viz_pub.publish(graph_msg)

        # publish graph for navigation
        graph_msg = TopologicalGraphMessage()
        edges = []
        graph_msg.n_vertices = len(self.vertices)
        for u in range(len(self.vertices)):
            for v, rel_pose in self.adj_lists[u]:
                edge = Edge()
                edge.vertex_from = u
                edge.vertex_to = v
                edge.rel_pose = Point(rel_pose[0], rel_pose[1], 0)
                edges.append(edge)

    def save_to_json(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.vertices = list(self.vertices)
        for i in range(len(self.vertices)):
            vertex_dict = self.vertices[i]
            np.savez(os.path.join(output_path, '{}_cloud.npz'.format(i)), vertex_dict['cloud'])
            imsave(os.path.join(output_path, '{}_img_front.png'.format(i)), vertex_dict['img_front'])
            imsave(os.path.join(output_path, '{}_img_back.png'.format(i)), vertex_dict['img_back'])
            x, y, theta = vertex_dict['pose_for_visualization']
            descriptor = vertex_dict['descriptor']
            self.vertices[i] = {'pose_for_visualization': (x, y, theta), 'descriptor': [float(x) for x in list(descriptor[0])]}
        j = {'vertices': self.vertices, 'edges': self.adj_lists}
        fout = open(os.path.join(output_path, 'graph.json'), 'w')
        json.dump(j, fout)
        fout.close()
