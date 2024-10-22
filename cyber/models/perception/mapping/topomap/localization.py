import rospy
import os
import numpy as np
np.float = np.float64
import ros_numpy
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Bool
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import time
from utils import apply_pose_shift

tests_dir = '/home/kirill/TopoSLAM/OpenPlaceRecognition/test_registration'


class Localizer():
    def __init__(self, graph, gt_map, map_frame='map', publish=True, top_k=5):
        self.graph = graph
        self.gt_map = gt_map
        self.top_k = top_k
        self.img_front = None
        self.img_back = None
        self.cloud = None
        self.global_pose_for_visualization = None
        self.stamp = None
        self.localized_x = None
        self.localized_y = None
        self.localized_theta = None
        self.localized_stamp = None
        self.localized_img_front = None
        self.localized_img_back = None
        self.localized_cloud = None
        self.rel_poses = None
        self.dists = None
        self.cnt = 0
        self.n_loc_fails = 0
        if not os.path.exists(tests_dir):
            os.mkdir(tests_dir)
        self.publish = publish
        self.map_frame = map_frame
        self.result_publisher = rospy.Publisher('/localized_nodes', Float32MultiArray, latch=True, queue_size=100)
        self.cand_cloud_publisher = rospy.Publisher('/candidate_cloud', PointCloud2, latch=True, queue_size=100)
        self.matched_points_publisher = rospy.Publisher('/matched_points', Marker, latch=True, queue_size=100)
        self.unmatched_points_publisher = rospy.Publisher('/unmatched_points', Marker, latch=True, queue_size=100)
        self.transforms_publisher = rospy.Publisher('/localization_transforms', Marker, latch=True, queue_size=100)
        self.first_pr_publisher = rospy.Publisher('/first_point', Marker, latch=True, queue_size=100)
        self.first_pr_image_publisher = rospy.Publisher('/place_recognition/image', Image, latch=True, queue_size=100)
        self.freeze_publisher = rospy.Publisher('/freeze', Bool, latch=True, queue_size=100)
        self.bridge = CvBridge()

    def save_reg_test_data(self, vertex_ids, transforms, pr_scores, reg_scores, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.savez(os.path.join(save_dir, 'ref_cloud.npz'), self.localized_cloud)
        # print('Mean of the ref cloud:', self.localized_cloud[:, :3].mean())
        tf_data = []
        gt_pose_data = [[self.localized_x, self.localized_y, self.localized_theta]]
        for idx, tf in zip(vertex_ids, transforms, strict=False):
            if idx >= 0:
                vertex_dict = self.graph.vertices[idx]
                x, y, theta = vertex_dict['pose_for_visualization']
                cloud = vertex_dict['cloud']
                # print('GT x, y, theta:', x, y, theta)
                # np.savetxt(os.path.join(save_dir, 'cand_cloud_{}.txt'.format(idx)), cloud)
                if tf is not None:
                    tf_data.append([idx] + list(tf))
                else:
                    tf_data.append([idx, 0, 0, 0, 0, 0, 0])
                gt_pose_data.append([x, y, theta])
        # print('TF data:', tf_data)
        np.savetxt(os.path.join(save_dir, 'gt_poses.txt'), np.array(gt_pose_data))
        np.savetxt(os.path.join(save_dir, 'transforms.txt'), np.array(tf_data))
        np.savetxt(os.path.join(save_dir, 'pr_scores.txt'), np.array(pr_scores))
        np.savetxt(os.path.join(save_dir, 'reg_scores.txt'), np.array(reg_scores))

    def publish_localization_results(self, vertex_id_first, vertex_ids_matched, vertex_ids_unmatched, rel_poses):
        # Publish top-1 PlaceRecognition vertex
        vertices_marker = Marker()
        # vertices_marker = ns = 'points_and_lines'
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = rospy.Time.now()
        vertices_marker.scale.x = 0.4
        vertices_marker.scale.y = 0.4
        vertices_marker.scale.z = 0.4
        if vertex_id_first in vertex_ids_matched:
            vertices_marker.color.r = 0
        else:
            vertices_marker.color.r = 1
        vertices_marker.color.g = 1
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        x, y, _ = self.graph.vertices[vertex_id_first]['pose_for_visualization']
        img_front = self.graph.vertices[vertex_id_first]['img_front']
        vertices_marker.points.append(Point(x, y, 0.1))
        self.first_pr_publisher.publish(vertices_marker)

        # Publish top-1 PlaceRecognition image
        img_msg = self.bridge.cv2_to_imgmsg(img_front)
        img_msg.encoding = 'rgb8'
        img_msg.header.stamp = rospy.Time.now()
        self.first_pr_image_publisher.publish(img_msg)

        # Publish matched vertices
        vertices_marker = Marker()
        # vertices_marker = ns = 'points_and_lines'
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = rospy.Time.now()
        vertices_marker.scale.x = 0.2
        vertices_marker.scale.y = 0.2
        vertices_marker.scale.z = 0.2
        vertices_marker.color.r = 0
        vertices_marker.color.g = 1
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        localized_vertices = [self.graph.vertices[i] for i in vertex_ids_matched]
        for vertex_dict in localized_vertices:
            x, y, _ = vertex_dict['pose_for_visualization']
            vertices_marker.points.append(Point(x, y, 0.1))
        self.matched_points_publisher.publish(vertices_marker)

        transforms_marker = Marker()
        transforms_marker.type = Marker.LINE_LIST
        transforms_marker.header.frame_id = self.map_frame
        transforms_marker.header.stamp = rospy.Time.now()
        transforms_marker.scale.x = 0.1
        transforms_marker.color.r = 1
        transforms_marker.color.g = 0
        transforms_marker.color.b = 0
        transforms_marker.color.a = 0.5
        transforms_marker.pose.orientation.w = 1
        for vertex_dict, rel_pose in zip(localized_vertices, rel_poses, strict=False):
            x, y, theta = vertex_dict['pose_for_visualization']
            transforms_marker.points.append(Point(x, y, 0.1))
            x, y, theta = apply_pose_shift([x, y, theta], *rel_pose)
            transforms_marker.points.append(Point(x, y, 0.1))
        self.transforms_publisher.publish(transforms_marker)

        # Publish unmatched vertices
        vertices_marker.id = 0
        vertices_marker.header.frame_id = self.map_frame
        vertices_marker.header.stamp = rospy.Time.now()
        vertices_marker.scale.x = 0.2
        vertices_marker.scale.y = 0.2
        vertices_marker.scale.z = 0.2
        vertices_marker.color.r = 1
        vertices_marker.color.g = 1
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        localized_vertices = [self.graph.vertices[i] for i in vertex_ids_unmatched]
        for vertex_dict in localized_vertices:
            x, y, _ = vertex_dict['pose_for_visualization']
            vertices_marker.points.append(Point(x, y, 0.1))
        self.unmatched_points_publisher.publish(vertices_marker)

    def publish_result(self, vertex_ids, rel_poses):
        if rel_poses.size == 0:
            return
        dists = np.sqrt(rel_poses[:, 0] ** 2 + rel_poses[:, 1] ** 2)
        ids = [i for i in range(len(vertex_ids))]
        ids.sort(key=lambda i: dists[i])
        vertex_ids = [vertex_ids[i] for i in ids]
        rel_poses = [rel_poses[i] for i in ids]
        result_msg = Float32MultiArray()
        result_msg.layout.dim.append(MultiArrayDimension())
        result_msg.layout.data_offset = 0
        n = len(vertex_ids)
        result_msg.layout.dim[0].size = n * 4
        result_msg.layout.dim[0].stride = n * 4
        for i in range(n):
            result_msg.data.append(vertex_ids[i])
        for i in range(n):
            result_msg.data.append(rel_poses[i][0])
        for i in range(n):
            result_msg.data.append(rel_poses[i][1])
        for i in range(n):
            result_msg.data.append(rel_poses[i][2])
        self.result_publisher.publish(result_msg)
        if len(vertex_ids) > 0:
            i = vertex_ids[0]
            cloud = self.graph.vertices[i]['cloud']
            cloud_with_fields = np.zeros((cloud.shape[0]), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),  # ]),
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
            if self.stamp is None:
                cloud_msg.header.stamp = rospy.Time.now()
            else:
                cloud_msg.header.stamp = self.stamp
            cloud_msg.header.frame_id = 'base_link'
            self.cand_cloud_publisher.publish(cloud_msg)

    def localize(self, event=None):
        if self.global_pose_for_visualization is None:
            print('No global pose provided!')
            return
        dt = (rospy.Time.now() - self.stamp).to_sec()
        # print('Localization lag:', dt)
        vertex_ids = []
        rel_poses = []
        freeze_msg = Bool()
        freeze_msg.data = True
        self.freeze_publisher.publish(freeze_msg)
        t1 = time.time()
        start_global_pose = self.global_pose_for_visualization
        start_stamp = self.stamp
        start_img_front = self.img_front
        start_img_back = self.img_back
        start_cloud = self.cloud
        # print('Position at start:', self.global_pose_for_visualization)
        self.graph.global_pose_for_visualization = self.global_pose_for_visualization
        if self.cloud is not None:
            vertex_ids_pr_raw, vertex_ids_pr, transforms, pr_scores, reg_scores = self.graph.get_k_most_similar(self.img_front,
                                                                                                                self.img_back,
                                                                                                                self.cloud,
                                                                                                                self.stamp,
                                                                                                                k=self.top_k)
            t2 = time.time()
            # print('Get k most similar time:', t2 - t1)
            # save_dir = os.path.join(tests_dir, 'test_{}'.format(self.cnt))
            # self.cnt += 1
            # if not os.path.exists(save_dir):
            #    os.mkdir(save_dir)
            # self.save_reg_test_data(vertex_ids_pr_raw, transforms, pr_scores, reg_scores, save_dir)
            t3 = time.time()
            # print('Saving time:', t3 - t2)
            vertex_ids_pr_unmatched = [idx for idx in vertex_ids_pr_raw if idx not in vertex_ids_pr]
            # print('Matched indices:', [idx for idx in vertex_ids_pr if idx >= 0])
            # print('Unmatched indices:', vertex_ids_pr_unmatched)
            rel_poses = [[tf[3], tf[4], tf[2]] for idx, tf in zip(vertex_ids_pr, transforms, strict=False) if idx >= 0]
            self.publish_localization_results(vertex_ids_pr_raw[0], [idx for idx in vertex_ids_pr if idx >= 0], vertex_ids_pr_unmatched, rel_poses)
            t4 = time.time()
            # print('Publish time:', t4 - t3)
        else:
            vertex_ids_pr = []
        t4 = time.time()
        pr_scores = [pr_scores[i] for i, idx in enumerate(vertex_ids_pr) if idx >= 0]
        reg_scores = [reg_scores[i] for i, idx in enumerate(vertex_ids_pr) if idx >= 0]
        transforms = [transforms[i] for i, idx in enumerate(vertex_ids_pr) if idx >= 0]
        transforms = np.array(transforms)
        vertex_ids_pr = [i for i in vertex_ids_pr if i >= 0]
        if len(vertex_ids_pr) == 0:
            self.n_loc_fails += 1
        # for i, v in enumerate(self.graph.vertices):
        for i, idx in enumerate(vertex_ids_pr):
            v = self.graph.vertices[idx]
            # if self.gt_map.in_sight(x, y, v[0], v[1]):
            vertex_ids.append(idx)
            # print('Transform:', transforms[i])
            rel_poses.append(transforms[i, [3, 4, 2]])
            # print('True dist:', dist)
            # print('Descriptor dist:', pr_scores[i])
            # print('Reg score:', reg_scores[i])
        rel_poses = np.array(rel_poses)
        freeze_msg = Bool()
        freeze_msg.data = False
        self.freeze_publisher.publish(freeze_msg)
        t5 = time.time()
        # print('Cloud publish time:', t5 - t4)
        # print('Localization time:', t5 - t1)
        if len(vertex_ids) > 0:
            self.localized_x, self.localized_y, self.localized_theta = start_global_pose
            self.localized_stamp = start_stamp
            self.localized_img_front = start_img_front
            self.localized_img_back = start_img_back
            self.localized_cloud = start_cloud
        if self.publish:
            self.publish_result(vertex_ids, rel_poses)
        return vertex_ids, rel_poses
