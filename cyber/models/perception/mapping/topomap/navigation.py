#! /usr/bin/env python
import rospy
import numpy as np
np.float = np.float64
import tf
import time
import actionlib
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from geometry_msgs.msg import PoseStamped
from toposlam_msgs.msg import TopoSLAMNavigationAction, TopoSLAMNavigationActionGoal, TopoSLAMNavigationActionFeedback, TopoSLAMNavigationActionResult
from toposlam_msgs.msg import TopologicalPath
from utils import *

current_pose = []
path = []
DEFAULT_TOLERANCE = 1
DEFAULT_RATE = 10
DEFAULT_TIMEOUT = 30
DEFAULT_N_FAILS = 5


class TopoSLAMNavigationServer:
    def __init__(self):
        rospy.init_node('toposlam_navigation')
        self.goal = TopoSLAMNavigationActionGoal()
        self.feedback = TopoSLAMNavigationActionFeedback()
        self.result = TopoSLAMNavigationActionResult()
        tolerance = rospy.get_param('~tolerance', DEFAULT_TOLERANCE)
        rate = rospy.get_param('~rate', DEFAULT_RATE)
        timeout = rospy.get_param('~timeout', DEFAULT_TIMEOUT)
        print('TOLERANCE IS', tolerance)
        self.tolerance = tolerance
        self.rate = rospy.Rate(rate)
        self.timeout = timeout
        self.n_vertices = 0
        self.adj_lists = []
        self.tf_listener = tf.TransformListener()
        # self.graph_subscriber = rospy.Subscriber('/graph', TopologicalGraph, self.graph_callback)
        self.last_vertex_id_subscriber = rospy.Subscriber('/last_vertex_id', Int32, self.last_vertex_id_callback)
        self.rel_pose_of_vcur_subscriber = rospy.Subscriber('/rel_pose_of_vcur', PoseStamped, self.rel_pose_of_vcur_callback)
        self.topological_path_subscriber = rospy.Subscriber('/topological_path', TopologicalPath, self.topological_path_callback)
        self.server = actionlib.SimpleActionServer('move_to_point', TopoSLAMNavigationAction, self.execute, False)
        self.target_node_publisher = rospy.Publisher('/target_node', Int32, latch=True, queue_size=100)
        self.local_planning_task_publisher = rospy.Publisher('/local_planning_task', Float32MultiArray, latch=True, queue_size=100)
        self.rel_x, self.rel_y, self.rel_theta = None, None, None
        self.last_vertex_id = None
        self.topological_path = []
        self.rel_poses = []
        self.local_path = []
        self.server.start()

    # def graph_callback(self, msg):
    #    self.n_vertices = msg.n_vertices
    #    self.adj_lists = [[] for _ in range(self.n_vertices)]
    #    for e in msg.edges:
    #        self.adj_lists[msg.vertex_from].append((msg.vertex_to, msg.rel_pose))
    #        self.adj_lists[msg.vertex_to].append((msg.vertex_from, msg.rel_pose))

    def last_vertex_id_callback(self, msg):
        self.last_vertex_id = msg.data

    def rel_pose_of_vcur_callback(self, msg):
        self.rel_x, self.rel_y = msg.pose.position.x, msg.pose.position.y
        _, __, self.rel_theta = tf.transformations.euler_from_quaternion([msg.pose.orientation.w,
                                                                          msg.pose.orientation.x,
                                                                          msg.pose.orientation.y,
                                                                          msg.pose.orientation.z])

    def topological_path_callback(self, msg):
        print('Topological path received')
        self.topological_path = list(msg.nodes)
        if len(self.topological_path) == 0:
            print('EMPTY PATH IN TOPOLOGICAL GRAPH!!!')
            self.rel_poses = []
        else:
            assert self.topological_path[-1] == self.target_vertex_id
            self.rel_poses = [(pt.x, pt.y, pt.z) for pt in msg.rel_poses]

    def local_path_callback(self, msg):
        print('Local path received')
        self.local_path = [[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses]

    def publish_topological_planning_task(self):
        target_node_msg = Int32()
        target_node_msg.data = self.target_vertex_id
        self.target_node_publisher.publish(target_node_msg)
        print('Topological task published')

    def publish_local_planning_task(self):
        if self.rel_x is None:
            print('Waiting for rel pose from topoSLAM...')
            return
        if self.target_vertex_id == self.last_vertex_id:
            target_x, target_y = self.target_rel_x, self.target_rel_y
        else:
            if len(self.rel_poses) == 0:
                return
            target_x, target_y, target_theta = self.rel_poses[0]
        task_msg = Float32MultiArray()
        task_msg.layout.dim.append(MultiArrayDimension())
        task_msg.layout.dim[0].size = 4
        task_msg.layout.dim[0].stride = 4
        task_msg.data = [self.rel_x, self.rel_y, target_x, target_y]
        self.local_planning_task_publisher.publish(task_msg)
        print('Local planning task published')

    # def control_status_callback(self, msg):
    #    if msg.data == 'reached':
    #        self.goal_reached = True
    def goal_reached(self):
        if self.last_vertex_id is None:
            print('Waiting for topoSLAM callback...')
            return False
        if self.target_vertex_id == self.last_vertex_id:
            print('We are in the target location!')
            d = np.sqrt((self.rel_x - self.target_rel_x) ** 2 + (self.rel_y - self.target_rel_y) ** 2)
            print('Distance to goal:', d)
            return (d < self.tolerance)
        if len(self.topological_path) == 2:
            print('The next location is the target!')
            rel_pose_vcur_to_goal = apply_pose_shift(self.rel_poses[0], self.target_rel_x, self.target_rel_y, 0)
            d = np.sqrt((self.rel_x - rel_pose_vcur_to_goal[0]) ** 2 + (self.rel_y - rel_pose_vcur_to_goal[1]) ** 2)
            print('Distance to goal:', d)
            return (d < self.tolerance)
        return False

    def wait_until_come(self):
        start_time = time.time()
        n_path_fails = 0
        succeeded = False
        while time.time() - start_time < self.timeout and not rospy.is_shutdown():
            self.publish_topological_planning_task()
            self.publish_local_planning_task()
            if self.goal_reached():
                print('Goal reached!')
                succeeded = True
                break
            self.feedback.feedback.rel_x = self.rel_x
            self.feedback.feedback.rel_y = self.rel_y
            self.feedback.feedback.vertex_id = self.last_vertex_id
            self.rate.sleep()
        if time.time() - start_time > self.timeout and not succeeded:
            print('Goal timed out!')
        return succeeded

    def execute(self, goal):
        self.target_vertex_id = goal.vertex_id
        self.target_rel_x = goal.rel_x
        self.target_rel_y = goal.rel_y
        success = self.wait_until_come()
        self.result.result.rel_x = self.feedback.feedback.rel_x
        self.result.result.rel_y = self.feedback.feedback.rel_y
        if success:
            self.server.set_succeeded(self.result.result)
        else:
            self.server.set_preempted()


if __name__ == '__main__':
    server = TopoSLAMNavigationServer()
    rospy.spin()
