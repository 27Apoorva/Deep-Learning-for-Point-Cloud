import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import math
import warnings
import argparse
import math
from math import sin, cos, pi

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

# with open('estimate_poses.txt') as f:
#     poses = np.array([[float(x) for x in line.split()] for line in f])
#     print poses


with open('estimate_poses.txt') as f:
    poses = np.array([[float(x) for x in line.split()] for line in f])

with open('ground_truth.txt') as f:
    ground_truth = np.array([[float(x) for x in line.split()] for line in f])

rospy.init_node('odometry_publisher')

odom_pub = rospy.Publisher("odom", Odometry, queue_size=50)
odom_broadcaster = tf.TransformBroadcaster()
odom = Odometry()
odom_quat = tf.transformations.quaternion_from_euler(0, 0, 0)

ground_pub = rospy.Publisher("ground", Odometry, queue_size=50)
ground_odom = Odometry()
i=0
# x = poses[0][0]
# y = poses[0][1]
# z = poses[0][2]
last_time = rospy.Time.now()
r = rospy.Rate(1.0)
while not rospy.is_shutdown():
    current_time = rospy.Time.now()
    x = poses[i][0]
    y = poses[i][1]
    z = poses[i][2]

    odom_broadcaster.sendTransform(
            (0, 0, 0.),
            (0,0,0,1),
            current_time,
            "base_link",
            "odom"
    )

    # for i in range(poses.shape[0]):
    odom.header.stamp = current_time
    odom.header.frame_id = "odom"
    # import pdb; pdb.set_trace()
    odom.pose.pose = Pose(Point(x, y, z), Quaternion(*odom_quat))
    odom.child_frame_id = "base_link"
    # odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

    ground_odom.header.stamp = current_time
    ground_odom.header.frame_id = "odom"
    # import pdb; pdb.set_trace()
    ground_odom.pose.pose = Pose(Point(ground_truth[i][0], ground_truth[i][1], ground_truth[i][2]), Quaternion(*odom_quat))
    ground_odom.child_frame_id = "base_link"

# publish the message
    odom_pub.publish(odom)
    ground_pub.publish(ground_odom)
    i = i+1
    last_time = current_time
    if(i == 274):
        i = 0
    r.sleep()
