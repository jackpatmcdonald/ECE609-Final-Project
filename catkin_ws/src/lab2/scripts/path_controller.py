#!/usr/bin/env python3

import rospy
import math
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class PathController:
    # setup
    def __init__(self):
        rospy.init_node('path_controller')
        # initial pose
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        # Robot parameters - change as needed
        self.linear_speed = 0.15
        self.angular_speed = 0.40
        self.goal_tolerance = 0.10
        self.angle_tolerance = 0.05

        self.path = []
        self.running = False
        
        # Publisher
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # Subscribers
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/planned_path', Path, self.path_cb)

        rospy.spin()

    #------------------------callback functions---------------------
    # odometry
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # get the path from astar_planner
    def path_cb(self, msg):
        # get path
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        # follow the yellow brick road
        self.follow_path()


    #----------MAth functions--------------
    # get turn angle
    def angle_to(self, tx, ty):
        # atan2 because 4 quadrants
        return math.atan2(ty - self.y, tx - self.x)

    # pythagorean
    def distance_to(self, tx, ty):
        return math.sqrt((tx - self.x)**2 + (ty - self.y)**2)

    # normalize angle if necessary
    def normalize_angle(self, a):
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

    # ------------Robot Motion----------------------

    # stop the robot
    def stop(self):
        self.cmd_pub.publish(Twist())

    # rotate the robot
    def rotate(self, target):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            error = self.normalize_angle(target - self.yaw)
            if abs(error) < self.angle_tolerance:
                self.stop()
                return
            twist = Twist()
            twist.angular.z = max(-self.angular_speed, min(self.angular_speed, 2.0 * error))
            self.cmd_pub.publish(twist)
            rate.sleep()

    # go to specified point
    def nav_to_point(self, tx, ty):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            # check if reached [point]
            if self.distance_to(tx, ty) < self.goal_tolerance:
                self.stop()
                return
            heading_error = self.normalize_angle(self.angle_to(tx, ty) - self.yaw)
            twist = Twist()
            twist.linear.x  = self.linear_speed
            twist.angular.z = max(-self.angular_speed, min(self.angular_speed, 1.5 * heading_error))
            self.cmd_pub.publish(twist)
            rate.sleep()
    
    # follow path
    def follow_path(self):
        self.running = True
        rate = rospy.Rate(10)
        for i, (wx, wy) in enumerate(self.path):
            self.rotate(self.angle_to(wx, wy))
            rate.sleep()
            self.nav_to_point(wx, wy)
            rate.sleep()

        rospy.loginfo("Path complete!")
        self.stop()
        self.running = False


if __name__ == '__main__':
    PathController()