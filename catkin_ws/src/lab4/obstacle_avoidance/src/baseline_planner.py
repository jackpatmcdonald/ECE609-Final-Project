#!/usr/bin/env python3

import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from obstacle_avoidance.msg import ObstacleArray
import tf.transformations as tft

# ── Parameters ───────────────────────────────────────────────────────────────
clearance_radius = 0.5   # stop if any obstacle is within this distance (m)
goal_threshold = 0.15  # goal reached tolerance (m)
linear_speed = 0.15  # m/s — slow, steady baseline
angular_gain = 2.0   # proportional gain for heading correction
max_omega = 2.84

class BaselinePlanner:
    def __init__(self):
        rospy.init_node("baseline_planner")
        # set goal
        goal = np.array([
            rospy.get_param("~goal_x", 3.0),
            rospy.get_param("~goal_y", 0.0),
        ])
        raw = rospy.get_param("~waypoints", [])
        shelf_wps = [np.array(raw[i:i+2]) for i in range(0, len(raw) - 1, 2)]
        self.waypoints = shelf_wps + [goal]
        self.wp_idx    = 0
        rospy.loginfo(f"Baseline planner — {len(self.waypoints)} waypoints, "
                      f"final goal ({goal[0]:.2f}, {goal[1]:.2f})")

        # inital states
        self.pose = np.zeros(2)
        self.yaw = 0.0
        self.obstacles = []
        self.running = False
        # publishers
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.status_pub = rospy.Publisher("/planner_status", String, queue_size=10)

        # set subscribers
        rospy.Subscriber("/odom", Odometry, self.odom_cb)
        rospy.Subscriber("/obstacles", ObstacleArray, self.obstacle_callback)

        self.rate = rospy.Rate(10)
        rospy.loginfo("BaselinePlanner ready")
        self.run()

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.pose = np.array([p.x, p.y])
        q = msg.pose.pose.orientation
        _, _, self.yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.running = True

    def obstacle_callback(self, msg: ObstacleArray):
        self.obstacles = msg.obstacles

    def run(self):
        while not rospy.is_shutdown():
            if not self.running:
                self.rate.sleep()
                continue

            target = self.waypoints[self.wp_idx]
            dist = np.linalg.norm(target - self.pose)

            if dist < goal_threshold:
                if self.wp_idx < len(self.waypoints) - 1:
                    rospy.loginfo(f"Baseline: waypoint {self.wp_idx} reached, "
                                  f"heading to waypoint {self.wp_idx + 1}")
                    self.wp_idx += 1
                else:
                    self._stop()
                    self.status_pub.publish("SUCCESS")
                    rospy.loginfo("Baseline: goal reached!")
                    break

            elif self._path_blocked():
                self._stop()
                self.status_pub.publish("WAITING")
            else:
                self._drive_toward(target)
                self.status_pub.publish("RUNNING")
            self.rate.sleep()
        self._stop()

    # true if obstacle is in clearance
    def _path_blocked(self) -> bool:
        for obs in self.obstacles:
            if math.hypot(obs.velocity.x, obs.velocity.y) < 0.05:
                continue  # static object — ignore
            dist = math.hypot(obs.position.x, obs.position.y)
            if dist < clearance_radius + obs.radius:
                return True
        return False

    # command robot to drive
    def _drive_toward(self, target: np.ndarray):
        diff = target - self.pose
        target_yaw = math.atan2(diff[1], diff[0])
        angle_error = self._normalize_angle(target_yaw - self.yaw)

        cmd = Twist()
        cmd.linear.x  = linear_speed * max(0.0, math.cos(angle_error))
        cmd.angular.z = np.clip(angular_gain * angle_error, -max_omega, max_omega)
        self.cmd_pub.publish(cmd)
    
    # stop robot
    def _stop(self):
        self.cmd_pub.publish(Twist())

    @staticmethod
    def _normalize_angle(a):
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

if __name__ == "__main__":
    BaselinePlanner()
