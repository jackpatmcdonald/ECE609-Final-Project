#!/usr/bin/env python3

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from obstacle_avoidance.msg import ObstacleArray
import tf.transformations as tft


# ── parameters ────────────────────────────────────────────────────────────
K_att = 0.8 # attractive gain toward goal
K_rep = 1.2 # repulsive gain from obstacles
d0 = 0.8 # influence radius (m) — obstacles beyond this are ignored
goal_thresh  = 0.15 # distance (m) to declare "goal reached"
max_lin_vel = 0.22 # TurtleBot3 Waffle Pi max linear speed (m/s)
max_omega = 2.84 # TurtleBot3 Waffle Pi max angular speed (rad/s)
stuck_timer = 3.0 # seconds without progress before declaring "stuck"
stuck_dist = 0.05 # minimum progress (m) in stuck_timer window
# ─────────────────────────────────────────────────────────────────────────────


class PotentialFieldPlanner:
    def __init__(self):
        rospy.init_node("potential_field_planner")

        goal = np.array([
            rospy.get_param("~goal_x", 3.0),
            rospy.get_param("~goal_y", 0.0),])
        
        raw = rospy.get_param("~waypoints", [])
        shelf_wps = [np.array(raw[i:i+2]) for i in range(0, len(raw) - 1, 2)]
        self.waypoints = shelf_wps + [goal]
        self.wp_idx    = 0
        rospy.loginfo(f"PotentialField planner — {len(self.waypoints)} waypoints, "
                      f"final goal ({goal[0]:.2f}, {goal[1]:.2f})")

        self.pose = np.zeros(2) # (x, y)
        self.yaw = 0.0
        self.obstacles: list = []
        self.running = False

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist,queue_size=1)
        self.status_pub = rospy.Publisher("/planner_status", String, queue_size=10)

        rospy.Subscriber("/odom", Odometry, self.odom_cb)
        rospy.Subscriber("/obstacles", ObstacleArray, self.obstacle_cb)

        # Stuck detection
        self.last_progress_time = rospy.Time.now()
        self.last_progress_pose = self.pose.copy()

        self.rate = rospy.Rate(10)  # 10 Hz control loop
        rospy.loginfo("PotentialFieldPlanner ready")
        self.run()

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.pose = np.array([p.x, p.y])
        q = msg.pose.pose.orientation
        _, _, self.yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.running = True

    def obstacle_cb(self, msg: ObstacleArray):
        self.obstacles = msg.obstacles

    # Main control loop
    def run(self):
        while not rospy.is_shutdown():
            if not self.running:
                self.rate.sleep()
                continue
            # get waypoints
            dist = np.linalg.norm(self.waypoints[self.wp_idx] - self.pose)

            if dist < goal_thresh:
                if self.wp_idx < len(self.waypoints) - 1:
                    rospy.loginfo(f"PotentialField: waypoint {self.wp_idx} reached, "
                                  f"heading to waypoint {self.wp_idx + 1}")
                    self.wp_idx += 1
                    self.last_progress_time = rospy.Time.now()
                    self.last_progress_pose = self.pose.copy()
                else:
                    self._stop()
                    self.status_pub.publish("SUCCESS")
                    rospy.loginfo("Goal reached!")
                    break

            # Compute resultant force vector
            f_att = self._attractive_force()
            f_rep = self._repulsive_force()
            f_total = f_att + f_rep

            # Stuck detection
            if self._check_stuck():
                self.status_pub.publish("STUCK")
                rospy.logwarn("Robot appears stuck — applying escape jitter")
                self._escape_jitter()
            else:
                self.status_pub.publish("RUNNING")
                self._publish_cmd(f_total)

            self.rate.sleep()
        self._stop()

    # force computations

    # force goal
    def _attractive_force(self) -> np.ndarray:
        diff = self.waypoints[self.wp_idx] - self.pose
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2)
        return K_att * diff / dist   # unit vector scaled by gain

    # sum of repulsive forces at a given moment
    def _repulsive_force(self) -> np.ndarray:
        f = np.zeros(2)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        for obs in self.obstacles:
            # Distance in robot frame = magnitude of the position vector
            obs_local = np.array([obs.position.x, obs.position.y])
            d = np.linalg.norm(obs_local)
            if d < 1e-6 or d > d0:
                continue
            magnitude = K_rep * (1.0 / d - 1.0 / d0) * (1.0 / d ** 2)
            # Unit vector from obstacle toward robot in robot frame = -obs_local/d
            # Rotate to world frame via R(yaw)
            rep_robot = -obs_local / d
            rep_world = np.array([c * rep_robot[0] - s * rep_robot[1], s * rep_robot[0] + c * rep_robot[1]])
            f += magnitude * rep_world
        return f

    # Velocity command - command 2d vector into velocity
    def _publish_cmd(self, force: np.ndarray):
        force_angle = math.atan2(force[1], force[0])
        angle_error = self._normalize_angle(force_angle - self.yaw)

        cmd = Twist()
        # Scale linear speed down when turning sharply
        cmd.linear.x  = np.clip(np.linalg.norm(force) * math.cos(angle_error), 0.0, max_lin_vel)
        cmd.angular.z = np.clip(2.0 * angle_error, -max_omega, max_omega)
        self.cmd_pub.publish(cmd)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    # escape if stuck - rotate in place
    def _escape_jitter(self):
        cmd = Twist()
        cmd.angular.z = 1.0
        self.cmd_pub.publish(cmd)
        rospy.sleep(0.5)
        self.last_progress_time = rospy.Time.now()
        self.last_progress_pose = self.pose.copy()

    # check if stuck
    def _check_stuck(self) -> bool:
        moved = np.linalg.norm(self.pose - self.last_progress_pose)
        elapsed = (rospy.Time.now() - self.last_progress_time).to_sec()
        if moved > stuck_dist:
            self.last_progress_time = rospy.Time.now()
            self.last_progress_pose = self.pose.copy()
            return False
        return elapsed > stuck_timer

    @staticmethod
    def _normalize_angle(a: float) -> float:
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

if __name__ == "__main__":
    PotentialFieldPlanner()
