#!/usr/bin/env python3

import rospy
import csv
import os
import time
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from obstacle_avoidance.msg import ObstacleArray
import numpy as np

# output csv to compare trials - always appends
OUTPUT_FILE = os.path.expanduser("~/obstacle_avoidance_results.csv")
FIELDNAMES  = [
    "trial", "strategy", "density", "success",
    "path_length_m", "avg_replanning_latency_ms",
    "duration_s", "timestamp"
]

class MetricsRecorder:
    def __init__(self):
        rospy.init_node("metrics_recorder")

        self.strategy = rospy.get_param("~strategy", "potential_field")
        self.density = rospy.get_param("~density",  "medium")
        self.trial = rospy.get_param("~trial",    0)

        #intial states
        self.path_length = 0.0
        self.prev_pos = None
        self.status = "RUNNING"
        self.start_time = rospy.Time.now()

        # Latency: time between last obstacle detection and cmd_vel publication
        self.last_obs_time = None
        self.latencies_ms = []

        #subscribers
        rospy.Subscriber("/odom", Odometry, self.odom_cb)
        rospy.Subscriber("/planner_status", String, self.status_cb)
        rospy.Subscriber("/obstacles", ObstacleArray, self.obstacle_cb)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_cb)

        # Ensure CSV exists
        if not os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

        rospy.loginfo(f"MetricsRecorder — trial {self.trial}, strategy={self.strategy}, density={self.density}")
        rospy.on_shutdown(self._save)
        rospy.spin()

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        pos = np.array([p.x, p.y])
        if self.prev_pos is not None:
            self.path_length += float(np.linalg.norm(pos - self.prev_pos))
        self.prev_pos = pos

    def status_cb(self, msg: String):
        self.status = msg.data
        if msg.data in ("SUCCESS", "COLLISION"):
            self._save()

    def obstacle_cb(self, msg: ObstacleArray):
        if msg.obstacles:
            self.last_obs_time = time.monotonic()

    # measure latency (see above)
    def cmd_vel_cb(self, msg: Twist):
        if self.last_obs_time is not None:
            latency_ms = (time.monotonic() - self.last_obs_time) * 1000.0
            if latency_ms < 2000:   # ignore stale pairs
                self.latencies_ms.append(latency_ms)
            self.last_obs_time = None
    # save file
    def _save(self):
        duration = (rospy.Time.now() - self.start_time).to_sec()
        avg_lat  = float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0
        success  = 1 if self.status == "SUCCESS" else 0

        row = {
            "trial": self.trial,
            "strategy": self.strategy,
            "density": self.density,
            "success": success,
            "path_length_m": round(self.path_length, 3),
            "avg_replanning_latency_ms": round(avg_lat, 2),
            "duration_s": round(duration, 2),
            "timestamp": rospy.Time.now().to_sec(),
        }

        # write file
        with open(OUTPUT_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
        # confirm
        rospy.loginfo(
            f"[Metrics] Trial {self.trial} saved — "
            f"success={success}, path={self.path_length:.2f}m, "
            f"lat={avg_lat:.1f}ms, duration={duration:.1f}s"
        )

if __name__ == "__main__":
    MetricsRecorder()
