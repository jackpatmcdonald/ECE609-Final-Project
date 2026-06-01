#!/usr/bin/env python3
import rospy
import numpy as np
from typing import Dict, List, Optional, Tuple
from sensor_msgs.msg import LaserScan
from obstacle_avoidance.msg import Obstacle, ObstacleArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Vector3


# ── parameters ──────────────────────────────────────────────────────
cluster_distance_thresh = 0.3   # max gap (m) between points in the same cluster
min_cluster_size = 3  # ignore clusters smaller than this (noise filter)
max_range = 3.5   # ignore readings beyond this distance (m)
alpha = 0.4   # low-pass filter weight for velocity estimates
# ────────────────────────────────────────────────────────────────────────────

class ObstacleSensor:
    def __init__(self):
        rospy.init_node("obstacle_sensor")

        self.pub = rospy.Publisher("/obstacles", ObstacleArray, queue_size=10)
        rospy.Subscriber("/scan", LaserScan, self.scan_cb)

        # State for velocity estimation
        self.prev_centroids: Dict[int, np.ndarray] = {}   # id -> (x, y)
        self.prev_velocities: Dict[int, np.ndarray] = {}  # id -> (vx, vy)
        self.prev_time: Optional[rospy.Time] = None
        self.next_id = 0

        rospy.loginfo("ObstacleSensor ready — listening on /scan")
        rospy.spin()

    # scan callback
    def scan_cb(self, msg: LaserScan):
        now = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        dt = (now - self.prev_time).to_sec() if self.prev_time else None
        self.prev_time = now

        points = self._scan_to_points(msg)
        clusters = self._cluster_points(points)
        centroids, radii = self._cluster_stats(clusters)

        # Associate clusters with previous centroids (greedy nearest-neighbour)
        ids, velocities = self._associate_and_estimate(centroids, dt)

        # Build and publish message
        out = ObstacleArray()
        out.header = Header(stamp=now, frame_id=msg.header.frame_id)
        for i, (c, r, oid, v) in enumerate(zip(centroids, radii, ids, velocities)):
            obs = Obstacle()
            obs.header = out.header
            obs.position = Point(x=c[0], y=c[1], z=0.0)
            obs.velocity = Vector3(x=v[0], y=v[1], z=0.0)
            obs.radius = float(r)
            obs.id = oid
            out.obstacles.append(obs)
        self.pub.publish(out)

    # convert polar coordinates to cartesian
    def _scan_to_points(self, msg: LaserScan) -> np.ndarray:
        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        ranges = np.array(msg.ranges, dtype=float)

        # Filter invalid / out-of-range readings
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < max_range)
        r = ranges[valid]
        a = angles[valid]

        x = r * np.cos(a)
        y = r * np.sin(a)
        return np.column_stack((x, y))  # shape (N, 2)
    
    # distance-threshold clustering of points
    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        if len(points) == 0:
            return []

        clusters: List[List[int]] = []
        current: List[int] = [0]

        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - points[i - 1]) < cluster_distance_thresh:
                current.append(i)
            else:
                if len(current) >= min_cluster_size:
                    clusters.append(np.array(current))
                current = [i]

        if len(current) >= min_cluster_size:
            clusters.append(np.array(current))

        return [points[idx] for idx in clusters]

    # return centroid and bounding radius for cluster
    def _cluster_stats(self, clusters: List[np.ndarray]) -> Tuple[list, list]:
        centroids, radii = [], []
        for pts in clusters:
            c = pts.mean(axis=0)
            r = np.max(np.linalg.norm(pts - c, axis=1))
            centroids.append(c)
            radii.append(max(r, 0.1))   # min radius 10cm
        return centroids, radii

    # nearest-neighbor association between current and previous scan
    def _associate_and_estimate(
        self, centroids: list, dt: Optional[float]
    ) -> Tuple[List[int], List[np.ndarray]]:
        if not self.prev_centroids:
            # First scan — assign new IDs, zero velocity
            ids, vels = [], []
            for c in centroids:
                ids.append(self.next_id)
                vels.append(np.zeros(2))
                self.prev_centroids[self.next_id] = c
                self.prev_velocities[self.next_id] = np.zeros(2)
                self.next_id += 1
            return ids, vels

        prev_ids = list(self.prev_centroids.keys())
        prev_pts = np.array([self.prev_centroids[i] for i in prev_ids])

        ids, vels = [], []
        used_prev = set()

        for c in centroids:
            if len(prev_pts) > 0:
                dists = np.linalg.norm(prev_pts - c, axis=1)
                best  = int(np.argmin(dists))
                pid   = prev_ids[best]

                if dists[best] < 1.0 and pid not in used_prev:
                    # Matched — update velocity estimate
                    used_prev.add(pid)
                    oid = pid
                    if dt and dt > 0:
                        raw_v = (c - self.prev_centroids[pid]) / dt
                        prev_v = self.prev_velocities.get(pid, np.zeros(2))
                        v = alpha * raw_v + (1 - alpha) * prev_v
                    else:
                        v = self.prev_velocities.get(pid, np.zeros(2))
                else:
                    # New obstacle
                    oid = self.next_id
                    self.next_id += 1
                    v = np.zeros(2)
            else:
                oid = self.next_id
                self.next_id += 1
                v = np.zeros(2)

            ids.append(oid)
            vels.append(v)
            self.prev_centroids[oid]  = c
            self.prev_velocities[oid] = v

        # pop disappeared obstacles
        active = set(ids)
        for pid in list(self.prev_centroids.keys()):
            if pid not in active:
                del self.prev_centroids[pid]
                self.prev_velocities.pop(pid, None)
        return ids, vels


if __name__ == "__main__":
    ObstacleSensor()
