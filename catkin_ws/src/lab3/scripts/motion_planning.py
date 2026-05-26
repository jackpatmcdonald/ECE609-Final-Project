#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import csv
import time
import sys
from sensor_msgs.msg import JointState

# Inverse kinematics
from inv_kinematics import inverse_kinematics, make_pose

planners = ["RRTConnectkConfigDefault", "PRMkConfigDefault", "KPIECEkConfigDefault"]

# Target poses
target_poses = [
    {"name": "Pose A", "xyz": [0.25,  0.00, 0.15], "ry": np.pi/6},
    {"name": "Pose B", "xyz": [0.15,  0.15, 0.20], "ry": np.pi/4},
    {"name": "Pose C", "xyz": [0.20, -0.10, 0.08], "ry": np.pi/3},
]

# Corresponding goals
target_joints = [
    {"name": "Pose A", "q": [0.30, -0.50, 0.80, -0.30]},
    {"name": "Pose B", "q": [0.70, -0.40, 0.70, -0.25]},
    {"name": "Pose C", "q": [-0.30, -0.70, 1.00, -0.20]},
]

class PlanningExperiment:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("planning_experiment", anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("arm")

        self.group.set_max_velocity_scaling_factor(0.3)
        self.group.set_max_acceleration_scaling_factor(0.3)
        self.group.set_num_planning_attempts(5)
        self.group.set_planning_time(10.0)

        # Storage for recorded trajectories
        self.joint_states_log = []
        self.joint_sub = rospy.Subscriber(
            "/joint_states", JointState, self._joint_cb)

        self.results = []  # all experiment results
        rospy.sleep(1.0)
        rospy.loginfo("PlanningExperiment ready.")

    def _joint_cb(self, msg):
        self.joint_states_log.append({
            "time": msg.header.stamp.to_sec(),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort":   list(msg.effort),
        })

    def go_home(self):
        """Return to the named home configuration."""
        self.group.set_named_target("home")
        self.group.go(wait=True)
        self.group.stop()
        rospy.sleep(1.0)

    # ── Task-space planning (Task 4) ───────────────────────────────────────
    def plan_task_space(self, pose_goal, planner):
        self.group.set_planner_id(planner)

        target = geometry_msgs.msg.Pose()
        target.position.x = pose_goal["xyz"][0]
        target.position.y = pose_goal["xyz"][1]
        target.position.z = pose_goal["xyz"][2]

        # Convert ry euler angle to quaternion
        ry = pose_goal["ry"]
        target.orientation.x = 0.0
        target.orientation.y = np.sin(ry / 2)
        target.orientation.z = 0.0
        target.orientation.w = np.cos(ry / 2)

        self.group.set_pose_target(target)
        return self._execute_and_record(pose_goal["name"], planner, "task_space")

    # ── Joint-space planning (Task 5) ──────────────────────────────────────
    def plan_joint_space(self, joint_goal, planner):
        self.group.set_planner_id(planner)
        self.group.set_joint_value_target(joint_goal["q"])
        return self._execute_and_record(joint_goal["name"], planner, "joint_space")

    # ── Common execution + recording ───────────────────────────────────────
    def _execute_and_record(self, pose_name, planner, space):
        self.joint_states_log.clear()

        t_start = time.time()
        plan = self.group.plan()
        planning_time = time.time() - t_start

        # plan() returns (success, trajectory, planning_time, error_code)
        success = plan[0]
        trajectory = plan[1]

        if not success:
            rospy.logwarn(f"Planning failed: {pose_name} / {planner} / {space}")
            return None

        # Execute
        self.group.execute(trajectory, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.sleep(0.5)

        # Final pose error
        current_pose = self.group.get_current_pose().pose
        result = {
            "pose":          pose_name,
            "planner":       planner,
            "space":         space,
            "planning_time": planning_time,
            "final_x":       current_pose.position.x,
            "final_y":       current_pose.position.y,
            "final_z":       current_pose.position.z,
            "joint_log":     list(self.joint_states_log),
        }
        self.results.append(result)
        rospy.loginfo(f"Done: {pose_name} | {planner} | {space} | "
                      f"plan_time={planning_time:.3f}s")
        return result

    # ── Run all experiments ────────────────────────────────────────────────
    def run_all(self):
        # Task 4: task-space
        rospy.loginfo("=== TASK 4: Task-Space Planning ===")
        for pose in target_poses:
            for planner in planners:
                self.go_home()
                self.plan_task_space(pose, planner)

        # Task 5: joint-space
        rospy.loginfo("=== TASK 5: Joint-Space Planning ===")
        for jgoal in target_joints:
            for planner in planners:
                self.go_home()
                self.plan_joint_space(jgoal, planner)

    # ── Save results to CSV ────────────────────────────────────────────────
    def save_csv(self, path="/tmp/planning_results.csv"):
        keys = ["pose", "planner", "space", "planning_time",
                "final_x", "final_y", "final_z"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.results:
                writer.writerow({k: r[k] for k in keys})
        rospy.loginfo(f"Results saved to {path}")


if __name__ == "__main__":
    exp = PlanningExperiment()
    exp.run_all()
    exp.save_csv()
    moveit_commander.roscpp_shutdown()