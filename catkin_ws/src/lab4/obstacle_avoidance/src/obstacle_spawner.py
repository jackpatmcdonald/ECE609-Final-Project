#!/usr/bin/env python3

import rospy
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel, SpawnModelRequest
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from std_msgs.msg import String

# launch file - called when lab is launched
CYLINDER_SDF = """
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <collision name="col">
        <geometry><cylinder><radius>0.15</radius><length>0.6</length></cylinder></geometry>
      </collision>
      <visual name="vis">
        <geometry><cylinder><radius>0.15</radius><length>0.6</length></cylinder></geometry>
        <material>
          <ambient>1 0.2 0 1</ambient>
          <diffuse>1 0.2 0 1</diffuse>
        </material>
      </visual>
      <inertial><mass>5.0</mass></inertial>
    </link>
  </model>
</sdf>
"""

DENSITY_MAP = {"low": 2, "medium": 4, "high": 7}

class ObstacleSpawner:
    def __init__(self):
        rospy.init_node("obstacle_spawner")

        density = rospy.get_param("~density", "medium")
        num = rospy.get_param("~num_obstacles",  DENSITY_MAP.get(density, 3))
        self.speed = rospy.get_param("~obstacle_speed", 0.3)

        rospy.loginfo(f"Spawning {num} obstacles (density={density})")

        # Wait for Gazebo spawn service
        rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=15.0)
        self.spawn_srv  = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.delete_srv = rospy.ServiceProxy("/gazebo/delete_model",    DeleteModel)

        # Obstacle publishers 
        self.names: list[str] = []
        self.cmd_pubs: list[rospy.Publisher] = []
        self.phases: list[float] = []

        self._spawn_obstacles(num)

        self.rate = rospy.Rate(10)
        self._move_loop()

    # add the obstacles - n=number of obstacles
    def _spawn_obstacles(self, n: int):
        for i in range(n):
            name = f"dyn_obs_{i}"

            # start positions along x, offset in y
            x = 1.0 + i * (7.0 / max(n - 1, 1))
            y = 1.5 * math.cos(i * math.pi / max(n - 1, 1) * 2)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.3
            pose.orientation.w = 1.0

            req = SpawnModelRequest()
            req.model_name = name
            req.model_xml = CYLINDER_SDF.format(name=name)
            req.initial_pose = pose
            req.reference_frame = "world"

            try:
                self.spawn_srv(req)
                self.names.append(name)
                self.phases.append(i * math.pi / 3.0)   # stagger oscillations
                rospy.loginfo(f"  Spawned {name} at ({x:.2f}, {y:.2f})")
            except Exception as e:
                rospy.logerr(f"Failed to spawn {name}: {e}")

    # move obstacle in sinusoidal patterns
    def _move_loop(self):
        from gazebo_msgs.msg import ModelState
        from gazebo_msgs.srv import SetModelState

        rospy.wait_for_service("/gazebo/set_model_state")
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        t = 0.0
        while not rospy.is_shutdown():
            t += 0.1   # matches 10 Hz rate

            for i, name in enumerate(self.names):
                phase = self.phases[i]
                # x positions match spawn spread (1–8 m) so there's no teleport on first tick
                x_base = 1.0 + i * (7.0 / max(len(self.names) - 1, 1))

                state = ModelState()
                state.model_name = name
                state.reference_frame = "world"
                state.pose.position.x = x_base
                state.pose.position.y = 1.5 * math.sin(self.speed * t + phase)
                state.pose.position.z = 0.3
                state.pose.orientation.w = 1.0
                # Set velocity so the sensor can estimate it
                state.twist.linear.y = self.speed * 1.5 * math.cos(self.speed * t + phase)

                try:
                    set_state(state)
                except Exception as e:
                    rospy.logwarn_once(f"set_model_state failed: {e}")
            self.rate.sleep()

    def _cleanup(self):
        for name in self.names:
            try:
                self.delete_srv(name)
            except Exception:
                pass

if __name__ == "__main__":
    spawner = ObstacleSpawner()
    rospy.on_shutdown(spawner._cleanup)
    rospy.spin()