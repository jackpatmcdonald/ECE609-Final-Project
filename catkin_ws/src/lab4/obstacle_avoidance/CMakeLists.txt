cmake_minimum_required(VERSION 3.0.2)
project(obstacle_avoidance)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  message_generation
)

# Custom message: ObstacleArray
add_message_files(
  FILES
  Obstacle.msg
  ObstacleArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    nav_msgs
    message_runtime
)

catkin_install_python(PROGRAMS
  src/obstacle_sensor.py
  src/potential_field_planner.py
  src/baseline_planner.py
  src/obstacle_spawner.py
  src/metrics_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
