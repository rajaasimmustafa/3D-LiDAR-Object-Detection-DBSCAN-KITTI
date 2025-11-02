cmake_minimum_required(VERSION 3.0.2)
project(lidar_detect_ros)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs visualization_msgs std_msgs)
catkin_package()
catkin_install_python(PROGRAMS
  nodes/lidar_bounding_box.py
  nodes/match_clusters_to_labels.py
  nodes/publish_kitti_bin.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
