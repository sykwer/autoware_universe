cmake_minimum_required(VERSION 3.14)
project(twist2accel)

find_package(autoware_cmake REQUIRED)
find_package(rclcpp_component_container_callback_isolated REQUIRED)

autoware_package()

ament_auto_add_executable(twist2accel
  src/twist2accel_node.cpp
  src/twist2accel.cpp
)
ament_target_dependencies(twist2accel rclcpp_component_container_callback_isolated)
target_include_directories(twist2accel PRIVATE
  ${rclcpp_component_container_callback_isolated_})

ament_auto_package(
  INSTALL_TO_SHARE
  launch
)
