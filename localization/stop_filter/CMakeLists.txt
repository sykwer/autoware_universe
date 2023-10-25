cmake_minimum_required(VERSION 3.14)
project(stop_filter)

find_package(autoware_cmake REQUIRED)
find_package(rclcpp_component_container_callback_isolated REQUIRED)
autoware_package()

ament_auto_add_executable(stop_filter
  src/stop_filter_node.cpp
  src/stop_filter.cpp
)
ament_target_dependencies(stop_filter rclcpp_component_container_callback_isolated)
target_include_directories(stop_filter PRIVATE
  ${rclcpp_component_container_callback_isolated_INCLUDE_DIRS})

ament_auto_package(
  INSTALL_TO_SHARE
  launch
)
