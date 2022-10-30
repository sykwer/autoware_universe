// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pointcloud_preprocessor/outlier_filter/ring_outlier_filter_nodelet.hpp"

#include <algorithm>
#include <vector>
#include <iostream>
#include <sys/mman.h>

namespace pointcloud_preprocessor
{

const size_t CHUNK_SIZE = 2050; // >=2048
const size_t CHUNKS_NUM = 130; // >= 128

template <typename T>
class MyAllocator {
public:
  using value_type = T;

  value_type *addr;
  size_t size;

  MyAllocator(value_type *addr, size_t sz) : addr(addr), size(sz) {}

  template <class U>
  MyAllocator(const MyAllocator<U>& other) : addr(other.addr), size(other.size) {}

  value_type* allocate(size_t n) {
    if (n > size) {
      std::cerr << "trying to allocate more than the pre-allocated memory pool size: n=" << n << ", size=" << size << std::endl;
      return NULL;
    }

    return addr;
  }

  void deallocate(value_type* p, size_t n) {
    (void) p;
    (void) n;
  }
};

template <typename T>
class MemPool {
public:
  MemPool(size_t chunk_size, size_t chunks_num) : chunk_size_(chunk_size), chunks_num_(chunks_num) {
    chunks_.reserve(chunks_num);
    void *addr = mmap(NULL, chunk_size_ * sizeof(T) * chunks_num_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    T *tp = (T*) addr;

    // memory touch (fill page table entry)
    for (size_t i = 0; i < chunk_size_ * chunks_num_; i++) {
      volatile T *tmp = tp + i;
      *tmp = 0;
    }

    for (size_t i = 0; i < chunks_num_; i++) {
      chunks_.emplace_back(tp, chunk_size_);
      tp += chunk_size_;
    }
  }

  ~MemPool() {
    munmap(chunks_.begin()->first, chunk_size_ * sizeof(T) * chunks_num_);
    chunks_.clear();
  }

  void push(T* tp, size_t chunk_size) {
    chunks_.emplace_back(tp, chunk_size);
  }

  std::pair<T*, size_t> pop() {
    if (chunks_.size() == 0) return std::make_pair((T*) NULL, (size_t) 0);
    std::pair<T*, size_t> ret = chunks_.back();
    chunks_.pop_back();
    return ret;
  }

private:
  size_t chunk_size_;
  size_t chunks_num_;
  std::vector<std::pair<T*, size_t>> chunks_; // <addr, size>
};


RingOutlierFilterComponent::RingOutlierFilterComponent(const rclcpp::NodeOptions & options)
: Filter("RingOutlierFilter", options)
{
  mp_ = std::make_unique<MemPool<std::size_t>>(CHUNK_SIZE, CHUNKS_NUM);

  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ = std::make_unique<DebugPublisher>(this, "ring_outlier_filter");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }

  // set initial parameters
  {
    distance_ratio_ = static_cast<double>(declare_parameter("distance_ratio", 1.03));
    object_length_threshold_ =
      static_cast<double>(declare_parameter("object_length_threshold", 0.1));
    num_points_threshold_ = static_cast<int>(declare_parameter("num_points_threshold", 4));
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&RingOutlierFilterComponent::paramCallback, this, _1));
}

void RingOutlierFilterComponent::faster_filter(const PointCloud2ConstPtr &input, PointCloud2 &output, const Eigen::Matrix4f &eigen_transform, bool need_transform) {
  // TODO: Implement conversion between frames
  (void) eigen_transform;
  (void) need_transform;

  std::scoped_lock lock(mutex_);
  stop_watch_ptr_->toc("processing_time", true);

  std::size_t output_size = 0;
  output.point_step = sizeof(PointXYZI);

  const auto ring_offset = input->fields[static_cast<size_t>(autoware_point_types::PointIndex::Ring)].offset;
  const auto azimuth_offset = input->fields[static_cast<size_t>(autoware_point_types::PointIndex::Azimuth)].offset;
  const auto distance_offset = input->fields[static_cast<size_t>(autoware_point_types::PointIndex::Distance)].offset;

  std::vector<std::vector<std::size_t, MyAllocator<std::size_t>>> ring2indices;
  ring2indices.reserve(128);
  for (int i = 0; i < 128; i++) {
    size_t *addr, chunk_size;
    std::tie(addr, chunk_size) = mp_->pop();
    MyAllocator<std::size_t> alloc(addr, chunk_size);
    ring2indices.push_back(std::vector<std::size_t, MyAllocator<std::size_t>>(alloc));
    ring2indices.back().reserve(2000);
  }

  for (std::size_t data_idx = 0U; data_idx < input->data.size(); data_idx += input->point_step) {
    const uint16_t ring = *reinterpret_cast<const uint16_t*>(&input->data[data_idx + ring_offset]);
    ring2indices[ring].push_back(data_idx);
  }

  // walk range: [walk_first_i, walk_last_i]
  int walk_first_i = 0;
  int walk_last_i = -1;

  for (const auto &indices : ring2indices) {
    if (indices.size() < 2) continue;
    walk_first_i = 0;

    for (std::size_t i = 0U; i < indices.size() - 1; i++) {
      const std::size_t &current_data_idx = indices[i];
      const std::size_t &next_data_idx = indices[i + 1];
      walk_last_i = i;

      const float &current_azimuth = *reinterpret_cast<const float*>(&input->data[current_data_idx + azimuth_offset]);
      const float &next_azimuth = *reinterpret_cast<const float*>(&input->data[next_data_idx + azimuth_offset]);
      float azimuth_diff = next_azimuth - current_azimuth;
      azimuth_diff = azimuth_diff < 0.f ? azimuth_diff + 36000.f : azimuth_diff;

      const float &current_distance = *reinterpret_cast<const float*>(&input->data[current_data_idx + distance_offset]);
      const float &next_distance = *reinterpret_cast<const float*>(&input->data[next_data_idx + distance_offset]);
      if (std::max(current_distance, next_distance) < std::min(current_distance, next_distance) * distance_ratio_ && azimuth_diff < 100.f) {
        continue; // Determined to be included in the same walk
      }

      if (isCluster(input, indices[walk_first_i], indices[walk_last_i], walk_last_i - walk_first_i + 1)) {
        for (int i = walk_first_i; i <= walk_last_i; i++) {
          PointXYZI *output_ptr = reinterpret_cast<PointXYZI*>(&output.data[output_size]);
          *output_ptr = *reinterpret_cast<const PointXYZI*>(&input->data[indices[i]]);
          output_size += output.point_step;
        }
      }

      walk_first_i = i + 1;
    }

    if (walk_first_i > walk_last_i) continue;

    if (isCluster(input, indices[walk_first_i], indices[walk_last_i], walk_last_i - walk_first_i + 1)) {
      for (int i = walk_first_i; i <= walk_last_i; i++) {
        PointXYZI *output_ptr = reinterpret_cast<PointXYZI*>(&output.data[output_size]);
        *output_ptr = *reinterpret_cast<const PointXYZI*>(&input->data[indices[i]]);
        output_size += output.point_step;
      }
    }
  }

  for (auto it = ring2indices.begin(); it != ring2indices.end(); it++) {
    auto alloc = it->get_allocator();
    mp_->push(alloc.addr, alloc.size);
  }

  output.data.resize(output_size);
  output.header.frame_id =input->header.frame_id;
  output.height = 1;
  output.fields = input->fields;
  output.is_bigendian = input->is_bigendian;
  output.is_dense = input->is_dense;
  output.width = static_cast<uint32_t>(output.data.size() / output.height / output.point_step);
  output.row_step = static_cast<uint32_t>(output.data.size() / output.height);

  // debug publisher here
}

void RingOutlierFilterComponent::filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output)
{
  std::scoped_lock lock(mutex_);
  stop_watch_ptr_->toc("processing_time", true);
  std::unordered_map<uint16_t, std::vector<std::size_t>> input_ring_map;
  input_ring_map.reserve(128);
  sensor_msgs::msg::PointCloud2::SharedPtr input_ptr =
    std::make_shared<sensor_msgs::msg::PointCloud2>(*input);

  const auto ring_offset =
    input->fields.at(static_cast<size_t>(autoware_point_types::PointIndex::Ring)).offset;
  for (std::size_t idx = 0U; idx < input_ptr->data.size(); idx += input_ptr->point_step) {
    input_ring_map[*reinterpret_cast<uint16_t *>(&input_ptr->data[idx + ring_offset])].push_back(
      idx);
  }

  PointCloud2Modifier<PointXYZI> output_modifier{output, input->header.frame_id};
  output_modifier.reserve(input->width);

  std::vector<std::size_t> tmp_indices;
  tmp_indices.reserve(input->width);

  const auto azimuth_offset =
    input->fields.at(static_cast<size_t>(autoware_point_types::PointIndex::Azimuth)).offset;
  const auto distance_offset =
    input->fields.at(static_cast<size_t>(autoware_point_types::PointIndex::Distance)).offset;
  for (const auto & ring_indices : input_ring_map) {
    if (ring_indices.second.size() < 2) {
      continue;
    }

    for (size_t idx = 0U; idx < ring_indices.second.size() - 1; ++idx) {
      const auto & current_idx = ring_indices.second.at(idx);
      const auto & next_idx = ring_indices.second.at(idx + 1);
      tmp_indices.emplace_back(current_idx);

      // if(std::abs(iter->distance - (iter+1)->distance) <= std::sqrt(iter->distance) * 0.08)
      const auto current_pt_azimuth =
        *reinterpret_cast<float *>(&input_ptr->data[current_idx + azimuth_offset]);
      const auto next_pt_azimuth =
        *reinterpret_cast<float *>(&input_ptr->data[next_idx + azimuth_offset]);
      float azimuth_diff = next_pt_azimuth - current_pt_azimuth;
      azimuth_diff = azimuth_diff < 0.f ? azimuth_diff + 36000.f : azimuth_diff;

      const auto current_pt_distance =
        *reinterpret_cast<float *>(&input_ptr->data[current_idx + distance_offset]);
      const auto next_pt_distance =
        *reinterpret_cast<float *>(&input_ptr->data[next_idx + distance_offset]);

      if (
        std::max(current_pt_distance, next_pt_distance) <
          std::min(current_pt_distance, next_pt_distance) * distance_ratio_ &&
        azimuth_diff < 100.f) {
        continue;
      }
      if (isCluster(input_ptr, tmp_indices)) {
        for (const auto & tmp_idx : tmp_indices) {
          output_modifier.push_back(
            std::move(*reinterpret_cast<PointXYZI *>(&input_ptr->data[tmp_idx])));
        }
      }
      tmp_indices.clear();
    }
    if (tmp_indices.empty()) {
      continue;
    }
    if (isCluster(input_ptr, tmp_indices)) {
      for (const auto & tmp_idx : tmp_indices) {
        output_modifier.push_back(
          std::move(*reinterpret_cast<PointXYZI *>(&input_ptr->data[tmp_idx])));
      }
    }
    tmp_indices.clear();
  }
  // add processing time for debug
  if (debug_publisher_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);
  }
}

rcl_interfaces::msg::SetParametersResult RingOutlierFilterComponent::paramCallback(
  const std::vector<rclcpp::Parameter> & p)
{
  std::scoped_lock lock(mutex_);

  if (get_param(p, "distance_ratio", distance_ratio_)) {
    RCLCPP_DEBUG(get_logger(), "Setting new distance ratio to: %f.", distance_ratio_);
  }
  if (get_param(p, "object_length_threshold", object_length_threshold_)) {
    RCLCPP_DEBUG(
      get_logger(), "Setting new object length threshold to: %f.", object_length_threshold_);
  }
  if (get_param(p, "num_points_threshold", num_points_threshold_)) {
    RCLCPP_DEBUG(get_logger(), "Setting new num_points_threshold to: %d.", num_points_threshold_);
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}
}  // namespace pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::RingOutlierFilterComponent)
