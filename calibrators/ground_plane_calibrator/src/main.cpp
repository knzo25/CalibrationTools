// Copyright 2024 TIER IV, Inc.
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

#include <ground_plane_calibrator/ground_plane_calibrator.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <thread>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::executors::MultiThreadedExecutor executor;
  rclcpp::NodeOptions node_options;
  std::shared_ptr<ground_plane_calibrator::ExtrinsicGroundPlaneCalibrator> node =
    std::make_shared<ground_plane_calibrator::ExtrinsicGroundPlaneCalibrator>(node_options);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();

  return 0;
}
