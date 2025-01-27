#!/usr/bin/env python3

# Copyright 2024 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import numpy as np

from sensor_calibration_manager.calibrator_base import CalibratorBase
from sensor_calibration_manager.calibrator_registry import CalibratorRegistry
from sensor_calibration_manager.ros_interface import RosInterface
from sensor_calibration_manager.types import FramePair


@CalibratorRegistry.register_calibrator(
    project_name="x2", calibrator_name="tag_based_sfm_base_lidars_calibrator"
)
class TagBasedSfmBaseLidarsCalibrator(CalibratorBase):
    required_frames = []

    def __init__(self, ros_interface: RosInterface, **kwargs):
        super().__init__(ros_interface)

        self.base_frame = kwargs["base_frame"]
        self.top_unit_frame = "top_unit_base_link"
        self.front_unit_frame = "front_unit_base_link"
        self.rear_unit_frame = "rear_unit_base_link"

        self.main_sensor_frame = kwargs["main_calibration_sensor_frame"]
        self.calibration_lidar_frames = [
            kwargs["calibration_lidar_1_frame"],
            kwargs["calibration_lidar_2_frame"],
            kwargs["calibration_lidar_3_frame"],
        ]
        self.calibration_lidar_base_frames = [
            lidar_frame + "_base_link" for lidar_frame in self.calibration_lidar_frames
        ]

        self.required_frames.extend(
            [
                self.base_frame,
                self.top_unit_frame,
                self.front_unit_frame,
                self.rear_unit_frame,
                self.main_sensor_frame,
                *self.calibration_lidar_frames,
                *self.calibration_lidar_base_frames,
            ]
        )

        self.add_calibrator(
            service_name="calibrate_base_lidars",
            expected_calibration_frames=[
                FramePair(parent=self.main_sensor_frame, child=self.base_frame),
                *[
                    FramePair(parent=self.main_sensor_frame, child=calibration_frame)
                    for calibration_frame in self.calibration_lidar_frames
                ],
            ],
        )

    def post_process(self, calibration_transforms: Dict[str, Dict[str, np.array]]):
        main_sensor_to_base_transform = calibration_transforms[self.main_sensor_frame][
            self.base_frame
        ]

        top_kit_to_main_lidar_transform = self.get_transform_matrix(
            self.top_unit_frame, self.main_sensor_frame
        )

        front_kit_to_front_lower_lidar_transform = self.get_transform_matrix(
            self.front_unit_frame, "pandar_40p_front"
        )

        rear_kit_to_rear_lower_lidar_transform = self.get_transform_matrix(
            self.rear_unit_frame, "pandar_40p_rear"
        )

        base_to_top_kit_transform = np.linalg.inv(
            top_kit_to_main_lidar_transform @ main_sensor_to_base_transform
        )

        base_to_front_kit_transform = (
            np.linalg.inv(main_sensor_to_base_transform)
            @ calibration_transforms[self.main_sensor_frame]["pandar_40p_front"]
            @ np.linalg.inv(front_kit_to_front_lower_lidar_transform)
        )
        base_to_rear_kit_transform = (
            np.linalg.inv(main_sensor_to_base_transform)
            @ calibration_transforms[self.main_sensor_frame]["pandar_40p_rear"]
            @ np.linalg.inv(rear_kit_to_rear_lower_lidar_transform)
        )

        results = {self.base_frame: {}}
        results[self.base_frame][self.top_unit_frame] = base_to_top_kit_transform
        results[self.base_frame][self.front_unit_frame] = base_to_front_kit_transform
        results[self.base_frame][self.rear_unit_frame] = base_to_rear_kit_transform

        return results
