from new_extrinsic_calibration_manager.calibrator_base import CalibratorBase
from new_extrinsic_calibration_manager.calibrator_registry import CalibratorRegistry
from new_extrinsic_calibration_manager.ros_interface import RosInterface
from new_extrinsic_calibration_manager.types import FramePair


@CalibratorRegistry.register_calibrator(
    project_name="dummy_project", calibrator_name="base_lidar_calibration"
)
class DummyBaseLidarCalibrator(CalibratorBase):
    required_frames = ["base_link, sensor_kit_base_link", "velodyne_top_base_link", "velodyne_top"]

    def __init__(self, ros_interface: RosInterface):
        super().__init__(ros_interface)

        print("DummyBaseLidarCalibrator")

        self.add_calibrator(
            service_name="calibrate_base_lidar",
            expected_calibration_frames=[FramePair(parent="base_link", child="velodyne_top")],
        )