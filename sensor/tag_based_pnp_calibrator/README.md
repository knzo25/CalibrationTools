# tag_based_pnp_calibrator

A tutorial for this calibrator can be found [here](../docs/tutorials/tag_based_pnp_calibrator.md)

## Purpose

The package `tag_based_pnp_calibrator` allows extrinsic calibration among the camera and lidar sensors used in autonomous driving and robotics.

## Inner-workings / Algorithms

The `tag_based_pnp_calibrator` utilizes the PnP (Perspective-n-Point) algorithm, a computer vision method that finds the best match between a set of 3D points (from the lidar) and their corresponding 2D projections (in the camera images), to accurately calculate the transformation between the camera and lidar. To run this package, you also need to execute the `apriltag_ros` and the `lidartag` packages to calculate the transformation.

The `apriltag_ros` package detects AprilTag markers from an image and outputs the detection results. Conversely, the `lidartag` package detects LidarTag markers and outputs its detection results.

The `tag_based_pnp_calibrator` utilizes the detections from both `apriltag_ros` package and `lidartag` package, employing a Kalman Filter to track these detections. If the detections converge, the calibrator applies the SQPnP algorithm provided by OpenCV to estimate the transformation between the image points from AprilTag and the object points from LidarTag.

### Diagram

Below, you can see how the algorithm is implemented in the `tag_based_pnp_calibrator` package.

![segment](../docs/images/tag_based_pnp_calibrator/tag_based_pnp_calibrator.jpg)

## ROS Interfaces

### Input

| Name                        | Type                                         | Description                                                                                 |
| --------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `{camera_info}`             | `sensor_msgs::msg::CameraInfo`               | Intrinsic parameters for the calibration cameras . `camera_info` is provided via parameters |
| `lidartag/detections_array` | `lidartag_msgs::msg::LidarTagDetectionArray` | LidarTag detections. `lidartag/detections_array` is defined in launcher.                    |
| `apriltag/detection_array`  | `apriltag_msgs::msg::AprilTagDetectionArray` | AprilTag detections. `apriltag/detection_array` is defined in launcher.                     |

### Output

| Name                   | Type                                             | Description                                         |
| ---------------------- | ------------------------------------------------ | --------------------------------------------------- |
| `filtered_projections` | `visualization_msgs::msg::MarkerArray`           | Publishes the calibration markers for visualization |
| `calibration_points`   | `tier4_calibration_msgs::msg::CalibrationPoints` | Publishes the tag points after calibration          |

### Services

| Name                    | Type                                                  | Description                                                                              |
| ----------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `extrinsic_calibration` | `tier4_calibration_msgs::` `srv::ExtrinsicCalibrator` | Generic calibration service. The call is blocking until the calibration process finishes |

## Parameters

### Core Parameters

| Name                                          | Type                  | Default Value                    | Description                                                                                                                                               |
| --------------------------------------------- | --------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `calib_rate`                                  | `double`              | `10.0`                           | The rate at which the calibration callback is invoked. This controls the frequency of calibration updates.                                                |
| `base_frame`                                  | `std::string`         | `base_link`                      | The base_frame is used for visualization.                                                                                                                 |
| `min_tag_size`                                | `double`              | `0.6`                            | The size of the AprilTag in meters (payload).                                                                                                             |
| `max_tag_distance`                            | `double`              | `20.0`                           | Maximum allowed distance in meter from the camera to the tags.                                                                                            |
| `max_allowed_homography_error`                | `double`              | `0.5`                            | AprilTag detection are discarded if the homography error is larger than `max_allowed_homography_error`.                                                   |
| `use_receive_time`                            | `bool`                | `false`                          | Flag to determine whether to use the receive time instead of the header timestamps.                                                                       |
| `use_rectified_image`                         | `bool`                | `true`                           | Flag to determine whether rectified images should be used in the calibration process.                                                                     |
| `calibration_crossvalidation_training_ratio`  | `double`              | `0.7`                            | The ratio of data used for training versus testing during the calibration's cross-validation process.                                                     |
| `calibration_convergence_min_pairs`           | `int`                 | `9`                              | The minimum number of AprilTag and LidarTag detection pairs required to consider the calibration process as potentially converged.                        |
| `calibration_convergence_min_area_percentage` | `double`              | `0.005`                          | Minimum percentage of the area that needs to be covered by detection.                                                                                     |
| `min_pnp_points`                              | `int`                 | `8`                              | Minimum number of points required for the Perspective-n-Point problem used in calibration to solve the pose estimation.                                   |
| `min_convergence_time`                        | `double`              | `6.0`                            | Minimum time required for the calibration process to be considered as converged.                                                                          |
| `max_no_observation_time`                     | `double`              | `3.0`                            | If the time between new detection and the last observed detection are larger than `max_no_observation_time`, remove the detection from active hypotheses. |
| `new_hypothesis_distance`                     | `double`              | `1.5`                            | Distance threshold for creating a new hypothesis.                                                                                                         |
| `tag_ids`                                     | `std::vector<int>`    | `[0, 1, 2, 3, 4, 5]`             | List of tag IDs that are used in the calibration process.                                                                                                 |
| `tag_sizes`                                   | `std::vector<double>` | `[0.6, 0.6, 0.6, 0.6, 0.6, 0.6]` | Payload sizes of the tags corresponding to the IDs in meter.                                                                                              |
| `lidartag_max_convergence_translation`        | `double`              | `0.05`                           | Maximum translation in meter allowed for a lidar tag detection hypothesis to be considered as converged.                                                  |
| `lidartag_max_convergence_translation_dot`    | `double`              | `0.03`                           | Maximum translation velocity in meter/second in allowed for a lidar tag detection hypothesis to be considered as converged.                               |
| `lidartag_max_convergence_rotation`           | `double`              | `3.0`                            | Maximum rotation in degree allowed for a lidar tag detection hypothesis to be considered as converged.                                                    |
| `lidartag_max_convergence_rotation_dot`       | `double`              | `2.5`                            | Maximum rotation velocity in degree/second allowed for a lidar tag detection hypothesis to be considered as converged.                                    |
| `lidartag_new_hypothesis_translation`         | `double`              | `0.1`                            | Translation threshold in meter for generating a new hypothesis in lidar tag tracking.                                                                     |
| `lidartag_new_hypothesis_rotation`            | `double`              | `15.0`                           | Rotation threshold in degree for generating a new hypothesis in lidar tag tracking.                                                                       |
| `lidartag_measurement_noise_translation`      | `double`              | `0.05`                           | The square of this value (meter) is part of input for Kalman Filter to measurement noise covariance matrix (R).                                           |
| `lidartag_measurement_noise_rotation`         | `double`              | `5.0`                            | The square of this value (degree) is part of input for measurementNoiseCov to measurement noise covariance matrix (R).                                    |
| `lidartag_process_noise_translation`          | `double`              | `0.01`                           | The square of this value (meter) is part of input for Kalman Filter to process noise covariance matrix (Q).                                               |
| `lidartag_process_noise_translation_dot`      | `double`              | `0.001`                          | The square of this value (meter/second) is part of input for Kalman Filter to process noise covariance matrix (Q).                                        |
| `lidartag_process_noise_rotation`             | `double`              | `1.0`                            | The square of this value (degree) is part of input for Kalman Filter to process noise covariance matrix (Q).                                              |
| `lidartag_process_noise_rotation_dot`         | `double`              | `0.1`                            | The square of this value (degree/second) is part of input for Kalman Filter to process noise covariance matrix (Q).                                       |
| `apriltag_max_convergence_translation`        | `double`              | `2.0`                            | Maximum translation error in centimeter allowed for an AprilTag detection hypothesis to be considered as converged.                                       |
| `apriltag_new_hypothesis_translation`         | `double`              | `20.0`                           | Translation threshold in centimeter for generating a new hypothesis in AprilTag tracking.                                                                 |
| `apriltag_measurement_noise_translation`      | `double`              | `0.2`                            | The square of this value (meter) is part of input for Kalman Filter to measurement noise covariance matrix (R).                                           |
| `apriltag_process_noise_translation`          | `double`              | `0.02`                           | The square of this value (meter) is part of input for Kalman Filter to process noise covariance matrix (Q).                                               |

## Requirements

### LiDARTag

To perform camera-lidar calibration using this tool, it is necessary to prepare LidarTags and lidars with intensity measures. To ensure that no objects obstruct the tag detection and to achieve the most stable detection possible, it is highly recommended to also prepare fixed mounts for these tags, as shown below.

Note that the ones we used are Lidartags of size 0.8 meters. Meaning the Apriltag payload is 0.6 meters. We have also tried with Lidartags of sizes 0.6 meters, but to use them the user needs to set several parameters by himself.

<p align="center">
    <img src="../docs/images/tag_based_pnp_calibrator/lidartag-mount.jpg"  alt="lidartag-mount" width="500">
</p>

## References

References/External links
[1] Jiunn-Kai (Bruce) Huang, Shoutian Wang, Maani Ghaffari, and Jessy W. Grizzle, "LiDARTag: A Real-Time Fiducial Tag System for Point Clouds," in IEEE Robotics and Automation Letters. Volume: 6, Issue: 3, July 2021.

## Known issues/limitations

- The tool uses a basic OpenCV camera model for calibration (plumb bomb).

## Pro tips/recommendations

- During calibration, ensure that the lidar scan covers the tag, similar to the first example shown in the image below. However, if the tag resolution is low, as in the second example, and the lidar still detects the tag, it may still be acceptable, but avoidable when possible. The third example demonstrates a scenario where the lidar scan fails to cover the tag, resulting in the inability to detect the LidarTag.

<p align="center">
    <img src="../docs/images/tag_based_pnp_calibrator/lidarscan_on_tag.jpg"  alt="lidarscan_on_tag" width="500">
</p>

- It is highly recommended to place the tag perpendicular to the calibration lidar as shown in the following image:

<p align="center">
    <img src="../docs/images/tag_based_pnp_calibrator/tag_position.jpg"  alt="tag_position" width="500">
</p>