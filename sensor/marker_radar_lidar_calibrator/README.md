# marker_radar_lidar_calibrator

A tutorial for this calibrator can be found [here](../docs/tutorials/marker_radar_lidar_calibrator.md)

## Purpose

The package `marker_radar_lidar_calibrator` allows extrinsic calibration among the radar sensor and lidar sensor used in autonomous driving and robotics.

## Inner-workings / Algorithms

The calibration process involves five steps: constructing a background model, detecting reflectors from sensor data, matching and filtering lidar and radar detections, conducting the calibration, and showing the evaluation result.

### Step 1: Background model extraction (radar & lidar)

Once the calibrator is initiated and the user presses the `Extract background model` button, the calibrator will begin using the lidar pointcloud and radar messages to create background models. These models include voxels that indicate whether they represent the background or not.

### Step 2: Foreground extraction and reflector detection

After the background models for the lidar and radar are created, we can extract the foreground lidar and radar points from the incoming lidar pointcloud and radar messages if they do not match the background voxels.

All of the foreground radar points are defined as reflector detections. However, to classify foreground lidar points as reflector detections, it is necessary to apply a clustering algorithm, implement additional filtering, and calculate the center of the cluster.

### Step 3: Matching and filtering

Afterward, we match the lidar’s reflector detection and the radar’s reflection detection as pairs if they are closest to each other.

### Step 4: Calibration

Finally, we can use the coordinates of the reflector detection pairs to calculate the transformation based on the SVD-based estimation algorithm.

### Step 5: Evaluation

Additionally, we provide a metric plotter that can indicate whether the calibration errors have converged. Once the cross-validation errors have converged, users should be able to stop the calibration process.

### Diagram

Below, you can see how the algorithm is implemented in the `marker_radar_lidar_calibrator` package.

![marker_radar_lidar_calibrator](../docs/images/marker_radar_lidar_calibrator/marker_radar_lidar_calibrator.jpg)

## ROS Interfaces

### Input

| Name                       | Type                                                           | Description                                                                           |
| -------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `{input_lidar_pointcloud}` | `sensor_msgs::msg::PointCloud2`                                | lidar pointcloud for calibration. `input_lidar_pointcloud` is provided via parameters |
| `{input_radar_msg}`        | `radar_msgs::msg::RadarTracks` or `radar_msgs::msg::RadarScan` | radar msg for calibration, `input_radar_msg` is provided via parameters               |

### Output

| Name                          | Type                                   | Description                                              |
| ----------------------------- | -------------------------------------- | -------------------------------------------------------- |
| `lidar_background_pointcloud` | `sensor_msgs::msg::PointCloud2`        | Publishes the background pointcloud data from lidar      |
| `lidar_foreground_pointcloud` | `sensor_msgs::msg::PointCloud2`        | Publishes the foreground pointcloud data from lidar      |
| `lidar_colored_clusters`      | `sensor_msgs::msg::PointCloud2`        | Publishes colored clusters from lidar data               |
| `lidar_detection_markers`     | `visualization_msgs::msg::MarkerArray` | Publishes lidar detections                               |
| `radar_background_pointcloud` | `sensor_msgs::msg::PointCloud2`        | Publishes the background pointcloud data from radar      |
| `radar_foreground_pointcloud` | `sensor_msgs::msg::PointCloud2`        | Publishes the foreground pointcloud data from radar      |
| `radar_detection_markers`     | `visualization_msgs::msg::MarkerArray` | Publishes radar detections                               |
| `matches_markers`             | `visualization_msgs::msg::MarkerArray` | Publishes markers for matched points between sensors     |
| `tracking_markers`            | `visualization_msgs::msg::MarkerArray` | Publishes markers used for tracking calibration          |
| `text_markers`                | `visualization_msgs::msg::Marker`      | Publishes text markers that show the calibration metrics |
| `calibration_metrics`         | `std_msgs::msg::Float32MultiArray`     | Publishes calibration metrics                            |

### Services

| Name                       | Type                                                  | Description                                                                              |
| -------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `extrinsic_calibration`    | `tier4_calibration_msgs::` `srv::ExtrinsicCalibrator` | Generic calibration service. The call is blocking until the calibration process finishes |
| `extract_background_model` | `std_srvs::srv::Empty`                                | Strat to extract the background model from radar and lidar                               |
| `add_lidar_radar_pair`     | `std_srvs::srv::Empty`                                | User is able to click this buttom to add lidar-radar pair                                |
| `delete_lidar_radar_pair`  | `std_srvs::srv::Empty`                                | User is able to click this button to delete the previous lidar-radar pair                |
| `send_calibration`         | `std_srvs::srv::Empty`                                | Send the calibration result to the sensor calibration manager                            |

## Parameters

### Core Parameters

| Name                                        | Type          | Default Value                                           | Description                                                                                                          |
| ------------------------------------------- | ------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `radar_optimization_frame`                  | `std::string` | `base_link`                                             | The frame that the radar frame optimize the transformation to. (Not available yet)                                   |
| `msg_type`                                  | `std::string` | `radar tracks`/`radar scan`                             | The msg type of the input radar message. (Not available yet)                                                         |
| `transformation_type`                       | `std::string` | `yaw_only_rotation_2d` `svd_2d` `svd_3d` `roll_zero_3d` | The algorithms for optimizing the transformation between radar frame and radar optimizied frame. (Not available yet) |
| `use_lidar_initial_crop_box_filter`         | `bool`        | `True`                                                  | Enables or disables the initial cropping filter for lidar data processing.                                           |
| `lidar_initial_crop_box_min_x`              | `double`      | `-50.0`                                                 | Minimum x-coordinate for the initial lidar cropping box.                                                             |
| `lidar_initial_crop_box_min_y`              | `double`      | `-50.0`                                                 | Minimum y-coordinate for the initial lidar cropping box.                                                             |
| `lidar_initial_crop_box_min_z`              | `double`      | `-50.0`                                                 | Minimum z-coordinate for the initial lidar cropping box.                                                             |
| `lidar_initial_crop_box_max_x`              | `double`      | `50.0`                                                  | Maximum x-coordinate for the initial lidar cropping box.                                                             |
| `lidar_initial_crop_box_max_y`              | `double`      | `50.0`                                                  | Maximum y-coordinate for the initial lidar cropping box.                                                             |
| `lidar_initial_crop_box_max_z`              | `double`      | `50.0`                                                  | Maximum z-coordinate for the initial lidar cropping box.                                                             |
| `use_radar_initial_crop_box_filter`         | `bool`        | `True`                                                  | Enables or disables the initial cropping filter for radar data processing.                                           |
| `radar_initial_crop_box_min_x`              | `double`      | `-50.0`                                                 | Minimum x-coordinate for the initial radar cropping box.                                                             |
| `radar_initial_crop_box_min_y`              | `double`      | `-50.0`                                                 | Minimum y-coordinate for the initial radar cropping box.                                                             |
| `radar_initial_crop_box_min_z`              | `double`      | `-50.0`                                                 | Minimum z-coordinate for the initial radar cropping box.                                                             |
| `radar_initial_crop_box_max_x`              | `double`      | `50.0`                                                  | Maximum x-coordinate for the initial radar cropping box.                                                             |
| `radar_initial_crop_box_max_y`              | `double`      | `50.0`                                                  | Maximum y-coordinate for the initial radar cropping box.                                                             |
| `radar_initial_crop_box_max_z`              | `double`      | `50.0`                                                  | Maximum z-coordinate for the initial radar cropping box.                                                             |
| `lidar_background_model_leaf_size`          | `double`      | `0.1`                                                   | Leaf size for the lidar background model voxel grid.                                                                 |
| `radar_background_model_leaf_size`          | `double`      | `0.1`                                                   | Leaf size for the radar background model voxel grid.                                                                 |
| `max_calibration_range`                     | `double`      | `50.0`                                                  | Maximum range for calibration in meters.                                                                             |
| `background_model_timeout`                  | `double`      | `5.0`                                                   | Timeout in seconds for background model updates.                                                                     |
| `min_foreground_distance`                   | `double`      | `0.4`                                                   | Minimum distance for the foreground extraction, typically double the background model leaf size.                     |
| `background_extraction_timeout`             | `double`      | `15.0`                                                  | Timeout in seconds for background extraction processes.                                                              |
| `ransac_threshold`                          | `double`      | `0.2`                                                   | Threshold used for RANSAC inliers in meters.                                                                         |
| `ransac_max_iterations`                     | `int`         | `100`                                                   | Maximum number of RANSAC iterations for model fitting.                                                               |
| `lidar_cluster_max_tolerance`               | `double`      | `0.5`                                                   | Maximum cluster tolerance for extracting lidar cluster.                                                              |
| `lidar_cluster_min_points`                  | `int`         | `3`                                                     | Minimum number of points required to form a valid lidar cluster.                                                     |
| `lidar_cluster_max_points`                  | `int`         | `2000`                                                  | Maximum number of points allowed in a lidar cluster.                                                                 |
| `radar_cluster_max_tolerance`               | `double`      | `0.5`                                                   | Maximum cluster tolerance for extracting radar cluster.                                                              |
| `radar_cluster_min_points`                  | `int`         | `1`                                                     | Minimum number of points required to form a valid radar cluster.                                                     |
| `radar_cluster_max_points`                  | `int`         | `10`                                                    | Maximum number of points allowed in a radar cluster.                                                                 |
| `reflector_radius`                          | `double`      | `0.1`                                                   | Radius of the reflector used in calibration in meters.                                                               |
| `reflector_max_height`                      | `double`      | `1.2`                                                   | Maximum height of the reflector in meters.                                                                           |
| `max_matching_distance`                     | `double`      | `1.0`                                                   | Maximum threshold for matching distance between lidar and radar.                                                     |
| `max_initial_calibration_translation_error` | `double`      | `1.0`                                                   | Maximum allowable translation error in calibration process, if it is more than the value, WARNING will show.         |
| `max_initial_calibration_rotation_error`    | `double`      | `45.0`                                                  | Maximum allowable rotation error in calibration process, if it is more than the value, WARNING will show.            |
| `max_number_of_combination_samples`         | `int`         | `10000`                                                 | Maximum number of samples from combinations that used for cross validation during calibration process.               |

## Requirements

### radar reflector

This type of reflector is crucial for such calibrations because it has a highly predictable and consistent response to radar. The triangular shape, often composed of three metal plates arranged in a prism form, ensures that the reflector returns signals in specific, predictable ways.

<p align="center">
    <img src="../docs/images/marker_radar_lidar_calibrator/radar_reflector.png" alt="radar_reflector" width="150">
<p align="center">

## Known issues/limitations

- While performing the calibration, we have included a button to delete any mismatched pairs (e.g., an object detected by both radar and lidar). However, some outliers may not be easily detectable by human vision, leading to imperfect results as the calibration proceeds even with these anomalies present. Future enhancements will aim to improve outlier detection, thereby refining the calibration accuracy.

## Pro tips/recommendations

- While performing the calibration, try setting the radar reflector at different heights using the tripod, and also place the reflector at various distances. Please also ensure that the center of the radar reflector faces the radar sensor.