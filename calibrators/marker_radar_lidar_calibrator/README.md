# marker_radar_lidar_calibrator

A tutorial for this calibrator can be found [here](../docs/tutorials/marker_radar_lidar_calibrator.md)

## Purpose

The package `marker_radar_lidar_calibrator` allows extrinsic calibration between radar and 3d lidar sensors used in autonomous driving and robotics.

Currently, the calibrator only supports radars that include distance and azimuth angle, but without elevation angle. For example, ARS 408 radar can be calibrated with this tool. Also, note that the 3d lidar should have a resolution that is high enough to scan several points on the [radar reflector](#radar-reflector) (calibration target).

## Inner-workings / Algorithms

The calibrator is designed to estimate the transformation between radar and lidar sensors. It starts by pinpointing the central points of reflectors within lidar pointclouds and radar objects, then aligns these points for precise matching. An SVD-based and a yaw-only rotation estimation algorithm are applied to these correlated points to determine the transformation.

Specifically, the calibration process consists of four primary steps: constructing a background model, extracting the foreground to detect reflectors, matching and filtering lidar and radar detections, and finally executing the calibration.

### Step 1: Background model construction

Firstly, given the challenge of reliably detecting reflectors, such as when radar outputs numerous objects making it difficult to identify the reflector among them, background models for both lidar and radar are constructed from the lidar pointclouds and radar objects within a user-defined calibration area, which lacks any calibration targets (such as radar reflectors). More specifically, these background models consist of uniform binary voxel grids that denote whether each voxel represents the background.

### Step 2: Foreground extraction and reflector detection

After the background models for the lidar and radar are established, we extract the foreground points from incoming lidar pointclouds and radar objects that are not in the background voxels. All foreground radar points are automatically categorized as potential reflector detections.

For foreground lidar points, however, the [reflector](#radar-reflector) detection process involves more steps. We first apply a clustering algorithm, then find the highest point in each cluster, and filter out the cluster if the highest point is larger than `reflector_max_height`. Next, we average all points within a `reflector_radius` from the highest point to estimate the center point of the reflector.

### Step 3: Matching and filtering

Since it is not possible to directly differentiate reflector detections, whether there are multiple targets in the calibration area or if the detections are from humans or radar reflectors, we rely on the initial calibration to pair each lidar detection with its closest radar detection, and vice versa. A detection pair is accepted if they are mutually their the closest matches. Once a match is made, it is evaluated against existing hypotheses (monitored by a Kalman filter): if it aligns with an existing hypothesis, that hypothesis is updated; if it does not align with any, a new hypothesis is created. When a hypothesis achieves convergence, it is finalized and added to the calibration list.

### Step 4: Calibration

After matching detection pairs from the sensors, we can compute the transformation between them using estimation algorithms. Currently, we support two methods: a 2d SVD-based approach and a yaw-only rotation approach.

For the 2d SVD-based method, since radar detections lack a Z component, we convert the problem to 2d by setting the Z component of lidar detections to zero. We then estimate the transformation using the SVD method provided by PCL.

The yaw-only rotation method, on the other hand, calculates the average yaw angle difference of all pairs and estimates the transformation, considering only rotation, between the sensors. Generally, the 2d calibration is preferred when valid; otherwise, the yaw rotation is used as the calibration output.

It's also important to note that in the near future, the calibrator will be updated to support radar that includes elevation angle and provides different transformation algorithms.

### Diagram

Below, you can see how the algorithm is implemented in the `marker_radar_lidar_calibrator` package.

![marker_radar_lidar_calibrator](../docs/images/marker_radar_lidar_calibrator/marker_radar_lidar_calibrator.jpg)

## ROS Interfaces

### Input

| Name                     | Type                            | Description                                |
| ------------------------ | ------------------------------- | ------------------------------------------ |
| `input_lidar_pointcloud` | `sensor_msgs::msg::PointCloud2` | Topic of lidar pointcloud for calibration. |
| `input_radar_msg`        | `radar_msgs::msg::RadarTracks`  | Topic of radar objects for calibration.    |

### Output

| Name                          | Type                                   | Description                                               |
| ----------------------------- | -------------------------------------- | --------------------------------------------------------- |
| `lidar_background_pointcloud` | `sensor_msgs::msg::PointCloud2`        | Publishes the background pointcloud from lidar.           |
| `lidar_foreground_pointcloud` | `sensor_msgs::msg::PointCloud2`        | Publishes the foreground pointcloud from lidar.           |
| `lidar_colored_clusters`      | `sensor_msgs::msg::PointCloud2`        | Publishes colored pointcloud clusters from lidar.         |
| `lidar_detection_markers`     | `visualization_msgs::msg::MarkerArray` | Publishes lidar detections.                               |
| `radar_background_pointcloud` | `sensor_msgs::msg::PointCloud2`        | Publishes the background pointcloud from radar.           |
| `radar_foreground_pointcloud` | `sensor_msgs::msg::PointCloud2`        | Publishes the foreground pointcloud from radar.           |
| `radar_detection_markers`     | `visualization_msgs::msg::MarkerArray` | Publishes radar detections.                               |
| `matches_markers`             | `visualization_msgs::msg::MarkerArray` | Publishes matched lidar and radar detection.              |
| `tracking_markers`            | `visualization_msgs::msg::MarkerArray` | Publishes tracks of reflectors.                           |
| `text_markers`                | `visualization_msgs::msg::Marker`      | Publishes text markers that show the calibration metrics. |
| `calibration_metrics`         | `std_msgs::msg::Float32MultiArray`     | Publishes calibration metrics.                            |

### Services

| Name                       | Type                                                  | Description                                                                              |
| -------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `extrinsic_calibration`    | `tier4_calibration_msgs::` `srv::ExtrinsicCalibrator` | Generic calibration service. The call is blocked until the calibration process finishes. |
| `extract_background_model` | `std_srvs::srv::Empty`                                | Start to extract the background model from radar and lidar data.                         |
| `add_lidar_radar_pair`     | `std_srvs::srv::Empty`                                | Add lidar-radar pairs for calibration.                                                   |
| `delete_lidar_radar_pair`  | `std_srvs::srv::Empty`                                | Delete the previous lidar-radar pair.                                                    |
| `send_calibration`         | `std_srvs::srv::Empty`                                | Send the calibration result to the sensor calibration manager.                           |

## Parameters

### Core Parameters

| Name                                        | Type          | Default Value                                           | Description                                                                                                                                                        |
| ------------------------------------------- | ------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `radar_parallel_frame`                      | `std::string` | `base_link`                                             | The frame that the radar frame optimizes the transformation to.                                                                                                    |
| `msg_type`                                  | `std::string` | `radar tracks`/`radar scan`                             | The type of the input radar objects. (Not available yet, currently only support radar tracks)                                                                      |
| `transformation_type`                       | `std::string` | `yaw_only_rotation_2d` `svd_2d` `svd_3d` `roll_zero_3d` | The algorithms for optimizing the transformation between radar frame and radar parallel frame. (Not available yet)                                                 |
| `use_lidar_initial_crop_box_filter`         | `bool`        | `true`                                                  | Enables or disables the initial cropping filter for lidar data processing.                                                                                         |
| `lidar_initial_crop_box_min_x`              | `double`      | `-50.0`                                                 | Minimum x-coordinate in meters for the initial lidar calibration area.                                                                                             |
| `lidar_initial_crop_box_min_y`              | `double`      | `-50.0`                                                 | Minimum y-coordinate in meters for the initial lidar calibration area.                                                                                             |
| `lidar_initial_crop_box_min_z`              | `double`      | `-50.0`                                                 | Minimum z-coordinate in meters for the initial lidar calibration area.                                                                                             |
| `lidar_initial_crop_box_max_x`              | `double`      | `50.0`                                                  | Maximum x-coordinate in meters for the initial lidar calibration area.                                                                                             |
| `lidar_initial_crop_box_max_y`              | `double`      | `50.0`                                                  | Maximum y-coordinate in meters for the initial lidar calibration area.                                                                                             |
| `lidar_initial_crop_box_max_z`              | `double`      | `50.0`                                                  | Maximum z-coordinate in meters for the initial lidar calibration area.                                                                                             |
| `use_radar_initial_crop_box_filter`         | `bool`        | `true`                                                  | Enables or disables the initial cropping filter for radar data processing.                                                                                         |
| `radar_initial_crop_box_min_x`              | `double`      | `-50.0`                                                 | Minimum x-coordinate in meters for the initial radar calibration area.                                                                                             |
| `radar_initial_crop_box_min_y`              | `double`      | `-50.0`                                                 | Minimum y-coordinate in meters for the initial radar calibration area.                                                                                             |
| `radar_initial_crop_box_min_z`              | `double`      | `-50.0`                                                 | Minimum z-coordinate in meters for the initial radar calibration area.                                                                                             |
| `radar_initial_crop_box_max_x`              | `double`      | `50.0`                                                  | Maximum x-coordinate in meters for the initial radar calibration area.                                                                                             |
| `radar_initial_crop_box_max_y`              | `double`      | `50.0`                                                  | Maximum y-coordinate in meters for the initial radar calibration area.                                                                                             |
| `radar_initial_crop_box_max_z`              | `double`      | `50.0`                                                  | Maximum z-coordinate in meters for the initial radar calibration area.                                                                                             |
| `lidar_background_model_leaf_size`          | `double`      | `0.1`                                                   | Voxel size in meters for the lidar background model.                                                                                                               |
| `radar_background_model_leaf_size`          | `double`      | `0.1`                                                   | Voxel size in meters for the radar background model.                                                                                                               |
| `max_calibration_range`                     | `double`      | `50.0`                                                  | Maximum range for calibration in meters.                                                                                                                           |
| `background_model_timeout`                  | `double`      | `5.0`                                                   | The background model will terminate if there are no new points in the background within this period, measured in seconds.                                          |
| `min_foreground_distance`                   | `double`      | `0.4`                                                   | Minimum distance in meters for extracting foreground points.                                                                                                       |
| `background_extraction_timeout`             | `double`      | `15.0`                                                  | Timeout in seconds for background extraction processes.                                                                                                            |
| `ransac_threshold`                          | `double`      | `0.2`                                                   | Distance threshold for the segmentation model.                                                                                                                     |
| `ransac_max_iterations`                     | `int`         | `100`                                                   | Maximum number of iterations for the segmentation model.                                                                                                           |
| `lidar_cluster_max_tolerance`               | `double`      | `0.5`                                                   | Maximum cluster tolerance for extracting lidar cluster.                                                                                                            |
| `lidar_cluster_min_points`                  | `int`         | `3`                                                     | Minimum number of points required to form a valid lidar cluster.                                                                                                   |
| `lidar_cluster_max_points`                  | `int`         | `2000`                                                  | Maximum number of points allowed in a lidar cluster.                                                                                                               |
| `radar_cluster_max_tolerance`               | `double`      | `0.5`                                                   | Maximum cluster tolerance for extracting radar cluster.                                                                                                            |
| `radar_cluster_min_points`                  | `int`         | `1`                                                     | Minimum number of points required to form a valid radar cluster.                                                                                                   |
| `radar_cluster_max_points`                  | `int`         | `10`                                                    | Maximum number of points allowed in a radar cluster.                                                                                                               |
| `reflector_radius`                          | `double`      | `0.1`                                                   | Radius of the reflector in meters.                                                                                                                                 |
| `reflector_max_height`                      | `double`      | `1.2`                                                   | Maximum height in meters of the reflector in meters.                                                                                                               |
| `max_matching_distance`                     | `double`      | `1.0`                                                   | Maximum distance threshold in meters for matching lidar and radar.                                                                                                 |
| `max_initial_calibration_translation_error` | `double`      | `1.0`                                                   | Maximum allowable translation error in meters in the calibration process. If this error exceeds the specified value, a warning message will appear in the console. |
| `max_initial_calibration_rotation_error`    | `double`      | `45.0`                                                  | Maximum allowable rotation error in degrees in the calibration process. If this error exceeds the specified value, a warning message will appear in the console.   |
| `max_number_of_combination_samples`         | `int`         | `10000`                                                 | Maximum number of samples from combinations that are used for cross-validation during the calibration process.                                                     |

## Requirements

### Radar reflector

The type of reflector shown in the image below is crucial for our calibration because it has a highly predictable and consistent response to radar waves. The triangular shape, often composed of three metal plates arranged in a prism form, ensures that the reflector returns signals in specific, predictable ways.

It is recommended that the user build the radar reflector on a tripod, securing it with tape to ensure stability. Additionally, nothing should be attached above the radar reflector; it must be the highest object on the entire calibration target. Additionally, make sure the height of the radar reflector is not larger than the `reflector_max_height` parameter.

<p align="center">
    <img src="../docs/images/marker_radar_lidar_calibrator/radar_reflector.png" alt="radar_reflector" width="150">
<p align="center">

## Known issues/limitations

- While extracting the background model, ensure that the radar reflectors are not in the calibration area and that no one is moving around the calibration area.

- While performing the calibration, the calibrator provides a button to delete any mismatched pairs (e.g., an object detected by both radar and lidar). However, some outliers may not be easily detectable by human eyes, leading to imperfect results as the calibration proceeds even with these anomalies present. Future enhancements will aim to improve outlier detection, thereby refining the calibration accuracy.

- The calibrator should be able to handle different lidar and radar sensors. So far, We calibrated Velodyne VLS-128 lidar sensor, Pandar-40P lidar sensor, and ARS408 radar sensor with good calibration results.

## Pro tips/recommendations

- While performing the calibration, it is required that all radar reflectors and the radar itself lie on the same plane. The lidar does not have the same limitation.
- During calibration, place the reflectors at various distances and ensure that the center of the radar reflector faces the radar sensor.
- The calibration area is recommended to setup like the image below.

<p align="center">
    <img src="../docs/images/marker_radar_lidar_calibrator/marker_radar_lidar_vis.svg" alt="radar_reflector" width="500">
<p align="center">