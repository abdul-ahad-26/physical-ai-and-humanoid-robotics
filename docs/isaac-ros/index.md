---
title: Isaac ROS (Perception, Navigation)
sidebar_position: 9.2
description: NVIDIA Isaac ROS packages for perception, navigation, and robotics applications
---

# Isaac ROS (Perception, Navigation)

## Learning Objectives

- Understand the Isaac ROS package ecosystem and architecture
- Install and configure Isaac ROS packages for robotics applications
- Implement perception pipelines using Isaac ROS packages
- Configure navigation systems with Isaac ROS components
- Integrate Isaac ROS with existing ROS 2 systems
- Deploy Isaac ROS in real-world robotics scenarios
- Troubleshoot common Isaac ROS issues

## Introduction to Isaac ROS

Isaac ROS is NVIDIA's collection of GPU-accelerated packages for ROS 2 that provide high-performance implementations of common robotics perception, navigation, and manipulation algorithms. Built on NVIDIA's AI and graphics technologies, Isaac ROS packages deliver significant performance improvements over traditional CPU-based implementations.

### Key Isaac ROS Capabilities

- **GPU-Accelerated Perception**: Computer vision algorithms optimized for NVIDIA GPUs
- **Real-time SLAM**: Visual-inertial and visual-odometry systems
- **Sensor Processing**: Optimized processing for cameras, LiDAR, and IMU sensors
- **AI Integration**: Direct integration with NVIDIA's AI frameworks
- **Simulation Integration**: Seamless connection with Isaac Sim
- **Hardware Acceleration**: Leverage TensorRT, CUDA, and cuDNN for performance

### Isaac ROS vs Traditional ROS Packages

| Aspect | Traditional ROS | Isaac ROS |
|--------|----------------|-----------|
| Performance | CPU-based | GPU-accelerated |
| Algorithms | Generic implementations | NVIDIA-optimized |
| Dependencies | Standard libraries | NVIDIA libraries |
| Hardware | Any | NVIDIA GPU required |
| Performance | Standard | 10x+ faster for many tasks |

## Isaac ROS Package Overview

### Core Perception Packages

#### Isaac ROS AprilTag
Detects AprilTag fiducial markers for precise pose estimation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header


class AprilTagNode(Node):
    """
    Isaac ROS AprilTag node implementation
    """

    def __init__(self):
        super().__init__('isaac_ros_apriltag_node')

        # Subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect',
            self.image_callback,
            10
        )

        # Subscriber for camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/apriltag_detections',
            10
        )

        # Publisher for poses
        self.pose_pub = self.create_publisher(
            PoseArray,
            '/apriltag_poses',
            10
        )

        # AprilTag parameters
        self.tag_family = 'tag36h11'
        self.tag_size = 0.166  # meters
        self.max_tags = 30

        # Camera info storage
        self.camera_info = None

        self.get_logger().info('Isaac ROS AprilTag node initialized')

    def camera_info_callback(self, msg):
        """
        Store camera calibration information
        """
        self.camera_info = msg

    def image_callback(self, msg):
        """
        Process incoming camera image for AprilTag detection
        """
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera info...')
            return

        # In a real Isaac ROS implementation, this would interface with the
        # Isaac ROS AprilTag detection pipeline which leverages GPU acceleration
        # For this example, we'll create mock detections
        detections = self.process_apriltag_detection(msg)

        if detections:
            # Publish detections
            detection_msg = Detection2DArray()
            detection_msg.header = msg.header
            detection_msg.detections = detections
            self.detection_pub.publish(detection_msg)

            # Extract and publish poses
            pose_array = self.extract_poses(detections)
            pose_array.header = msg.header
            self.pose_pub.publish(pose_array)

    def process_apriltag_detection(self, image_msg):
        """
        Process AprilTag detection (mock implementation showing Isaac ROS concepts)
        """
        # In a real Isaac ROS AprilTag implementation:
        # 1. Image is processed using GPU-accelerated AprilTag detection
        # 2. Tag poses are estimated using calibrated camera parameters
        # 3. Results include tag IDs, corners, and 3D poses

        # This is a simplified mock implementation
        # Real implementation would use Isaac ROS AprilTag detection
        return []

    def extract_poses(self, detections):
        """
        Extract poses from detections
        """
        pose_array = PoseArray()

        for detection in detections:
            # Extract pose from detection results
            # In Isaac ROS, this would come from the GPU-accelerated pose estimation
            if detection.results:
                pose = detection.results[0].pose.pose
                pose_array.poses.append(pose)

        return pose_array
```

#### Isaac ROS Visual SLAM
Provides real-time visual SLAM capabilities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


class VisualSlamNode(Node):
    """
    Isaac ROS Visual SLAM node
    """

    def __init__(self):
        super().__init__('isaac_ros_vslam_node')

        # Subscribers for stereo camera or RGB-D
        self.left_image_sub = self.create_subscription(
            Image,
            '/camera/left/image_rect',
            self.left_image_callback,
            10
        )

        self.right_image_sub = self.create_subscription(
            Image,
            '/camera/right/image_rect',
            self.right_image_callback,
            10
        )

        # Optional IMU for improved tracking
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publishers for pose and map
        self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)

        # VSLAM parameters
        self.track_features = True
        self.map_building = True
        self.use_imu = True

        # Internal state
        self.latest_left_image = None
        self.latest_right_image = None
        self.current_pose = None

        self.get_logger().info('Isaac ROS Visual SLAM node initialized')

    def left_image_callback(self, msg):
        """
        Process left camera image for stereo VSLAM
        """
        self.latest_left_image = msg
        self.process_stereo_images()

    def right_image_callback(self, msg):
        """
        Process right camera image for stereo VSLAM
        """
        self.latest_right_image = msg
        self.process_stereo_images()

    def imu_callback(self, msg):
        """
        Process IMU data for improved tracking
        """
        # In a real Isaac ROS VSLAM implementation, IMU data would be fused
        # with visual features for more robust tracking
        pass

    def process_stereo_images(self):
        """
        Process stereo images for VSLAM
        """
        if self.latest_left_image is None or self.latest_right_image is None:
            return

        # In a real Isaac ROS VSLAM implementation:
        # 1. Feature detection and matching using GPU acceleration
        # 2. Dense stereo reconstruction
        # 3. Visual-inertial fusion
        # 4. Loop closure and map optimization

        # Mock implementation showing the concept
        current_pose = self.compute_visual_odometry(
            self.latest_left_image,
            self.latest_right_image
        )

        if current_pose:
            # Publish odometry
            odom_msg = Odometry()
            odom_msg.header.stamp = self.latest_left_image.header.stamp
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_link'
            odom_msg.pose.pose = current_pose
            self.odom_pub.publish(odom_msg)

            # Publish pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.latest_left_image.header.stamp
            pose_msg.header.frame_id = 'odom'
            pose_msg.pose = current_pose
            self.pose_pub.publish(pose_msg)

            # Update internal state
            self.current_pose = current_pose

    def compute_visual_odometry(self, left_image, right_image):
        """
        Compute visual odometry from stereo images (mock implementation)
        """
        # This would use Isaac ROS's GPU-accelerated VSLAM algorithms
        # to compute pose relative to initial position
        return None  # Placeholder
```

### Navigation Packages

#### Isaac ROS Navigation
GPU-accelerated navigation stack:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Bool


class IsaacNavigationNode(Node):
    """
    Isaac ROS Navigation node with GPU acceleration
    """

    def __init__(self):
        super().__init__('isaac_ros_navigation_node')

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.global_plan_pub = self.create_publisher(Path, '/global_plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.feedback_pub = self.create_publisher(Bool, '/navigation_feedback', 10)

        # Isaac ROS navigation parameters
        self.planner_frequency = 5.0  # Hz
        self.controller_frequency = 20.0  # Hz
        self.use_gpu_path_planning = True
        self.gpu_path_optimizer_enabled = True

        # Navigation state
        self.current_goal = None
        self.current_map = None
        self.navigation_active = False
        self.global_plan = None
        self.local_plan = None

        self.get_logger().info('Isaac ROS Navigation node initialized')

    def goal_callback(self, msg):
        """
        Handle navigation goal with Isaac ROS GPU-accelerated planning
        """
        self.current_goal = msg
        self.get_logger().info(f'New goal received: [{msg.pose.position.x}, {msg.pose.position.y}]')

        if self.current_map:
            self.start_navigation()
        else:
            self.get_logger().warn('No map available, waiting for map...')

    def laser_callback(self, msg):
        """
        Process laser scan for local obstacle avoidance
        """
        if self.navigation_active:
            # Use Isaac ROS GPU-accelerated obstacle detection
            self.update_local_plan(msg)

    def map_callback(self, msg):
        """
        Process occupancy grid map
        """
        self.current_map = msg
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height} resolution: {msg.info.resolution}')

        # If we have a goal and now have a map, start navigation
        if self.current_goal:
            self.start_navigation()

    def costmap_callback(self, msg):
        """
        Process costmap updates
        """
        # In Isaac ROS, this would interface with GPU-accelerated costmap processing
        pass

    def start_navigation(self):
        """
        Start navigation to goal using Isaac ROS planners
        """
        if not self.current_goal or not self.current_map:
            return

        # Plan global path using Isaac ROS GPU-accelerated planner
        global_plan = self.plan_global_path_gpu(self.current_map, self.current_goal)

        if global_plan:
            self.global_plan = global_plan
            self.global_plan_pub.publish(global_plan)

            # Start local planning and control
            self.navigation_active = True
            self.execute_navigation()
        else:
            self.get_logger().error('Failed to plan global path')

    def plan_global_path_gpu(self, map_msg, goal_msg):
        """
        Plan global path using Isaac ROS GPU-accelerated path planner
        """
        # In a real Isaac ROS implementation, this would use GPU-accelerated
        # path planning algorithms like GPU-based A* or Dijkstra

        # Mock implementation showing the concept
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # In Isaac ROS, this would generate waypoints using GPU acceleration
        # For this example, we'll create a simple straight-line path
        start_x, start_y = 0.0, 0.0  # Would come from current position
        goal_x = goal_msg.pose.position.x
        goal_y = goal_msg.pose.position.y

        # Generate intermediate waypoints
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            wp_x = start_x + t * (goal_x - start_x)
            wp_y = start_y + t * (goal_y - start_y)

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = wp_x
            pose.pose.position.y = wp_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        return path_msg

    def update_local_plan(self, scan_msg):
        """
        Update local plan based on laser scan using Isaac ROS obstacle avoidance
        """
        # In Isaac ROS, this would use GPU-accelerated local planning
        # and obstacle avoidance algorithms
        pass

    def execute_navigation(self):
        """
        Execute navigation with Isaac ROS GPU-accelerated control
        """
        # This would interface with Isaac ROS's GPU-accelerated trajectory controller
        # and local planner for smooth navigation
        pass

    def stop_navigation(self):
        """
        Stop navigation and clear plans
        """
        self.navigation_active = False
        self.global_plan = None
        self.local_plan = None

        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        self.get_logger().info('Navigation stopped')
```

### Perception and AI Packages

#### Isaac ROS DetectNet
AI-powered object detection:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header
import numpy as np


class DetectNetNode(Node):
    """
    Isaac ROS DetectNet node for GPU-accelerated object detection
    """

    def __init__(self):
        super().__init__('isaac_ros_detectnet_node')

        # Subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect',
            self.image_callback,
            10
        )

        # Publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detectnet/detections',
            10
        )

        # Isaac ROS DetectNet parameters
        self.model_name = 'ssd_mobilenet_v2_coco'
        self.confidence_threshold = 0.5
        self.max_objects = 50

        # GPU memory management
        self.gpu_memory_fraction = 0.8
        self.tensorrt_precision = 'fp16'  # Half precision for better performance

        self.get_logger().info('Isaac ROS DetectNet node initialized')

    def image_callback(self, msg):
        """
        Process image for object detection using Isaac ROS DetectNet
        """
        # In a real Isaac ROS DetectNet implementation:
        # 1. Image is preprocessed for TensorRT inference
        # 2. GPU-accelerated inference is performed
        # 3. Results are post-processed and formatted as Detection2DArray

        # For this example, we'll simulate the process
        self.get_logger().info(f'Processing image for detection: {msg.width}x{msg.height}')

        # Simulate GPU-accelerated detection
        detections = self.run_detectnet_inference(msg)

        if detections:
            detection_msg = Detection2DArray()
            detection_msg.header = msg.header
            detection_msg.detections = detections
            self.detection_pub.publish(detection_msg)

            self.get_logger().info(f'Published {len(detections)} detections')

    def run_detectnet_inference(self, image_msg):
        """
        Run object detection inference (mock implementation showing Isaac ROS concepts)
        """
        # In Isaac ROS DetectNet:
        # 1. Image is converted to TensorRT-compatible format
        # 2. TensorRT engine performs inference on GPU
        # 3. Results are decoded and formatted as detections

        # This is a placeholder for the GPU-accelerated inference
        # that would happen in Isaac ROS DetectNet
        return []

    def load_trt_model(self):
        """
        Load TensorRT optimized model for GPU inference
        """
        # In Isaac ROS, this would load a pre-optimized TensorRT engine
        # for the specified model (e.g., SSD MobileNet, YOLO, etc.)
        pass
```

#### Isaac ROS Segmentation
Semantic and instance segmentation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header


class SegmentationNode(Node):
    """
    Isaac ROS Segmentation node for GPU-accelerated semantic/instance segmentation
    """

    def __init__(self):
        super().__init__('isaac_ros_segmentation_node')

        # Subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect',
            self.image_callback,
            10
        )

        # Publisher for segmentation results
        self.segmentation_pub = self.create_publisher(
            Image,  # Segmentation mask as image
            '/segmentation/mask',
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/segmentation/detections',
            10
        )

        # Segmentation parameters
        self.model_name = 'fcn-resnet50'
        self.segmentation_type = 'semantic'  # or 'instance'
        self.confidence_threshold = 0.7

        self.get_logger().info('Isaac ROS Segmentation node initialized')

    def image_callback(self, msg):
        """
        Process image for segmentation using Isaac ROS
        """
        # In a real Isaac ROS Segmentation implementation:
        # 1. Image is fed to GPU-accelerated segmentation network
        # 2. Pixel-wise classification is computed
        # 3. Results are processed into masks and object detections

        self.get_logger().info(f'Processing image for segmentation: {msg.width}x{msg.height}')

        # Simulate GPU-accelerated segmentation
        segmentation_mask, detections = self.run_segmentation_inference(msg)

        if segmentation_mask is not None:
            # Publish segmentation mask
            mask_msg = segmentation_mask  # Would be properly formatted image msg
            mask_msg.header = msg.header
            self.segmentation_pub.publish(mask_msg)

        if detections:
            # Publish detections
            detection_msg = Detection2DArray()
            detection_msg.header = msg.header
            detection_msg.detections = detections
            self.detection_pub.publish(detection_msg)

    def run_segmentation_inference(self, image_msg):
        """
        Run segmentation inference (mock implementation)
        """
        # In Isaac ROS Segmentation:
        # 1. Image is fed to GPU-accelerated segmentation model
        # 2. Per-pixel classification is computed
        # 3. Results are processed into masks and object detections

        # Return mock results
        return None, []
```

## Isaac ROS Integration Examples

### Complete Navigation Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from vision_msgs.msg import Detection2DArray


class IsaacROSCompletePipelineNode(Node):
    """
    Complete Isaac ROS pipeline integrating perception, SLAM, and navigation
    """

    def __init__(self):
        super().__init__('isaac_ros_complete_pipeline')

        # Initialize Isaac ROS components
        self.setup_perception_pipeline()
        self.setup_slam_pipeline()
        self.setup_navigation_pipeline()

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(OccupancyGrid, '/debug_map', 10)

        # System state
        self.localization_active = False
        self.mapping_active = True
        self.navigation_active = False

        # Performance monitoring
        self.inference_times = []
        self.pipeline_latency = []

        self.get_logger().info('Isaac ROS Complete Pipeline node initialized')

    def setup_perception_pipeline(self):
        """
        Set up Isaac ROS perception components
        """
        # Subscribe to camera and sensors
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_rect', self.camera_callback, 10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

    def setup_slam_pipeline(self):
        """
        Set up Isaac ROS SLAM components
        """
        # Subscribe to odometry from VSLAM
        self.vslam_odom_sub = self.create_subscription(
            Odometry, '/visual_slam/odometry', self.vslam_odom_callback, 10
        )

        # Subscribe to map
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

    def setup_navigation_pipeline(self):
        """
        Set up Isaac ROS navigation components
        """
        # Subscribe to navigation goals
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10
        )

        # Subscribe to obstacle detections
        self.obstacle_sub = self.create_subscription(
            Detection2DArray, '/detectnet/detections', self.obstacle_callback, 10
        )

    def camera_callback(self, msg):
        """
        Process camera image through Isaac ROS perception pipeline
        """
        start_time = self.get_clock().now().nanoseconds / 1e9

        # In Isaac ROS, this would trigger the complete perception pipeline:
        # 1. Isaac ROS AprilTag detection
        # 2. Isaac ROS DetectNet for object detection
        # 3. Isaac ROS Segmentation for scene understanding
        # 4. Isaac ROS Depth estimation

        # For this example, we'll simulate the pipeline
        self.process_perception_pipeline(msg)

        end_time = self.get_clock().now().nanoseconds / 1e9
        self.pipeline_latency.append(end_time - start_time)

    def laser_callback(self, msg):
        """
        Process laser scan through Isaac ROS pipeline
        """
        # In Isaac ROS, this would interface with GPU-accelerated
        # obstacle detection and costmap generation
        self.update_local_map(msg)

    def vslam_odom_callback(self, msg):
        """
        Process VSLAM odometry
        """
        # Update robot pose estimate from Isaac ROS VSLAM
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        """
        Handle navigation goal
        """
        if self.localization_active:
            self.start_navigation(msg.pose)

    def obstacle_callback(self, msg):
        """
        Process obstacle detections from Isaac ROS perception
        """
        # Use Isaac ROS perception results for navigation
        self.update_obstacle_information(msg)

    def process_perception_pipeline(self, image_msg):
        """
        Process image through complete Isaac ROS perception pipeline
        """
        # This would orchestrate the Isaac ROS perception modules:
        # - Object detection
        # - Semantic segmentation
        # - Depth estimation
        # - Scene understanding
        pass

    def update_local_map(self, laser_msg):
        """
        Update local map with laser data using Isaac ROS
        """
        # In Isaac ROS, this would use GPU-accelerated costmap generation
        pass

    def start_navigation(self, goal_pose):
        """
        Start navigation to goal using Isaac ROS
        """
        # Plan path using Isaac ROS GPU-accelerated planners
        self.navigation_active = True
        # Implementation would use Isaac ROS navigation stack
```

## Configuration and Launch Files

### Isaac ROS Launch File Example

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    camera_namespace = LaunchConfiguration('camera_namespace', default='/camera')
    robot_namespace = LaunchConfiguration('robot_namespace', default='')

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    camera_namespace_arg = DeclareLaunchArgument(
        'camera_namespace',
        default_value='/camera',
        description='Namespace for camera topics'
    )

    robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value='',
        description='Robot namespace'
    )

    # Isaac ROS Visual SLAM container
    visual_slam_container = ComposableNodeContainer(
        name='visual_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='isaac_ros::visual_slam::VisualSlamNode',
                name='visual_slam',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'enable_rectification': True,
                    'rectified_images': True,
                    'enable_debug_mode': False,
                    'input_viz_thresh': 0.99
                }],
                remappings=[
                    ('/visual_slam/image0', [camera_namespace, '/left/image_rect']),
                    ('/visual_slam/camera_info0', [camera_namespace, '/left/camera_info']),
                    ('/visual_slam/image1', [camera_namespace, '/right/image_rect']),
                    ('/visual_slam/camera_info1', [camera_namespace, '/right/camera_info']),
                    ('/visual_slam/imu', '/imu/data')
                ]
            )
        ],
        output='screen'
    )

    # Isaac ROS AprilTag container
    apriltag_container = ComposableNodeContainer(
        name='apriltag_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_apriltag',
                plugin='AprilTagNode',
                name='apriltag',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'family': 'tag36h11',
                    'size': 0.166,
                    'max_tags': 30,
                    'debug': False
                }],
                remappings=[
                    ('/image', [camera_namespace, '/image_rect']),
                    ('/camera_info', [camera_namespace, '/camera_info'])
                ]
            )
        ],
        output='screen'
    )

    # Isaac ROS Detection container
    detection_container = ComposableNodeContainer(
        name='detection_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_detectnet',
                plugin='isaac_ros::detection::DetectNetNode',
                name='detectnet',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'model_name': 'ssd_mobilenet_v2_coco',
                    'input_width': 300,
                    'input_height': 300,
                    'confidence_threshold': 0.5,
                    'max_batch_size': 1
                }],
                remappings=[
                    ('/image', [camera_namespace, '/image_raw']),
                    ('/detections', '/detectnet/detections')
                ]
            )
        ],
        output='screen'
    )

    # Isaac ROS Navigation nodes
    nav_nodes = [
        Node(
            package='isaac_ros_navigation',
            executable='isaac_ros_navigation_server',
            name='navigation_server',
            namespace=robot_namespace,
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('my_robot_config'),
                    'config',
                    'navigation.yaml'
                ]),
                {
                    'use_sim_time': use_sim_time,
                    'planner_frequency': 5.0,
                    'controller_frequency': 20.0,
                    'use_gpu_path_planning': True
                }
            ],
            remappings=[
                ('/map', '/map'),
                ('/cmd_vel', '/cmd_vel'),
                ('/odom', '/odom'),
                ('/scan', '/scan')
            ],
            output='screen'
        ),
        Node(
            package='isaac_ros_navigation',
            executable='isaac_ros_navigation_lifecycle_manager',
            name='navigation_lifecycle_manager',
            namespace=robot_namespace,
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ]

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(use_sim_time_arg)
    ld.add_action(camera_namespace_arg)
    ld.add_action(robot_namespace_arg)

    # Add containers
    ld.add_action(visual_slam_container)
    ld.add_action(apriltag_container)
    ld.add_action(detection_container)

    # Add navigation nodes
    for nav_node in nav_nodes:
        ld.add_action(nav_node)

    return ld
```

## Performance Optimization

### GPU Acceleration Configuration

```yaml
# config/isaac_ros_performance.yaml
isaac_ros_visual_slam:
  ros__parameters:
    # Enable GPU acceleration
    enable_gpu_acceleration: true

    # CUDA settings
    cuda_device_id: 0
    cuda_stream_count: 4

    # Feature tracking optimization
    max_features: 2000
    min_feature_distance: 10.0
    feature_tracker_quality_level: 0.0001

    # Bundle adjustment settings
    enable_bundle_adjustment: true
    ba_solver_type: 'LevenbergMarquardt'
    ba_max_iterations: 10

isaac_ros_detectnet:
  ros__parameters:
    # TensorRT optimization
    tensorrt_engine: '/path/to/optimized/engine.plan'
    tensorrt_precision: 'FP16'  # FP16 for better performance
    max_batch_size: 1

    # GPU memory settings
    gpu_memory_fraction: 0.8
    input_tensor_layout: 'NHWC'

isaac_ros_apriltag:
  ros__parameters:
    # Multi-threading
    num_threads: 4

    # Optimization settings
    quad_decimate: 1.0  # Lower for better performance
    decode_sharpening: 0.25

isaac_ros_navigation:
  ros__parameters:
    # GPU-accelerated path planning
    use_gpu_path_planning: true
    gpu_path_optimizer_enabled: true

    # Local planner optimization
    use_gpu_local_planner: true
    gpu_collision_checker_enabled: true

    # Costmap optimization
    use_gpu_costmap: true
    gpu_costmap_update_rate: 10.0
```

### Resource Management

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import subprocess
import re


class IsaacROSResourceMonitor(Node):
    """
    Monitor system resources for Isaac ROS optimization
    """

    def __init__(self):
        super().__init__('isaac_ros_resource_monitor')

        # Publisher for resource status
        self.resource_pub = self.create_publisher(Int32, '/system_resources', 10)

        # Timer for periodic monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_resources)

        self.get_logger().info('Isaac ROS Resource Monitor initialized')

    def monitor_resources(self):
        """
        Monitor CPU, GPU, and memory usage
        """
        # Get GPU utilization (NVIDIA-specific)
        gpu_usage = self.get_gpu_utilization()

        # Get CPU and memory usage
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()

        # Create composite resource metric
        resource_metric = int((gpu_usage + cpu_usage + memory_usage) / 3)

        # Publish resource status
        resource_msg = Int32()
        resource_msg.data = resource_metric
        self.resource_pub.publish(resource_msg)

        # Log resource usage
        self.get_logger().debug(
            f'GPU: {gpu_usage}%, CPU: {cpu_usage}%, Memory: {memory_usage}%'
        )

        # Adjust Isaac ROS parameters based on resource usage
        if resource_metric > 80:
            self.throttle_processing()
        elif resource_metric < 50:
            self.increase_processing_rate()

    def get_gpu_utilization(self):
        """
        Get GPU utilization percentage
        """
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True)
            gpu_util = result.stdout.strip()
            return int(gpu_util) if gpu_util.isdigit() else 0
        except:
            return 0

    def get_cpu_usage(self):
        """
        Get CPU usage percentage
        """
        try:
            result = subprocess.run(['top', '-bn1', '-p1'], capture_output=True, text=True)
            # Parse CPU usage from top output
            for line in result.stdout.split('\n'):
                if '%Cpu(s)' in line:
                    # Extract CPU usage from line like: "%Cpu(s):  5.2 us,  2.1 sy,  0.0 ni, 92.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st"
                    parts = line.split(',')
                    for part in parts:
                        if 'id' in part:  # idle percentage
                            idle_pct = float(part.split()[0])
                            return 100 - idle_pct
            return 0
        except:
            return 0

    def get_memory_usage(self):
        """
        Get memory usage percentage
        """
        try:
            result = subprocess.run(['free'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            mem_line = lines[1]  # Memory line
            parts = mem_line.split()
            total = int(parts[1])
            used = int(parts[2])
            return int((used / total) * 100)
        except:
            return 0

    def throttle_processing(self):
        """
        Reduce processing rate when resources are high
        """
        # In a real system, this would adjust Isaac ROS parameters
        # to reduce computational load
        self.get_logger().warn('Reducing processing rate due to high resource usage')

    def increase_processing_rate(self):
        """
        Increase processing rate when resources are available
        """
        # In a real system, this would adjust Isaac ROS parameters
        # to increase computational utilization
        self.get_logger().info('Increasing processing rate - resources available')
```

## Troubleshooting Tips

- If Isaac ROS packages don't start, verify CUDA and TensorRT are properly installed
- For poor VSLAM performance, check camera calibration and lighting conditions
- If object detection is slow, verify GPU memory availability and TensorRT engine
- For navigation issues, ensure proper TF tree and sensor data quality
- If IMU integration fails, check sensor data frequency and calibration
- For memory issues, reduce image resolution or processing frequency
- If tracking is lost frequently, improve lighting or add more visual features
- For localization drift, calibrate sensors and verify proper initialization
- If GPU acceleration isn't working, verify Isaac ROS packages were built with CUDA support
- For Docker issues, ensure GPU passthrough is properly configured with nvidia-docker

## Cross-References

For related concepts, see:
- [ROS 2 Fundamentals](../ros2-core/index.md) - For ROS 2 integration
- [NVIDIA Isaac Sim](../isaac/index.md) - For simulation integration
- [Sensors](../sensors/index.md) - For sensor configuration
- [Navigation Systems](../hri/index.md) - For navigation concepts
- [Unity Visualization](../unity/index.md) - For alternative simulation

## Summary

Isaac ROS provides a powerful set of GPU-accelerated packages for robotics perception, SLAM, and navigation. When properly configured and integrated, Isaac ROS can significantly enhance the performance and capabilities of robotic systems. Success requires understanding both the hardware acceleration capabilities and the specific configuration requirements for each package.