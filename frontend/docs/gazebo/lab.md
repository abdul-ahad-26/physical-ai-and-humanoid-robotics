---
title: Gazebo Simulation Exercises
sidebar_position: 6.2
description: Hands-on exercises to practice Gazebo simulation workflows
---

# Gazebo Simulation Exercises

## Learning Objectives

- Create and configure a basic simulation environment
- Integrate a robot model with Gazebo
- Control a simulated robot using ROS 2
- Configure sensors in simulation
- Troubleshoot common simulation issues
- Analyze simulation results

## Prerequisites

- ROS 2 environment properly sourced
- Gazebo installed and configured
- Basic understanding of URDF robot models
- Python knowledge for ROS 2 nodes

## Exercise 1: Launching Your First Simulation

### Task
Launch a basic Gazebo simulation and familiarize yourself with the interface.

### Steps

1. Start Gazebo with an empty world:
   ```bash
   gz sim -r -v 4 empty.sdf
   ```

2. Explore the Gazebo interface:
   - Camera controls (pan, zoom, rotate)
   - World properties
   - Model database
   - Physics settings

3. Add a simple object to the simulation:
   - Use the Insert tab to add a cube or sphere
   - Position the object in the world
   - Observe the physics simulation

4. Close the simulation when finished:
   ```bash
   # In another terminal
   pkill gz
   ```

### Expected Results
You should see a 3D simulation environment with a ground plane and a sun light source. You should be able to navigate the camera and add objects to the scene.

## Exercise 2: Creating a Custom World

### Task
Create a custom world file with specific environmental features.

### Steps

1. Create a new world file `~/gazebo_worlds/my_room.sdf`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="my_room">
       <!-- Include standard models -->
       <include>
         <uri>model://ground_plane</uri>
       </include>
       <include>
         <uri>model://sun</uri>
       </include>

       <!-- Define physics -->
       <physics name="1ms" type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1</real_time_factor>
         <real_time_update_rate>1000</real_time_update_rate>
       </physics>

       <!-- Create a simple room -->
       <model name="wall_1">
         <pose>0 5 1 0 0 0</pose>
         <link name="link">
           <visual name="visual">
             <geometry>
               <box>
                 <size>10 0.2 2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.8 0.8 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
             </material>
           </visual>
           <collision name="collision">
             <geometry>
               <box>
                 <size>10 0.2 2</size>
               </box>
             </geometry>
           </collision>
           <inertial>
             <mass>1</mass>
             <inertia>
               <ixx>1</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>1</iyy>
               <iyz>0</iyz>
               <izz>1</izz>
             </inertia>
           </inertial>
         </link>
       </model>

       <!-- Add a simple object -->
       <model name="box_object">
         <pose>0 0 0.5 0 0 0</pose>
         <link name="link">
           <visual name="visual">
             <geometry>
               <box>
                 <size>1 1 1</size>
               </box>
             </geometry>
             <material>
               <ambient>0.1 0.9 0.1 1</ambient>
               <diffuse>0.1 0.9 0.1 1</diffuse>
             </material>
           </visual>
           <collision name="collision">
             <geometry>
               <box>
                 <size>1 1 1</size>
               </box>
             </geometry>
           </collision>
           <inertial>
             <mass>1</mass>
             <inertia>
               <ixx>1</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>1</iyy>
               <iyz>0</iyz>
               <izz>1</izz>
             </inertia>
           </inertial>
         </link>
       </model>
     </world>
   </sdf>
   ```

2. Launch Gazebo with your custom world:
   ```bash
   gz sim -r -v 4 ~/gazebo_worlds/my_room.sdf
   ```

### Expected Results
You should see a custom world with walls and a box object. The physics should work correctly, and the box should respond to gravity.

## Exercise 3: Robot Integration

### Task
Integrate a simple robot model into Gazebo and verify it works correctly.

### Steps

1. Create a simple differential drive robot URDF file `~/robot_models/simple_robot.urdf`:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_robot">
     <!-- Base link -->
     <link name="base_link">
       <visual>
         <geometry>
           <cylinder radius="0.2" length="0.1"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.2" length="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1"/>
         <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Left wheel -->
     <link name="left_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.2"/>
         <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
       </inertial>
     </link>

     <!-- Right wheel -->
     <link name="right_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.2"/>
         <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
       </inertial>
     </link>

     <!-- Joints -->
     <joint name="left_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="left_wheel"/>
       <origin xyz="0 0.15 -0.05" rpy="1.5708 0 0"/>
       <axis xyz="0 0 1"/>
     </joint>

     <joint name="right_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="right_wheel"/>
       <origin xyz="0 -0.15 -0.05" rpy="1.5708 0 0"/>
       <axis xyz="0 0 1"/>
     </joint>

     <!-- Gazebo plugin for ROS control -->
     <gazebo>
       <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
         <parameters>$(find my_robot_package)/config/my_controllers.yaml</parameters>
       </plugin>
     </gazebo>
   </robot>
   ```

2. Launch Gazebo with the robot:
   ```bash
   # Convert URDF to SDF and spawn
   ros2 run gazebo_ros spawn_entity.py -entity simple_robot -file ~/robot_models/simple_robot.urdf -x 0 -y 0 -z 0.2
   ```

### Expected Results
You should see your robot model in the simulation environment. It should be positioned above the ground and respond to physics.

## Exercise 4: ROS 2 Integration

### Task
Connect your simulated robot to ROS 2 and control it with ROS 2 commands.

### Steps

1. Launch Gazebo with ROS 2 bridge:
   ```bash
   # Terminal 1: Start Gazebo
   gz sim -r -v 4 empty.sdf

   # Terminal 2: Start ROS 2 bridge
   source /opt/ros/humble/setup.bash
   ros2 run ros_gz_bridge parameter_bridge
   ```

2. Check available topics:
   ```bash
   ros2 topic list
   ```

3. Control the robot (if you have a differential drive robot):
   ```bash
   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 1.0}, angular: {z: 0.5}}'
   ```

4. Monitor sensor data:
   ```bash
   ros2 topic echo /camera/image_raw
   ```

### Expected Results
You should be able to control your simulated robot using ROS 2 commands and receive sensor data from the simulation.

## Exercise 5: Sensor Configuration

### Task
Add and configure sensors on your robot model.

### Steps

1. Modify your robot URDF to include a camera sensor:
   ```xml
   <!-- Add to your base_link or a separate sensor link -->
   <link name="camera_link">
     <visual>
       <geometry>
         <box size="0.05 0.05 0.05"/>
       </geometry>
     </visual>
     <collision>
       <geometry>
         <box size="0.05 0.05 0.05"/>
       </geometry>
     </collision>
     <inertial>
       <mass value="0.01"/>
       <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
     </inertial>
   </link>

   <joint name="camera_joint" type="fixed">
     <parent link="base_link"/>
     <child link="camera_link"/>
     <origin xyz="0.15 0 0.05" rpy="0 0 0"/>
   </joint>

   <!-- Gazebo sensor plugin -->
   <gazebo reference="camera_link">
     <sensor name="camera" type="camera">
       <camera>
         <horizontal_fov>1.047</horizontal_fov>
         <image>
           <width>640</width>
           <height>480</height>
         </image>
         <clip>
           <near>0.1</near>
           <far>100</far>
         </clip>
       </camera>
       <always_on>1</always_on>
       <update_rate>30</update_rate>
       <visualize>true</visualize>
     </sensor>
   </gazebo>
   ```

2. Launch the simulation and verify the camera is publishing data:
   ```bash
   # Check if camera topic exists
   ros2 topic list | grep camera

   # View camera data
   ros2 topic echo /camera/image_raw
   ```

### Expected Results
Your robot should have a camera sensor that publishes image data to ROS 2.

## Exercise 6: Simulation Analysis

### Task
Analyze simulation performance and accuracy.

### Steps

1. Run a simulation scenario for a fixed time:
   ```bash
   # Record simulation time vs real time
   # Monitor physics update rate
   # Check for dropped frames
   ```

2. Compare simulation results with expected behavior:
   - Robot movement accuracy
   - Sensor data quality
   - Physics realism

3. Adjust physics parameters if needed:
   ```bash
   # Try different max_step_size values
   # Adjust real_time_factor
   # Modify solver parameters
   ```

### Expected Results
You should understand the trade-offs between simulation accuracy and performance.

## Troubleshooting Tips

- If the robot falls through the ground, check collision properties
- If simulation runs slowly, increase max_step_size or decrease update rate
- If sensors don't publish data, verify the ROS 2 bridge is running
- Use `gz topic -l` to list available topics
- Check that your robot model has proper inertial properties
- Verify joint limits and dynamics are properly defined

## Summary

These exercises provided hands-on experience with Gazebo simulation, from basic world creation to complex robot integration with ROS 2. Understanding these workflows is essential for effective robotics development and testing in simulation before moving to real hardware.