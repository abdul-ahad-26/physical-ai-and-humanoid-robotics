// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro/index'],
    },
    {
      type: 'category',
      label: 'Embodied Intelligence',
      items: [
        'embodied-intelligence/index',
      ],
    },
    {
      type: 'category',
      label: 'Sensors & Perception',
      items: [
        'sensors/index',
      ],
    },
    {
      type: 'category',
      label: 'ROS 2 Fundamentals',
      items: [
        'ros2-core/index',
        'ros2-core/lab',
        'ros2-core/code',
        'ros2-packages/index',
      ],
    },
    {
      type: 'category',
      label: 'Robot Modeling',
      items: [
        'urdf/index',
      ],
    },
    {
      type: 'category',
      label: 'Simulation',
      items: [
        'gazebo/index',
        'gazebo/lab',
        'gazebo/code',
        'unity/index',
      ],
    },
    {
      type: 'category',
      label: 'NVIDIA Isaac',
      items: [
        'isaac/index',
        'isaac-ros/index',
      ],
    },
    {
      type: 'category',
      label: 'AI for Robotics',
      items: [
        'vla/index',
        'vla/lab',
        'vla/code',
        'conversational-robotics/index',
        'conversational-robotics/lab',
        'conversational-robotics/code',
      ],
    },
    {
      type: 'category',
      label: 'Robotics Fundamentals',
      items: [
        'humanoid-kinematics/index',
        'manipulation/index',
        'hri/index',
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/index',
      ],
    },
    {
      type: 'category',
      label: 'Hardware & Infrastructure',
      items: [
        'hardware-labs/index',
        'hardware-labs/lab',
        'hardware-labs/code',
        'cloud-vs-onprem/index',
        'cloud-vs-onprem/lab',
        'cloud-vs-onprem/code',
      ],
    },
  ],
};

export default sidebars;