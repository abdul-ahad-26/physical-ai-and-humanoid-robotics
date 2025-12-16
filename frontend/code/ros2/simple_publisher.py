#!/usr/bin/env python3

"""
Simple ROS 2 Publisher Example

This example demonstrates how to create a basic publisher node in ROS 2 using Python.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    """
    A simple publisher node that publishes "Hello World" messages to a topic.
    """

    def __init__(self):
        super().__init__('simple_publisher')

        # Create a publisher that publishes String messages to the 'chatter' topic
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create a timer that calls the timer_callback method every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0
        self.get_logger().info('Simple Publisher has been started.')

    def timer_callback(self):
        """
        Callback function that is called by the timer.
        Publishes a message to the 'chatter' topic.
        """
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    rclpy.init(args=args)

    simple_publisher = SimplePublisher()

    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        simple_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()