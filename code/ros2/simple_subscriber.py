#!/usr/bin/env python3

"""
Simple ROS 2 Subscriber Example

This example demonstrates how to create a basic subscriber node in ROS 2 using Python.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    """
    A simple subscriber node that listens to messages on the 'chatter' topic.
    """

    def __init__(self):
        super().__init__('simple_subscriber')

        # Create a subscription to the 'chatter' topic with String messages
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10  # QoS history depth
        )

        # Prevent unused variable warning
        self.subscription

        self.get_logger().info('Simple Subscriber has been started.')

    def listener_callback(self, msg):
        """
        Callback function that is called when a message is received.
        """
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    rclpy.init(args=args)

    simple_subscriber = SimpleSubscriber()

    try:
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        simple_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()