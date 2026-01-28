#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from turtlebot3_msgs.srv import Goal
import math

GOAL_TOLERANCE = 0.15  # meters

class RealWorldGoals(Node):
    def __init__(self):
        super().__init__('real_world_goals')

        # Publishers
        self.goal_pub = self.create_publisher(Pose, '/goal_pose', 10)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)

        # Service
        self.srv = self.create_service(Goal, 'goal_comm', self.goal_cb)

        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # State
        self.has_goal = False
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.distance_to_goal = None

        self.get_logger().info("✅ RealWorldGoals node initialized")

    # ---------------- Service callback ----------------
    def goal_cb(self, req, res):
        """Called by DRL agent to set a new goal."""
        self.goal_x = req.x
        self.goal_y = req.y
        self.has_goal = True
        self.distance_to_goal = None

        # Publish goal for visualization
        pose = Pose()
        pose.position.x = self.goal_x
        pose.position.y = self.goal_y
        pose.orientation.w = 1.0
        self.goal_pub.publish(pose)

        self.get_logger().info(f"🚀 New goal received: ({self.goal_x:.2f}, {self.goal_y:.2f})")
        res.new_goal = True
        return res

    # ---------------- Odometry callback ----------------
    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        if self.has_goal:
            self.distance_to_goal = math.sqrt(
                (self.goal_x - self.robot_x) ** 2 +
                (self.goal_y - self.robot_y) ** 2
            )

            self.get_logger().info(
                f"Robot: ({self.robot_x:.2f}, {self.robot_y:.2f}) | "
                f"Goal: ({self.goal_x:.2f}, {self.goal_y:.2f}) | "
                f"Dist: {self.distance_to_goal:.3f} m"
            )

            # Stop robot if goal reached
            if self.distance_to_goal < GOAL_TOLERANCE:
                self.get_logger().info("🎯 Goal reached! Stopping robot.")
                self.has_goal = False
                self.stop_robot()

    # ---------------- Stop robot ----------------
    def stop_robot(self):
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.cmd_pub.publish(stop)

    # ---------------- Set goal manually ----------------
    def set_goal(self, x, y):
        """Manually set a goal relative to current robot position."""
        self.goal_x = x
        self.goal_y = y
        self.has_goal = True
        self.distance_to_goal = None

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.orientation.w = 1.0
        self.goal_pub.publish(pose)

        self.get_logger().info(f"🚀 New manual goal set: ({x:.2f}, {y:.2f})")


def main():
    rclpy.init()
    node = RealWorldGoals()

    # Service is ready immediately after create_service()
    node.get_logger().info("✅ /goal_comm service ready for DRL agent")

    # Optional: set a relative starting goal
    rclpy.spin_once(node, timeout_sec=0.5)
    node.set_goal(node.robot_x + 0.5, node.robot_y + 0.5)

    # Spin continuously to process goals and odometry
    rclpy.spin(node)
    rclpy.shutdown()



if __name__ == "__main__":
    main()
