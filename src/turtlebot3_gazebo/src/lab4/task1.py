#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from tf2_msgs.msg import TFMessage

import cv2
import numpy as np
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform

# Import other python packages that you think necessary
def getPoseTransSE3(pose: Pose):
    T = np.zeros((4,4))
    T[3, 3] = 1
    T[0, 3] = pose.position.x
    T[1, 3] = pose.position.y
    T[2, 3] = pose.position.z
    
    T[0:3, 0:3] = np.array([
        [1-2*(pose.orientation.y**2 + pose.orientation.z**2), 2*(pose.orientation.x * pose.orientation.y - pose.orientation.w * pose.orientation.z), 2*(pose.orientation.x * pose.orientation.z + pose.orientation.w * pose.orientation.y)],
        [2*(pose.orientation.x * pose.orientation.y + pose.orientation.w * pose.orientation.z), 1-2*(pose.orientation.x**2 + pose.orientation.z**2), 2*(pose.orientation.y * pose.orientation.z - pose.orientation.w * pose.orientation.x)],
        [2*(pose.orientation.x * pose.orientation.z - pose.orientation.w * pose.orientation.y), 2*(pose.orientation.y * pose.orientation.z + pose.orientation.w * pose.orientation.x), 1-2*(pose.orientation.x**2 + pose.orientation.y**2)]
    ])
    return T

def getTransSE3(transform: Transform):
    T = np.zeros((4,4))
    T[3, 3] = 1
    T[0, 3] = transform.translation.x
    T[1, 3] = transform.translation.y
    T[2, 3] = transform.translation.z
    
    T[0:3, 0:3] = np.array([
        [1-2*(transform.rotation.y**2 + transform.rotation.z**2), 2*(transform.rotation.x * transform.rotation.y - transform.rotation.w * transform.rotation.z), 2*(transform.rotation.x * transform.rotation.z + transform.rotation.w * transform.rotation.y)],
        [2*(transform.rotation.x * transform.rotation.y + transform.rotation.w * transform.rotation.z), 1-2*(transform.rotation.x**2 + transform.rotation.z**2), 2*(transform.rotation.y * transform.rotation.z - transform.rotation.w * transform.rotation.x)],
        [2*(transform.rotation.x * transform.rotation.z - transform.rotation.w * transform.rotation.y), 2*(transform.rotation.y * transform.rotation.z + transform.rotation.w * transform.rotation.x), 1-2*(transform.rotation.x**2 + transform.rotation.y**2)]
    ])
    return T

def inverseTrans(T):
    T_inv = np.zeros((4,4))
    T_inv[3, 3] = 1
    T_inv[0:3, 0:3] = T[0:3, 0:3].T
    T_inv[0:3, 3] = -T[0:3, 0:3].T @ T[0:3, 3]
    return T_inv

class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            10
        )
        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )

        self.odom_trans = np.eye(4)
        self.robot_base_trans = np.eye(4)


    def timer_cb(self):
        self.get_logger().info('Task1 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function

    # Define function(s) that complete the (automatic) mapping task
    def tf_callback(self, tf_msg: TFMessage):
        for stamped_transform in tf_msg.transforms:
            if stamped_transform.child_frame_id == "odom":
                self.odom_trans = getTransSE3(stamped_transform.transform)
            elif stamped_transform.child_frame_id == "base_footprint":
                self.robot_base_trans = getTransSE3(stamped_transform.transform)
            else:
                pass
        self.robot_pose_trans = self.odom_trans @ self.robot_base_trans
        #self.get_logger().info(f"Robot pose: {self.robot_pose_trans}")

    def map_callback(self, grid_map_msg: OccupancyGrid):
        self.get_logger().info("Map received")
        map_data = np.array(grid_map_msg.data).reshape((grid_map_msg.info.height, grid_map_msg.info.width))
        map_data = np.where(map_data == -1, 128, map_data) # substitute -1 with 128 for unexplored location
        map_data = np.where(map_data == 0, 255, map_data) # substitute -1 with 128 for unexplored location
        map_data = np.where(map_data == 1, 0, map_data) # substitute -1 with 128 for unexplored location
        # Flip y-axis for display from bottom to top (align with rViz and gazebo)
        cv2.imshow('Grid map', cv2.flip(
            cv2.resize(map_data.astype(np.uint8),  (grid_map_msg.info.height*3, grid_map_msg.info.width*3), interpolation=cv2.INTER_NEAREST), 
            0)
        )
        cv2.waitKey(1) # Must exist for imshow to work
    
    def map_grid_pos_to_real_pos(grid_x_idx, grid_y_idx, origin: Pose, resolution):
        # real x, y in grid origin coordinate
        x = grid_x_idx * resolution
        y = grid_y_idx * resolution
        homo_pos = np.array([x, y, 0, 1])
        # real x, y in world coordinate
        real_pos = getPoseTransSE3(origin) @ homo_pos
        return real_pos[0], real_pos[1]

    def real_pose_to_map_grid_pos(real_x, real_y, origin: Pose, resolution):
        # Find the closest grid pos
        # pos in grid origin coord
        pos = inverseTrans(getPoseTransSE3(origin)) @ np.array([real_x, real_y, 0, 1])
        # convert pos to grid idx
        grid_x_idx = round(pos[0] / resolution)
        grid_y_idx = round(pos[1] / resolution)
        return grid_x_idx, grid_y_idx


def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
