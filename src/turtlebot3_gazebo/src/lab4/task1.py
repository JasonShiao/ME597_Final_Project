#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from tf2_msgs.msg import TFMessage

import cv2
import numpy as np
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform
import queue
from enum import IntEnum
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.ndimage import binary_dilation

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


class MapMark(IntEnum):
    MAP_OPEN = 0
    MAP_CLOSE = 1
    FRONTIER_OPEN = 2
    FRONTIER_CLOSE = 3

class CellState(IntEnum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100

class Frontier():
    def __init__(self):
        self.points = []
        self.min_distance = np.inf
        self.centroid = (0, 0)
        self.middle = (0, 0)

    def size(self):
        return 1 if len(self.points) == 0 else len(self.points)
    
    def getCentroid(self):
        pass

    def getMinDistance(self):
        return self.min_distance


def WFDFrontierExtract(map: OccupancyGrid, robot_pos: tuple, min_frontier_size=0.4):
    """
    Extract frontier points from the map
    :param map: OccupancyGrid message
    :param robot_pos: robot position (x,y) in map frame
    :param min_frontier_size: minimum frontier size (in real world unit m) to be considered as a frontier
    :return: list of frontiers, each frontier is a list of points
    """
    def getAdjacent(p: tuple, diag = False):
        # 8 adjacent points if diag is True, 4 adjacent points if diag is False
        x, y = p
        neighbor = []
        neighbor.append((x-1, y))
        neighbor.append((x, y-1))
        neighbor.append((x, y+1))
        neighbor.append((x+1, y))
        if diag:
            neighbor.append((x-1, y-1))
            neighbor.append((x-1, y+1))
            neighbor.append((x+1, y-1))
            neighbor.append((x+1, y+1))
        return neighbor
    
    def isFrontier(p: tuple, map_data: np.ndarray):
        x, y = p
        # if at unexplored cell and there is at least one free grid in the "4" adjacent/reachable cells
        if x < 0 or x >= map_data.shape[1] or y < 0 or y >= map_data.shape[0]:
            return False
        if map_data[y, x] != CellState.UNKNOWN:
            return False
        adj = getAdjacent(p, diag=False)
        for n in adj:
            x, y = n
            if x < 0 or x >= map_data.shape[1] or y < 0 or y >= map_data.shape[0]:
                continue
            if map_data[y, x] == CellState.FREE:
                return True
        return False

    def buildFrontier(initial_point: tuple, robot_pos: tuple, map_data: np.ndarray, marker: dict):
        new_frontier = Frontier()
        frontier_q = queue.Queue()
        frontier_q.put(initial_point)
        marker[initial_point] = MapMark.FRONTIER_OPEN
        while not frontier_q.empty():
            p = frontier_q.get()
            if p in marker and marker[p] in [MapMark.MAP_CLOSE, MapMark.FRONTIER_CLOSE]:
                continue
            if isFrontier(p, map_data):
                new_frontier.points.append(p)
                x, y = p
                # centroid
                new_frontier.centroid = (new_frontier.centroid[0] + x, new_frontier.centroid[1] + y)
                # calculate the distance from the robot to the frontier
                distance = np.sqrt((x - robot_pos[0])**2 + (y - robot_pos[1])**2)
                if distance < new_frontier.min_distance:
                    new_frontier.min_distance = distance
                    new_frontier.middle = p
                for w in getAdjacent(p, diag=True):
                    if w not in marker or marker[w] not in [MapMark.FRONTIER_OPEN, MapMark.FRONTIER_CLOSE, MapMark.MAP_CLOSE]:
                        frontier_q.put(w)
                        marker[w] = MapMark.FRONTIER_OPEN
                marker[p] = MapMark.FRONTIER_CLOSE
        for f_point in new_frontier.points:
            marker[f_point] = MapMark.MAP_CLOSE
        
        # Finalize the centroid
        new_frontier.centroid = (round(new_frontier.centroid[0] / new_frontier.size()), round(new_frontier.centroid[1] / new_frontier.size()))
        return new_frontier

    # Note: in (y, x) format
    map_data = np.array(map.data).reshape((map.info.height, map.info.width))
    # Dilate the map for clearance
    occupied_mask = (map_data == CellState.OCCUPIED)
    dilation_mask = binary_dilation(occupied_mask, structure=np.ones((9,9)))
    dilated_map_data = np.copy(map_data)
    dilated_map_data[dilation_mask] = CellState.OCCUPIED
    dilated_map_data[map_data == CellState.UNKNOWN] = CellState.UNKNOWN

    map_q = queue.Queue()
    frontiers: list[Frontier] = []
    marker = {}
    idx_x, idx_y = real_pose_to_map_grid_pos(robot_pos[0], robot_pos[1], map.info.origin, map.info.resolution)
    map_q.put((idx_x, idx_y))
    marker[(idx_x, idx_y)] = MapMark.MAP_OPEN
    while not map_q.empty():
        p: tuple = map_q.get()
        if p in marker and marker[p] == MapMark.MAP_CLOSE:
            continue
        if isFrontier(p, dilated_map_data):
            # For a frontier point p, bfs to find all connected frontier points
            new_frontier = buildFrontier(p, robot_pos, dilated_map_data, marker)
            # Ignore small frontiers
            if new_frontier.size() * map.info.resolution >= min_frontier_size:
                frontiers.append(new_frontier)
        for v in getAdjacent(p, diag=True):
            if (v not in marker or marker[v] not in [MapMark.MAP_OPEN, MapMark.MAP_CLOSE]):
                # check if v has at least one free space neighbor (4 adjacent grids)
                reachable_neighbor = getAdjacent(v, diag=False)
                v_reachable = False
                for n in reachable_neighbor:
                    x, y = n
                    if (x >= 0 and x < dilated_map_data.shape[1] and y >= 0 and y < dilated_map_data.shape[0]) \
                       and dilated_map_data[y, x] == CellState.FREE:
                        v_reachable = True
                        break
                if v_reachable:
                    map_q.put(v)
                    marker[v] = MapMark.MAP_OPEN
        marker[p] = MapMark.MAP_CLOSE
    
    return frontiers


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
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier', 3)

        self.odom_trans = np.eye(4)       # map (world) frame to odom frame
        self.robot_base_trans = np.eye(4) # odom frame to robot base frame
        self.robot_pose_trans = np.eye(4) # map (world) frame to robot base frame


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
        # Get frontier and mark it on the map
        frontiers = WFDFrontierExtract(grid_map_msg, (self.robot_pose_trans[0, 3], self.robot_pose_trans[1, 3]))
        self.get_logger().info(f"Frontier count: {len(frontiers)}")
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map" # Change to your desired frame
        #marker.ns = "point"
        marker.id = 1
        marker.ns = "fontier_points"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        # Set properties
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.header.stamp = self.get_clock().now().to_msg()
        for frontier in frontiers:
            for f_point in frontier.points:
                grid_x, grid_y = f_point
                x, y = map_grid_pos_to_real_pos(grid_x, grid_y, grid_map_msg.info.origin, grid_map_msg.info.resolution)
                marker.points.append(Point(x=x, y=y, z=0.0))
        marker_array.markers.append(marker)
        marker = Marker()
        marker.header.frame_id = "map" 
        marker.id = 2
        marker.ns = "fontier_centroids"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.g = 1.0
        marker.color.a = 1.0
        for frontier in frontiers:
            x, y = frontier.centroid
            x, y = map_grid_pos_to_real_pos(x, y, grid_map_msg.info.origin, grid_map_msg.info.resolution)
            marker.points.append(Point(x=x, y=y, z=0.0))
        marker_array.markers.append(marker)
        self.frontier_pub.publish(marker_array)
        # Visualize for debugging
        #map_data = np.array(grid_map_msg.data).reshape((grid_map_msg.info.height, grid_map_msg.info.width))
        #map_data = np.where(map_data == CellState.UNKNOWN, 128, map_data) # substitute -1 with 128 for unexplored location
        #map_data = np.where(map_data == CellState.FREE, 255, map_data) # substitute 0 with 255 for free location
        #map_data = np.where(map_data == CellState.OCCUPIED, 0, map_data) # substitute 1 with 0 for occupied location
        # Flip y-axis for display from bottom to top (align with rViz and gazebo)
        #cv2.imshow('Grid map', cv2.flip(
        #    cv2.resize(map_data.astype(np.uint8),  (grid_map_msg.info.height*3, grid_map_msg.info.width*3), interpolation=cv2.INTER_NEAREST), 
        #    0)
        #)
        #cv2.waitKey(1) # Must exist for imshow to work
    

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
