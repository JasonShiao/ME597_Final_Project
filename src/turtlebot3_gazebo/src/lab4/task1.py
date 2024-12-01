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


class CellType(IntEnum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100


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

class RRTNode():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        #self.parent = None
        #self.children = []
    
    def position(self):
        return (self.x, self.y)
    
class RRT():
    def __init__(self, step_size):
        self.edges: list[tuple[RRTNode, RRTNode]] = []
        self.nodes: list[RRTNode] = []
        self.step_size = step_size
    
    def insert_node(self, new_node: RRTNode, parent_node: RRTNode):
        if parent_node:
            self.edges.append((parent_node, new_node))
            #self.nodes[parent_node.id()].children.append(new_node)
            #new_node.parent = parent_node
        else: # set root node -> reset the tree
            self.edges = []
            self.nodes = []
        self.nodes.append(new_node)
    
    def nearest_node(self, x, y):
        nearest_node = None
        nearest_dist = float('inf')
        for node in self.nodes:
            dist = (node.x - x)**2 + (node.y - y)**2
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_node = node
        return nearest_node
    
    def move(self, node, target_point):
        dx = target_point[0] - node.x
        dy = target_point[1] - node.y
        # move with step size along the direction from node to target_point
        dist = np.sqrt(dx**2 + dy**2)
        if dist > self.step_size:
            dx = dx / dist * self.step_size
            dy = dy / dist * self.step_size
        else: # dist smaller than step size
            pass
        return node.x + dx, node.y + dy
    
    def is_obstacle_free(self, node, target_point, map_data, map_info):
        # Bresenham's line algorithm
        x0_grid, y0_grid = real_pose_to_map_grid_pos(*(node.position()), map_info.origin, map_info.resolution)
        x1_grid, y1_grid = real_pose_to_map_grid_pos(*target_point, map_info.origin, map_info.resolution)
        dx = abs(x1_grid - x0_grid)
        dy = abs(y1_grid - y0_grid)
        if x0_grid < x1_grid:
            sx = 1
        else:
            sx = -1
        if y0_grid < y1_grid:
            sy = 1
        else:
            sy = -1
        err = dx - dy
        while True:
            if y0_grid < 0 or y0_grid >= map_data.shape[0] or x0_grid < 0 or x0_grid >= map_data.shape[1]:
                return False
            if map_data[y0_grid, x0_grid] == CellType.OCCUPIED:
                return False
            if x0_grid == x1_grid and y0_grid == y1_grid:
                break
            e2 = 2 * err
            if e2 > -dy:
                err = err - dy
                x0_grid = x0_grid + sx
            if e2 < dx:
                err = err + dx
                y0_grid = y0_grid + sy
        return True

    def is_unknown_region(self, x, y, map_data, map_info):
        x_grid, y_grid = real_pose_to_map_grid_pos(x, y, map_info.origin, map_info.resolution)
        return map_data[y_grid, x_grid] == CellType.UNKNOWN
    
    def explore(self, map_data, map_info): # grow the tree with one step
        p_rand = (np.random.rand() * map_info.resolution * map_data.shape[1], np.random.rand() * map_info.resolution * map_data.shape[1])
        # Get nearest point from tree to p_rand
        nearest_node = self.nearest_node(*p_rand)
        # Move from nearest node to p_rand
        p_new = self.move(nearest_node, p_rand)
        frontier_found = False
        if self.is_obstacle_free(nearest_node, p_new, map_data, map_info):
            new_node = RRTNode(*p_new)
            #self.insert_node(new_node, nearest_node)
            if self.is_unknown_region(*p_new, map_data, map_info):
                frontier_found = True
                #self.frontier_points.append(p_new)
                #self.insert_node(RRTNode(*self.robot_current_position), None)
        if frontier_found:
            return p_new
        else:
            return None


class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
        self.frontier_explore_timer = self.create_timer(0.5, self.frontier_explore_timer_cb)
        self.global_frontier_explore_timer = self.create_timer(1, self.global_frontier_explore_timer_cb)
        #self.navigation_timer = self.create_timer(0.2, self.navigation_timer_cb)
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
        self.rrt_pub = self.create_publisher(MarkerArray, '/rrt', 3)

        self.odom_trans = np.eye(4)       # map (world) frame to odom frame
        self.robot_base_trans = np.eye(4) # odom frame to robot base frame
        self.robot_pose_trans = np.eye(4) # map (world) frame to robot base frame

        self.robot_current_position = None # in real world coord
        self.last_process_position = None
        self.current_goal = None
        self.map: OccupancyGrid = None
        self.map_data: np.ndarray = None

        self.RRT = RRT(step_size=1.5) # default step size is 4m, TODO: Change according to map size
        self.frontier_points = []
        self.s_fixed = 1.0 # fixed distance for frontier point selection


    def frontier_explore_timer_cb(self):
        # Keep finding frontiers
        if not self.map:
            self.get_logger().info("Map not available")
            return
        if not self.robot_current_position:
            self.get_logger().info("Robot position not available")
            return
        
        # distance between last process position and current position
        #s_move = np.sqrt((self.robot_current_position[0] - self.last_process_position[0])**2 + (self.robot_current_position[1] - self.last_process_position[1])**2)
        #if s_move >= self.s_fixed:
        #    # Recalculate and reselect optimal frontier point
        #    # TODO:
        #    return
        # Explore frontier
        iter = 0
        while iter < 50:
            #self.get_logger().info("Exploring frontier")
            # Use real world coord instead of grid coord
            frontier_point = self.RRT.explore(self.map_data, self.map.info)
            if frontier_point:
                self.frontier_points.append(frontier_point)
                # Reset the RRT tree
                self.RRT.insert_node(RRTNode(*self.robot_current_position), None)
            iter += 1
        # Publish frontier points and edges
        #rrt_tree = MarkerArray()
        #idx = 0
        #for edge in self.RRT.edges:
        #    marker = Marker()
        #    marker.header.frame_id = "map"
        #    marker.header.stamp = self.get_clock().now().to_msg()
        #    marker.id = idx
        #    marker.ns = "local_frontier_detector"
        #    marker.type = Marker.LINE_STRIP
        #    marker.action = Marker.ADD
        #    marker.scale.x = 0.05
        #    marker.color.a = 1.0
        #    marker.color.r = 1.0
        #    marker.color.g = 0.0
        #    marker.color.b = 1.0
        #    point = Point()
        #    point.x = edge[0].x
        #    point.y = edge[0].y
        #    marker.points.append(point)
        #    point = Point()
        #    point.x = edge[1].x
        #    point.y = edge[1].y
        #    marker.points.append(point)
        #    rrt_tree.markers.append(marker)
        #    idx += 1
        #self.rrt_pub.publish(rrt_tree)

    def global_frontier_explore_timer_cb(self):
        # Use image segmentation to find frontiers
        if not self.map:
            self.get_logger().info("Map not available")
            return
        
        # Convert map to grayscale with free is white, occupied is black, unknown is gray
        map_gray = np.zeros_like(self.map_data, dtype=np.uint8)
        map_gray[self.map_data == CellType.UNKNOWN] = 128  # UNKNOWN -> gray
        map_gray[self.map_data == CellType.FREE] = 255     # FREE -> white
        map_gray[self.map_data == CellType.OCCUPIED] = 0   # OCCUPIED -> black

        # Edge detection and create a map of edges
        blurred_map = cv2.GaussianBlur(map_gray, (3, 3), 0)
        edges = cv2.Canny(blurred_map, 50, 150)
        edge_map = (edges == 255).astype(np.uint8)

        # Binarize and dilate the map for obstacles
        map_binarize_obstacle = (self.map_data == CellType.OCCUPIED).astype(np.uint8)
        map_binarize_obstacle = cv2.dilate(map_binarize_obstacle, None, iterations=5)
        # Create free/unknown map by applying bitwise NOT to obstacles
        map_binarize_free_unknown = 255 - map_binarize_obstacle    

        # AND operation of edge map and free/unknown map
        map_frontier = np.bitwise_and(edge_map, map_binarize_free_unknown)

        # Find contour and extract midpoints of edges as frontier points
        contours, _ = cv2.findContours(map_frontier, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frontier_points = []
        for contour in contours:
            M = cv2.moments(contour)
            #self.get_logger().info(f"Contour area: {cv2.contourArea(contour)}, {M['m00']}, points in contour {len(contour)}")
            if M["m00"] != 0: # avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #self.get_logger().info(f"Contour center: {cX}, {cY}")
                frontier_points.append((cX, cY))
            elif len(contour) > 3: # handle thin line edge and ignore small contour
                # Use the mean point of the contour
                cX = int(np.mean(contour[:, 0, 0]))
                cY = int(np.mean(contour[:, 0, 1]))
                #self.get_logger().info(f"Contour center: {cX}, {cY}")
                frontier_points.append((cX, cY))

        # Publish frontier points
        frontier_marker_array = MarkerArray()
        idx = 0
        # clear previous markers
        frontier_marker_array.markers = []
        clear_all_markers = Marker()
        clear_all_markers.action = Marker.DELETEALL
        clear_all_markers.ns = "global_frontier_detector"
        clear_all_markers.header.frame_id = "map"
        clear_all_markers.header.stamp = self.get_clock().now().to_msg()
        frontier_marker_array.markers.append(clear_all_markers)
        self.frontier_pub.publish(frontier_marker_array)
        frontier_marker_array.markers = []
        for i in range(len(frontier_points)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = idx
            marker.ns = "global_frontier_detector"
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            point = Point()
            point.x, point.y = map_grid_pos_to_real_pos(frontier_points[i][0], frontier_points[i][1], self.map.info.origin, self.map.info.resolution)
            marker.points.append(point)
            frontier_marker_array.markers.append(marker)
            idx += 1
        self.frontier_pub.publish(frontier_marker_array)

        # Publish local frontier points
        rrt_tree = MarkerArray()
        idx = 0
        for frontier_point in self.frontier_points:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = idx
            marker.ns = "local_frontier_detector"
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            point = Point()
            point.x, point.y = frontier_point[0], frontier_point[1]
            marker.points.append(point)
            rrt_tree.markers.append(marker)
            idx += 1
        self.rrt_pub.publish(rrt_tree)
        
    
    def navigation_timer_cb(self):
        if not self.current_goal:
            return
        # TODO:
        pass
    
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

        if not self.robot_current_position: # initialize last position and RRT
            self.robot_current_position = (self.robot_pose_trans[0, 3], self.robot_pose_trans[1, 3])
            self.last_process_position = self.robot_current_position
            self.RRT.insert_node(RRTNode(*self.robot_current_position), None)

    def map_callback(self, grid_map_msg: OccupancyGrid):
        self.get_logger().info("Map received")
        self.map = grid_map_msg
        # Convert map to numpy array
        self.map_data = np.array(grid_map_msg.data).reshape(grid_map_msg.info.height, grid_map_msg.info.width)
    

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
