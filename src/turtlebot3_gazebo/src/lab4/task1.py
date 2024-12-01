#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from tf2_msgs.msg import TFMessage

import cv2
import numpy as np
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform
from geometry_msgs.msg import PoseStamped, Twist
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

class ExplorationStatus(IntEnum):
    ROTATING_SCAN = 0
    ROTATING_SCAN_DONE = 1
    NAVIGATE_TO_FRONTIER = 2


def map_grid_pos_to_real_pos(grid_x_idx, grid_y_idx, origin: Pose, resolution):
    # real x, y in grid origin coordinate
    x = grid_x_idx * resolution
    y = grid_y_idx * resolution
    homo_pos = np.array([x, y, 0, 1])
    # real x, y in world coordinate
    real_pos = getPoseTransSE3(origin) @ homo_pos
    return real_pos[0], real_pos[1]

def real_pose_to_map_grid_pos(real_x, real_y, origin: Pose, resolution, closest=True):
    # Find the closest grid pos
    # pos in grid origin coord
    pos = inverseTrans(getPoseTransSE3(origin)) @ np.array([real_x, real_y, 0, 1])
    # convert pos to grid idx
    if closest:
        grid_x_idx = round(pos[0] / resolution)
        grid_y_idx = round(pos[1] / resolution)
    else:
        grid_x_idx = np.floor(pos[0] / resolution)
        grid_y_idx = np.floor(pos[0] / resolution)
    return grid_x_idx, grid_y_idx

def mean_shift_clustering(points, bandwidth, max_iteration=200, tolerance=5e-3):
    n_points = len(points)
    points = np.array(points)
    cluster_centers = points.copy()

    for iteration in range(max_iteration):
        new_centers = np.zeros_like(cluster_centers)
        for i, center in enumerate(cluster_centers):
            distances = np.linalg.norm(points - center, axis=1)
            in_bandwidth = points[distances <= bandwidth]
            new_centers[i] = np.mean(in_bandwidth, axis=0)
        shift = np.linalg.norm(new_centers - cluster_centers, axis=1)
        if np.max(shift) < tolerance:
            break

        cluster_centers = new_centers
    
    unique_cluster_centers = np.unique(cluster_centers, axis=0)
    # Convert to a list of tuple
    unique_cluster_centers = [tuple(center) for center in unique_cluster_centers]
    return unique_cluster_centers


def count_pixels_with_value_around_point(map_data, x, y, radius, target_val):
    # Get bounding square around the point first
    x_range = np.arange(x - radius, x + radius + 1)
    y_range = np.arange(y - radius, y + radius + 1)
    x_range = x_range[(x_range >= 0) & (x_range < map_data.shape[1])]
    y_range = y_range[(y_range >= 0) & (y_range < map_data.shape[0])]
    x_idx, y_idx = np.meshgrid(x_range, y_range)
    x_idx = x_idx.flatten()
    y_idx = y_idx.flatten()
    # Restrict to circle
    dist_squared = (x_idx - x) ** 2 + (y_idx - y) ** 2
    circle_mask = dist_squared <= radius ** 2
    count = np.sum((map_data[y_idx[circle_mask], x_idx[circle_mask]] == target_val))
    return count

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
    
    def move(self, start_point, target_point, step_size):
        dx = target_point[0] - start_point[0]
        dy = target_point[1] - start_point[1]
        # move with step size along the direction from node to target_point
        dist = np.sqrt(dx**2 + dy**2)
        if dist > step_size:
            dx = dx / dist * step_size
            dy = dy / dist * step_size
        else: # dist smaller than step size
            return target_point
        return (start_point[0] + dx, start_point[1] + dy)
    
    def check_obstacle_free_or_unexplore(self, start_point, target_point, map_data, map_info):
        # Ref: rrt_exploration ROS package
        x_i, y_i = start_point[0], start_point[1]
        fine_grain_step_size = map_info.resolution * 0.2
        step_count = int(np.ceil(np.sqrt((target_point[0] - x_i)**2 + (target_point[1] - y_i)**2) / fine_grain_step_size))
        found_obstacle = False
        found_unexplore = False
        for i in range(step_count):
            x_i, y_i = self.move((x_i, y_i), target_point, fine_grain_step_size)
            x_i_grid, y_i_grid = real_pose_to_map_grid_pos(x_i, y_i, map_info.origin, map_info.resolution)
            if y_i_grid < 0 or y_i_grid >= map_data.shape[0] or x_i_grid < 0 or x_i_grid >= map_data.shape[1]:
                pass
            elif map_data[y_i_grid, x_i_grid] == CellType.OCCUPIED:
                found_obstacle = True
            elif map_data[y_i_grid, x_i_grid] == CellType.UNKNOWN:
                found_unexplore = True
                break
        
        # if unexplored found, this will be at the border of unexplored region
        updated_target_point = (x_i, y_i)
        result = CellType.OCCUPIED if found_obstacle else CellType.UNKNOWN if found_unexplore else CellType.FREE
        
        return result, updated_target_point
        
    def explore(self, map_data, map_info): # grow the tree with one step
        # p_rand generated around current position instead of randomly in the entire map
        current_pos = self.nodes[0].position()
        xr = (np.random.rand() - 0.5) * map_data.shape[1] * map_info.resolution + current_pos[0]
        yr = (np.random.rand() - 0.5) * map_data.shape[0] * map_info.resolution + current_pos[1]
        p_rand = (xr, yr)
        # Get nearest point from tree to p_rand
        nearest_node = self.nearest_node(*p_rand)
        # Move from nearest node to p_rand
        p_new = self.move(nearest_node.position(), p_rand, self.step_size)
        frontier_found = False
        check_result, p_new = self.check_obstacle_free_or_unexplore(nearest_node.position(), p_new, map_data, map_info)
        if check_result == CellType.UNKNOWN:
            #new_node = RRTNode(*p_new)
            #self.insert_node(new_node, nearest_node)
            frontier_found = True
            #self.local_frontier_points.append(p_new)
            #self.insert_node(RRTNode(*self.robot_current_position), None)
            # TODO: Not add the node but Visualize the edge?
        elif check_result == CellType.FREE:
            new_node = RRTNode(*p_new)
            self.insert_node(new_node, nearest_node)
        
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
        self.task_allocator_timer = self.create_timer(0.2, self.task_allocator_timer_cb)
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
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.odom_trans = np.eye(4)       # map (world) frame to odom frame
        self.robot_base_trans = np.eye(4) # odom frame to robot base frame
        self.robot_pose_trans = np.eye(4) # map (world) frame to robot base frame

        self.robot_current_position = None # in real world coord
        self.current_goal = None
        self.map: OccupancyGrid = None
        self.map_data: np.ndarray = None

        self.RRT = RRT(step_size=1.0) # default step size is 4m, TODO: Change according to map size
        self.local_frontier_points = []
        self.global_frontier_points = []

        self.state = ExplorationStatus.ROTATING_SCAN
        self.rotate_scan_time = 0.0
        self.rotate_scan_duration = 3.0

    def frontier_explore_timer_cb(self):
        # Keep finding frontiers
        if not self.map:
            self.get_logger().info("Map not available")
            return
        if not self.robot_current_position:
            self.get_logger().info("Robot position not available")
            return
        
        # Explore frontier
        iter = 0
        while iter < 5:
            #self.get_logger().info("Exploring frontier")
            # Use real world coord instead of grid coord
            frontier_point = self.RRT.explore(self.map_data, self.map.info)
            if frontier_point:
                self.local_frontier_points.append(frontier_point)
                # Reset the RRT tree
                self.RRT.insert_node(RRTNode(*self.robot_current_position), None)
            iter += 1
        
        # Publish frontier points and edges
        rrt_tree = MarkerArray()
        idx = 0
        for edge in self.RRT.edges:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = idx
            marker.ns = "local_rrt_tree"
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            point = Point()
            point.x = edge[0].x
            point.y = edge[0].y
            marker.points.append(point)
            point = Point()
            point.x = edge[1].x
            point.y = edge[1].y
            marker.points.append(point)
            rrt_tree.markers.append(marker)
            idx += 1
        self.rrt_pub.publish(rrt_tree)

        # Publish local frontier points
        rrt_tree = MarkerArray()
        idx = 0
        clear_all_markers = Marker()
        clear_all_markers.action = Marker.DELETEALL
        clear_all_markers.ns = "local_frontier_detector"
        clear_all_markers.header.frame_id = "map"
        clear_all_markers.header.stamp = self.get_clock().now().to_msg()
        rrt_tree.markers.append(clear_all_markers)
        self.rrt_pub.publish(rrt_tree)
        rrt_tree.markers = []
        for frontier_point in self.local_frontier_points:
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
        self.global_frontier_points = []
        for contour in contours:
            M = cv2.moments(contour)
            #self.get_logger().info(f"Contour area: {cv2.contourArea(contour)}, {M['m00']}, points in contour {len(contour)}")
            if M["m00"] != 0: # avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #self.get_logger().info(f"Contour center: {cX}, {cY}")
                self.global_frontier_points.append(map_grid_pos_to_real_pos(cX, cY, self.map.info.origin, self.map.info.resolution))
            elif len(contour) > 3: # handle thin line edge and ignore small contour
                # Use the mean point of the contour
                cX = int(np.mean(contour[:, 0, 0]))
                cY = int(np.mean(contour[:, 0, 1]))
                #self.get_logger().info(f"Contour center: {cX}, {cY}")
                self.global_frontier_points.append(map_grid_pos_to_real_pos(cX, cY, self.map.info.origin, self.map.info.resolution))

        # Publish frontier points
        frontier_marker_array = MarkerArray()
        idx = 0
        # clear previous markers
        clear_all_markers = Marker()
        clear_all_markers.action = Marker.DELETEALL
        clear_all_markers.ns = "global_frontier_detector"
        clear_all_markers.header.frame_id = "map"
        clear_all_markers.header.stamp = self.get_clock().now().to_msg()
        frontier_marker_array.markers.append(clear_all_markers)
        self.frontier_pub.publish(frontier_marker_array)
        frontier_marker_array.markers = []
        for frontier_point in self.global_frontier_points:
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
            point.x, point.y = frontier_point[0], frontier_point[1]
            marker.points.append(point)
            frontier_marker_array.markers.append(marker)
            idx += 1
        self.frontier_pub.publish(frontier_marker_array)        
    
    def task_allocator_timer_cb(self):
        # Remove invalid frontier (no longer a frontier), check the map value is still unexplored
        self.local_frontier_points = [frontier for frontier in self.local_frontier_points if self.map_data[real_pose_to_map_grid_pos(frontier[0], frontier[1], self.map.info.origin, self.map.info.resolution)[::-1]] == CellType.UNKNOWN]
        #self.global_frontier_points = [frontier for frontier in self.global_frontier_points if self.map_data[real_pose_to_map_grid_pos(frontier[0], frontier[1], self.map.info.origin, self.map.info.resolution)[::-1]] == CellType.UNKNOWN]

        # Filter: Mean shift clustering to find the goal position
        if len(self.global_frontier_points) == 0 and len(self.local_frontier_points) == 0:
            return
        clustered_frontiers = mean_shift_clustering(self.global_frontier_points + self.local_frontier_points, 1.0, max_iteration=200, tolerance=5e-3)

        # State routine and transition
        if self.state == ExplorationStatus.ROTATING_SCAN:
            if self.rotate_scan_time >= self.rotate_scan_duration:
                self.state = ExplorationStatus.ROTATING_SCAN_DONE
                self.rotate_scan_time = 0.0
                self.get_logger().info("Rotating scan done")
            else:
                self.rotate_scan_time += 0.2
                rotate_cmd = Twist()
                rotate_cmd.angular.z = 0.8
                self.vel_pub.publish(rotate_cmd)
        elif self.state == ExplorationStatus.ROTATING_SCAN_DONE:
            if len(clustered_frontiers) == 0:
                return
            # Score the frontiers and select the best one
            best_score = float('-inf')
            best_frontier = None
            for clustered_frontier in clustered_frontiers:
                grid_x_idx, grid_y_idx = real_pose_to_map_grid_pos(clustered_frontier[0], clustered_frontier[1], self.map.info.origin, self.map.info.resolution)
                # info score is the number of unknown cells within a certain radius r
                info_score = count_pixels_with_value_around_point(self.map_data, grid_x_idx, grid_y_idx, 10, CellType.UNKNOWN)
                energy_cost = np.sqrt((clustered_frontier[0] - self.robot_current_position[0])**2 + (clustered_frontier[1] - self.robot_current_position[1])**2)
                c1 = 1.0 if energy_cost > 0.24 else 2.0
                score = c1 * info_score - 1.0 * energy_cost
                if score > best_score:
                    best_score = score
                    best_frontier = clustered_frontier
            self.current_goal = best_frontier
            self.get_logger().info(f"Best frontier: {best_frontier}")
            self.state = ExplorationStatus.NAVIGATE_TO_FRONTIER
        elif self.state == ExplorationStatus.NAVIGATE_TO_FRONTIER:
            # Navigate to the goal
            # TODO: Check if the goal is reached
            if np.sqrt((self.robot_current_position[0] - self.current_goal[0])**2 + (self.robot_current_position[1] - self.current_goal[1])**2) < 0.07:
                self.get_logger().info("Goal reached")
                self.state = ExplorationStatus.ROTATING_SCAN
                self.rotate_scan_time = 0.0
                self.current_goal = None
            # TODO: Select the best frontier as above
            pass
        else:
            self.get_logger().info("Invalid state")
    
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
            self.RRT.insert_node(RRTNode(*self.robot_current_position), None)
        else:
            self.robot_current_position = (self.robot_pose_trans[0, 3], self.robot_pose_trans[1, 3])

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
