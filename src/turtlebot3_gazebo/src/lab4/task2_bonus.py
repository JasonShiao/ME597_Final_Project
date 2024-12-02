#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Import other python packages that you think necessary
from nav_msgs.msg import OccupancyGrid, Path
from tf2_msgs.msg import TFMessage

import cv2
import numpy as np
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform
from geometry_msgs.msg import PoseStamped, Twist
from enum import IntEnum
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sklearn.cluster import MeanShift
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from copy import copy
import os

import casadi as ca # For MPC
from ament_index_python.packages import get_package_share_directory

# ===================== Helper Functions =====================
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

def SE3toPose(T):
    pose = Pose()
    pose.position.x = T[0, 3]
    pose.position.y = T[1, 3]
    pose.position.z = T[2, 3]
    
    r = Rotation.from_matrix(T[0:3, 0:3])
    quat = r.as_quat()
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose

# ===================== Map Processor ========================
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from graphviz import Graph
from nav_msgs.msg import MapMetaData
import yaml
import pandas as pd
from graphviz import Graph

class TreeNode():
    def __init__(self,name, group_id):
        self.name = name
        self.children = []
        self.weight = []
        self.group_id = group_id

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)


class Tree():
    def __init__(self):
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')

    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            # Connect to all direct children from the current node
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if start:
            self.root = node.name
        elif end:
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True


class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        #map_name = map_df.image[0]
        map_name = map_name + '.pgm' # Preserve the entire path
        im = Image.open(map_name)
        #size = 200, 200
        #im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array
    
    def get_real_world_pos(self, y_pixel_idx, x_pixel_idx):
        # Note: the pixel is in (y,x) and y-axis from top to bottom
        return self.limits[0] + x_pixel_idx * self.map_df.resolution[0], self.limits[3] - y_pixel_idx * self.map_df.resolution[0]

    def get_closest_pixel(self, x_pos, y_pos):
        # Note: the pixel is in (y,x) and y-axis from top to bottom
        return round((self.limits[3] - y_pos) / self.map_df.resolution[0]), round((x_pos - self.limits[0]) / self.map_df.resolution[0])

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree()

        # Change these initialization if needed
        kr = self.rect_kernel(6,1) # Inflate a safety margin for turtlebot ( > 0.2m / 0.05)
        self.inflate_map(kr,True)
        self.get_graph_from_map()

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = TreeNode('%d,%d'%(i,j), i * self.map.image_array.shape[1] + j)
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]): # y, from up to down
            for j in range(self.map.image_array.shape[1]): # x, from left to right
                if self.inf_map_img_array[i][j] == 0: # Free space
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            child_up.group_id = self.map_graph.g['%d,%d'%(i,j)].group_id
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            child_dw.group_id = self.map_graph.g['%d,%d'%(i,j)].group_id
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            child_lf.group_id = self.map_graph.g['%d,%d'%(i,j)].group_id
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            child_rg.group_id = self.map_graph.g['%d,%d'%(i,j)].group_id
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            child_up_lf.group_id = self.map_graph.g['%d,%d'%(i,j)].group_id
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            child_up_rg.group_id = self.map_graph.g['%d,%d'%(i,j)].group_id
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            child_dw_lf.group_id = self.map_graph.g['%d,%d'%(i,j)].group_id
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            child_dw_rg.group_id = self.map_graph.g['%d,%d'%(i,j)].group_id
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m


# ====================== Path Planner (RRT*) ========================
class RRTStarNode():
    def __init__(self, x, y):
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.cost = 0
        self.parent = None
    
    def position(self):
        return (self.x, self.y)

class RRTStar():
    def __init__(self, x, y, map_processor: MapProcessor):
        self.map_processor = map_processor
        self.eta = 2.0 # Step size
        root_node = RRTStarNode(x, y)
        root_node.parent = None
        self.nodes = {root_node.position(): root_node}
        self.tree = KDTree(list(self.nodes.keys()))
        self.edges = [] # list of [start_x, start_y, end_x, end_y]

    def _insert_node(self, node, parent):
        print("Inserting node at position: ", node.position())
        if node.position() in self.nodes:
            print("Node already exists!")
            return
        self.nodes[node.position()] = node
        self.tree = KDTree(list(self.nodes.keys()))
        if parent:
            self.edges.append([parent.x, parent.y, node.x, node.y])
            node.parent = parent

    def _sample_free(self):
        # TODO: Sample from free space
        # get corresponding map grid idx of the root node
        root_node_grid_y, root_node_grid_x = self.map_processor.map.get_closest_pixel(*(self.nodes[list(self.nodes.keys())[0]].position()))
        group_id = self.map_processor.map_graph.g['%d,%d'%(root_node_grid_y, root_node_grid_x)].group_id
        # Randomly sample a node from the same group_id
        free_nodes = [node for node in self.map_processor.map_graph.g.values() if node.group_id == group_id]
        random_node = free_nodes[np.random.randint(len(free_nodes))]
        return self.map_processor.map.get_real_world_pos(int(random_node.name.split(',')[0]), int(random_node.name.split(',')[1]))

    def _nearest_neighbor(self, ref_pos):
        dist, index = self.tree.query([ref_pos[0], ref_pos[1]])
        return self.nodes[list(self.nodes.keys())[index]]

    def _near(self, center_pos, radius):
        indices = self.tree.query_ball_point([center_pos[0], center_pos[1]], radius)
        return [self.nodes[list(self.nodes.keys())[i]] for i in indices]

    def _obstacle_free(self, start, target):
        x_i, y_i = start[0], start[1]
        fine_grain_step_size = self.map_processor.map.map_df.resolution[0] * 0.2
        step_count = int(np.ceil(np.sqrt((target[0] - x_i)**2 + (target[1] - y_i)**2) / fine_grain_step_size))
        found_obstacle = False
        for i in range(step_count):
            x_i, y_i = self._steer((x_i, y_i), target, fine_grain_step_size)
            x_i_grid, y_i_grid = self.map_processor.map.get_closest_pixel(x_i, y_i)
            if y_i_grid < 0 or y_i_grid >= self.map_processor.map.image_array.shape[0] or x_i_grid < 0 or x_i_grid >= self.map_processor.map.image_array.shape[1]:
                found_obstacle = True
                break
            elif self.map_processor.inf_map_img_array[y_i_grid, x_i_grid] == 0:
                found_obstacle = True
                print(f"x_i_grid: {x_i_grid}, y_i: {y_i_grid}")
                break
        
        return not found_obstacle

    def _collision_free(self, start, target, safe_margin=0.5):
        x_i, y_i = start[0], start[1]
        fine_grain_step_size = self.map_processor.map.map_df.resolution[0] * 0.2
        step_count = int(np.ceil(np.sqrt((target[0] - x_i)**2 + (target[1] - y_i)**2) / fine_grain_step_size))
        found_collision = False
        for i in range(step_count):
            x_i, y_i = self._steer((x_i, y_i), target, fine_grain_step_size)
            x_i_grid, y_i_grid = self.map_processor.map.get_closest_pixel(x_i, y_i)
            # Get nearby pixels with safe_margin
            x_i_grid_min = x_i_grid - safe_margin / self.map_processor.map.map_df.resolution[0]
            x_i_grid_max = x_i_grid + safe_margin / self.map_processor.map.map_df.resolution[0]
            y_i_grid_min = y_i_grid - safe_margin / self.map_processor.map.map_df.resolution[0]
            y_i_grid_max = y_i_grid + safe_margin / self.map_processor.map.map_df.resolution[0]
            # Check if any of the nearby pixels is out of the map
            if x_i_grid_min < 0 or x_i_grid_max >= self.map_processor.map.image_array.shape[1] or y_i_grid_min < 0 or y_i_grid_max >= self.map_processor.map.image_array.shape[0]:
                found_collision = True
                break
            # Check if any of the nearby pixels is an obstacle
            if np.any(self.map_processor.inf_map_img_array[y_i_grid_min:y_i_grid_max+1, x_i_grid_min:x_i_grid_max+1] == 0):
                found_collision = True
                break
        
        return not found_collision

    def _steer(self, start, target, step_size):
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        # move with step size along the direction from node to target_point
        dist = np.sqrt(dx**2 + dy**2)
        if dist > step_size:
            dx = dx / dist * step_size
            dy = dy / dist * step_size
        else: # dist smaller than step size
            return target
        return (start[0] + dx, start[1] + dy)
    
    def explore(self, iterations=100, gamma=50.0):
        for _ in range(iterations):
            x_rand = self._sample_free()
            node_nearest = self._nearest_neighbor(x_rand)
            x_new = self._steer(node_nearest.position(), x_rand, self.eta)
            if self._obstacle_free(node_nearest.position(), x_new):
                Nodes_near = self._near(x_new, min(gamma * np.sqrt(np.log(len(self.nodes)) / len(self.nodes)), self.eta))
                x_min = node_nearest.position()
                # TODO: Add other cost function? (e.g. cost = distance + distance from obstacles)
                c_min = node_nearest.cost + np.sqrt((x_new[0] - node_nearest.x)**2 + (x_new[1] - node_nearest.y)**2)
                for node_near in Nodes_near:
                    if self._collision_free(node_near.position(), x_new):
                        c_near = self.nodes[node_near.position()].cost + np.sqrt((x_new[0] - node_near.x)**2 + (x_new[1] - node_near.y)**2)
                        if c_near < c_min:
                            x_min = node_near.position()
                            c_min = c_near
                    else:
                        print(f"x_near: {node_near.position()}, x_new: {x_new}")
                        print("Collision detected between near node and new node!")
                node_new = RRTStarNode(x_new[0], x_new[1])
                node_new.cost = c_min
                self._insert_node(node_new, self.nodes[x_min])
                for node_near in Nodes_near:
                    if self._collision_free(x_new, node_near.position()):
                        c_near = self.nodes[x_new].cost + np.sqrt((node_near.x - x_new[0])**2 + (node_near.y - x_new[1])**2)
                        if c_near < self.nodes[node_near.position()].cost:
                            # remove the edge from the old parent TO node_near
                            if [self.nodes[node_near.position()].parent.x, self.nodes[node_near.position()].parent.y, node_near.x, node_near.y] in self.edges:
                                self.edges.remove([self.nodes[node_near.position()].parent.x, self.nodes[node_near.position()].parent.y, node_near.x, node_near.y])
                            self.nodes[node_near.position()].parent = node_new
                            self.nodes[node_near.position()].cost = c_near
                            self.edges.append([x_new[0], x_new[1], node_near.x, node_near.y])
                    else:
                        print(f"x_new: {x_new}, x_near: {node_near.position()}")
                        print("Collision detected between new node and near node!")
            else:
                print(f"x_nearest: {node_nearest.position()}, x_new: {x_new}")
                print("Obstacle detected between nearest node and new node!")

    def visualize(self):
        # Draw the nodes and edges for debug
        #fig, ax = plt.subplots(dpi=100)
        #plt.imshow(self.map_processor.inf_map_img_array)
        #plt.colorbar()
        #plt.pause(0.5)
        #plt.show(block=False)
        fig, ax = plt.subplots(dpi=150)
        plt.imshow(self.map_processor.map.image_array, extent=self.map_processor.map.limits, cmap=cm.gray)
        for node in self.nodes.values():
            ax.plot(node.x, node.y, 'ro')
        for edge in self.edges:
            ax.plot([edge[0], edge[2]], [edge[1], edge[3]], 'r-')
        ax.plot()
        plt.show()
        

# ================ Path Following Controller (MPC) ==================



# ========================= Navigator ===============================



class Task2(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self):
        super().__init__('task2_node')

        self.odom_trans = np.eye(4)       # map (world) frame to odom frame
        self.robot_base_trans = np.eye(4) # odom frame to robot base frame
        self.robot_pose_trans = np.eye(4) # map (world) frame to robot base frame

        self.robot_current_position = None # in real world coord

        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need
        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )

        package_dir = get_package_share_directory('turtlebot3_gazebo')
        map_file_path = os.path.join(package_dir, 'maps', 'map_2') # NOTE: .yaml suffix is handled within Map()
        self.map_processor = MapProcessor(map_file_path)
        # Print all graph nodes with their group id
        for node in self.map_processor.map_graph.g.values():
            print(node, node.group_id)
        # Test RRT* algorithm
        self.RRT = RRTStar(0, 0, self.map_processor)
        self.RRT.explore(10)
        for node in self.RRT.nodes.values():
            print(node.position(), node.cost)
        self.RRT.visualize()

    def timer_cb(self):
        self.get_logger().info('Task2 node is alive.', throttle_duration_sec=1)
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

        if not self.robot_current_position: # initialize last position and RRT
            self.robot_current_position = (self.robot_pose_trans[0, 3], self.robot_pose_trans[1, 3])
            #self.RRT.insert_node(RRTNode(*self.robot_current_position), None)
        else:
            self.robot_current_position = (self.robot_pose_trans[0, 3], self.robot_pose_trans[1, 3])


def main(args=None):
    rclpy.init(args=args)

    task2 = Task2()

    try:
        rclpy.spin(task2)
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
