#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Import other python packages that you think necessary
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import yaml
import pandas as pd
from copy import copy
from graphviz import Graph

import casadi as ca

from enum import IntEnum
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from ament_index_python.packages import get_package_share_directory

import os
import cv2
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Path
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation

# =================== Helper Functions ==================
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

# =================== Map Processor ==================
class TreeNode():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)


class Tree():
    def __init__(self,name):
        self.name = name
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
    def __init__(self,name, current_pos):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)
        self.current_pos = current_pos

        # Change these initialization if needed
        kr = self.rect_kernel(8,1) # Inflate a safety margin for turtlebot ( > 0.2m / 0.05)
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

        # Directly set 0 around the current position with radius 5 pixels
        y_pixel_idx, x_pixel_idx = self.map.get_closest_pixel(self.current_pos[0], self.current_pos[1])
        erode_radius = 4
        for i in range(y_pixel_idx - erode_radius, y_pixel_idx + erode_radius):
            for j in range(x_pixel_idx - erode_radius, x_pixel_idx + erode_radius):
                if (i >= 0) and (i < self.map.image_array.shape[0]) and (j >= 0) and (j < self.map.image_array.shape[1]):
                    self.inf_map_img_array[i][j] = 0
        
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = TreeNode('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]): # y, from up to down
            for j in range(self.map.image_array.shape[1]): # x, from left to right
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
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

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array
    
# ======================== Path Planner (A*) ========================
class AStar():
    def __init__(self,in_tree: Tree):
        self.in_tree: Tree = in_tree
        self.open_nodes = {} # key: node name, value: (priority, node)
        #self.q = PriorityQueue()
        self.dist = {} # g: cost so far (from start)
        self.h = {} # h: cost to goal
        self.via = {name:0 for name,node in in_tree.g.items()}
        self.processed_nodes = set() # set of node names those shortest path have been determined

    def __get_f_score(self,node):
        # f = g + h
        return self.dist[node.name] + self.h[node.name]

    def solve(self, start_node: TreeNode, end_node: TreeNode, visualize = False) -> bool:
        # Clean up for each solution
        self.open_nodes = {}
        self.dist = {}
        self.h = {}
        self.processed_nodes = set()

        # Calculate h value for each node
        for name, node in self.in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            goal = tuple(map(int, end_node.name.split(',')))
            # With h, we add a tendency/force (heuristic) to the end position
            self.h[name] = np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)

        self.via = {name:0 for name,node in self.in_tree.g.items()}
        # Set dist (g) for starting node
        self.dist[start_node.name] = 0
        self.open_nodes[start_node.name] = (self.__get_f_score(start_node), start_node)

        if visualize:
            plot_x_pos = []
            plot_y_pos = []
            plot_f_val = []
        
        solved = False
        while len(self.open_nodes) > 0:
            min_node_name = min(self.open_nodes, key=lambda k: self.open_nodes[k][0])
            priority, current_node = self.open_nodes[min_node_name]
            del self.open_nodes[min_node_name]

            # Skip if the node has been processed
            if current_node.name in self.processed_nodes:
                continue
            self.processed_nodes.add(current_node.name)

            #print(f"priority: {priority}, node: {current_node.name}")
            if visualize:
                plot_x_pos.append(tuple(map(int, current_node.name.split(',')))[1])
                plot_y_pos.append(tuple(map(int, current_node.name.split(',')))[0])
                plot_f_val.append(priority)
            
            if current_node.name == end_node.name:
                solved = True
                break
            for i in range(len(current_node.children)):
                child = current_node.children[i]
                w = current_node.weight[i] # distance from current node to child node
                new_dist = self.dist[current_node.name] + w
                # NOTE: use dist INSTEAD OF priority to determine whether we should update child node
                if child.name not in self.processed_nodes and (child.name not in self.dist or new_dist < self.dist[child.name]):
                    self.dist[child.name] = new_dist
                    self.via[child.name] = current_node.name
                    self.open_nodes[child.name] = (self.__get_f_score(child), child)
        
        # Plot for debug
        if visualize:
            grid_size_x = max(plot_x_pos)+1
            grid_size_y = max(plot_y_pos)+1
            grid = np.full((grid_size_y, grid_size_x), np.nan)
            for xi, yi, vi in zip(plot_x_pos, plot_y_pos, plot_f_val):
                grid[yi,xi] = vi
            plt.figure()
            plt.pcolormesh(grid, cmap='viridis', shading='auto')
            plt.gca().invert_yaxis()
            plt.colorbar(label='value')
            plt.show()
        
        return solved
    

    def reconstruct_path(self,sn,en):
        path = []
        dist = 0
        # Place code here
        current_node_key = self.via[en.name]
        path.append(en.name)
        dist = self.dist[en.name]
        while current_node_key != sn.name:
            path.append(current_node_key)
            current_node_key = self.via[current_node_key]
        path.append(sn.name)
        path.reverse()
        return path, dist

# ======================== Waypoint Reduction ========================
def waypoint_reduction(path_poses: list, epsilon):
    # Douglas-Peucker algorithm
    # Recurssion

    if len(path_poses) <= 2:
        return path_poses
    
    start = path_poses[0].pose.position
    end = path_poses[-1].pose.position
    max_dist = 0
    max_idx = -1
    for i in range(1, len(path_poses)-1):
        numerator = abs((end.y - start.y) * path_poses[i].pose.position.x - (end.x - start.x) * path_poses[i].pose.position.y + end.x * start.y - end.y * start.x)
        denominator = np.sqrt((end.y - start.y)**2 + (end.x - start.x)**2)
        dist = numerator / denominator
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist < epsilon: # All fall close to the line, remove all middle points
        return [path_poses[0], path_poses[-1]]
    
    # Divide at max idx
    path_poses_1 = waypoint_reduction(path_poses[:max_idx+1], epsilon)
    path_poses_2 = waypoint_reduction(path_poses[max_idx+1:], epsilon)
    merge = path_poses_1[:-1] + path_poses_2
    return merge

# ================= Path Following Controller (MPC) =================
class DifferentialDriveMPC:
    def __init__(self, dt=0.2, N=20, wheel_base=0.288, v_max=0.35, omega_max=0.5):
        self.dt = dt
        self.N = N  # Prediction horizon
        self.L = wheel_base
        self.v_max = v_max
        self.omega_max = omega_max

        # Create Opti object for optimization
        self.opti = ca.Opti()

        # Define optimization variables: state trajectory [x, y, theta]
        self.X = self.opti.variable(3, self.N + 1)  # 3 states: [x, y, theta]
        self.U = self.opti.variable(2, self.N)  # 2 inputs: [v, omega]

    def setup_mpc(self, start, goal, last_goal):
        """
        Set up the MPC problem, including objective and constraints.
        """
        # Initialize the objective function
        self.opti = ca.Opti()
        # Define optimization variables: state trajectory [x, y, theta]
        self.X = self.opti.variable(3, self.N + 1)  # 3 states: [x, y, theta]
        self.U = self.opti.variable(2, self.N)  # 2 inputs: [v, omega]
        objective = 0

        # Weights for state and control costs
        Q = np.diag([100, 100, 0])  # State error weights
        R = np.diag([1, 1])  # Control effort weights
        S = 100 # Straight path line deviation weights

        start = start.reshape((3, 1))  # Shape (3, 1)

        # Add the initial state constraint
        self.opti.subject_to(self.X[:, 0] == start)

        # Loop over the prediction horizon to add dynamics and input constraints
        for t in range(self.N):
            # State error cost
            state_error = self.X[:, t] - goal
            objective += ca.mtimes([state_error.T, Q, state_error])

            # Control effort cost
            control_effort = self.U[:, t]
            objective += ca.mtimes([control_effort.T, R, control_effort])
            # self.opti.minimize(objective)

            # Path line deviation cost
            numerator = ((goal[1] - last_goal[1]) * self.X[0,t] - (goal[0] - last_goal[0]) * self.X[1,t] + goal[0] * last_goal[1] - goal[1] * last_goal[0])**2
            denominator = np.sqrt((goal[1] - last_goal[1])**2 + (goal[0] - last_goal[0])**2)
            path_line_deviation = numerator / denominator
            objective += S * path_line_deviation

            # Dynamics constraint: X[t+1] = X[t] + f(X[t], U[t]) * dt
            next_state = self.X[:, t] + self.robot_dynamics(self.X[:, t], self.U[:, t]) * self.dt
            self.opti.subject_to(self.X[:, t + 1] == next_state)

            # Input constraints (inequality constraints)
            self.opti.subject_to(self.U[0, t] <= self.v_max)  # v <= v_max
            self.opti.subject_to(self.U[0, t] >= -self.v_max)  # v >= -v_max
            self.opti.subject_to(self.U[1, t] <= self.v_max)  # omega <= omega_max
            self.opti.subject_to(self.U[1, t] >= -self.v_max)  # omega >= -mega_max

            self.opti.minimize(objective)

        # Set the objective function to minimize
        self.opti.minimize(objective)


    def robot_dynamics(self, state, control):
        """
        Nonlinear dynamics of the differential drive robot.
        """
        x, y, theta = state[0], state[1], state[2]

        v= (control[0] + control[1]) / 2
        omega= (-control[0] + control[1]) / self.L

        # Compute state derivatives
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = omega

        return ca.vertcat(dx, dy, dtheta)

    def solve_mpc(self, start, goal, last_goal):
        """
        Solve the MPC optimization problem.
        """
        # Set up the MPC problem
        self.setup_mpc(start, goal, last_goal)

        # Provide an initial guess for the solver
        self.opti.set_initial(self.X, np.tile(start, (self.N + 1, 1)).T)
        self.opti.set_initial(self.U, np.zeros((2, self.N)))


        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}
        self.opti.solver('ipopt', opts)

        try:
            # Solve the problem
            solution = self.opti.solve()

            # Extract the optimal control inputs and state trajectory
            opt_U = solution.value(self.U)
            opt_X = solution.value(self.X)

            return opt_U[:, 0], opt_X

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            print("Debugging with opti.debug.value:")
            print("X:", self.opti.debug.value(self.X))
            print("U:", self.opti.debug.value(self.U))
            raise




# ============================ Navigator ============================
class NavigationStatus(IntEnum):
    IDLE = 0
    NAVIGATING = 1

class Navigator():
    def __init__(self):
        self.state = NavigationStatus.IDLE
        self.current_goal_ = None
        self.map_processor_ = None
        self.path = Path() # path with waypoints
        self.path.header.frame_id = 'map'

    def send_goal(self, current_pos, goal):
        self.current_goal_ = goal
        # Initialize map processor
        package_dir = get_package_share_directory('turtlebot3_gazebo')
        map_file_path = os.path.join(package_dir, 'maps', 'map_2') # NOTE: .yaml suffix is handled within Map()
        self.map_processor_ = MapProcessor(map_file_path, current_pos)
        # Visualize for debug
        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots(dpi=100)
        #plt.imshow(self.map_processor_.inf_map_img_array)
        #plt.colorbar()
        #plt.pause(0.5)
        #plt.show(block=False)

        # A* Path planning
        a_star_planner = AStar(self.map_processor_.map_graph)
        self.path.poses = []
        start_y_idx, start_x_idx = self.map_processor_.map.get_closest_pixel(current_pos[0], current_pos[1])
        end_y_idx, end_x_idx = self.map_processor_.map.get_closest_pixel(goal[0], goal[1])
        if (f'{start_y_idx},{start_x_idx}' not in self.map_processor_.map_graph.g) or (f'{end_y_idx},{end_x_idx}' not in self.map_processor_.map_graph.g):
            if (f'{start_y_idx},{start_x_idx}' not in self.map_processor_.map_graph.g):
                print(f"Start point {start_y_idx},{start_x_idx} not in the graph")
            if (f'{end_y_idx},{end_x_idx}' not in self.map_processor_.map_graph.g):
                print(f"End point {end_y_idx},{end_x_idx} not in the graph")
            print("Failed to find path")
            return False
        
        solved = a_star_planner.solve(self.map_processor_.map_graph.g[f'{start_y_idx},{start_x_idx}'], self.map_processor_.map_graph.g[f'{end_y_idx},{end_x_idx}'], visualize=False)
        if not solved:
            print("Failed to solve A*")
            return False
        
        #print(planner.dist)
        raw_path, dist = a_star_planner.reconstruct_path(self.map_processor_.map_graph.g[f'{start_y_idx},{start_x_idx}'], self.map_processor_.map_graph.g[f'{end_y_idx},{end_x_idx}'])
        for node in raw_path:
            y_pixel_idx, x_pixel_idx = map(int, node.split(','))
            x_real_pos, y_real_pos = self.map_processor_.map.get_real_world_pos(y_pixel_idx, x_pixel_idx)
            pose = PoseStamped()
            pose.header.frame_id = self.path.header.frame_id
            pose.header.stamp = self.path.header.stamp
            pose.pose.position.x = x_real_pos
            pose.pose.position.y = y_real_pos
            self.path.poses.append(pose)
        # Waypoint reduction
        self.path.poses = waypoint_reduction(self.path.poses, 0.02)
        return True

    def path_following(self, vehicle_pose_trans, target_waypoint_pose, last_waypoint_pose):
        # MPC controller
        # return Twist msg for cmd_vel or None if goal reached
        # Convert from SE3 (4x4 matrix) to Pose
        vehicle_pose = Pose()
        vehicle_pose.position.x = vehicle_pose_trans[0, 3]
        vehicle_pose.position.y = vehicle_pose_trans[1, 3]
        vehicle_pose.orientation.x, vehicle_pose.orientation.y, vehicle_pose.orientation.z, vehicle_pose.orientation.w = Rotation.from_matrix(vehicle_pose_trans[0:3, 0:3]).as_quat()
        current_orientation = np.arctan2(vehicle_pose.orientation.z, vehicle_pose.orientation.w)*2
        #current_orientation = TODO...
        start = np.array([vehicle_pose.position.x, vehicle_pose.position.y, current_orientation])
        # NOTE: MUST use arctan2 instead of arctan!
        #goal_orientation = TODO...
        # WARNING: TODO: Handle [-pi, pi] wrap-up problem for orientation
        goal = np.array([target_waypoint_pose.pose.position.x, target_waypoint_pose.pose.position.y, 0.0])

        # Used to generate path line from last waypoint to the next waypoint
        last_goal = np.array([last_waypoint_pose.pose.position.x, last_waypoint_pose.pose.position.y, 0])
        
        # Initialize the MPC controller
        mpc = DifferentialDriveMPC()

        print(f"start: {start}, goal: {goal}, last_goal: {last_goal}")

        state = start
        # Solve MPC to get the optimal control input
        u_opt, pred_trajectory = mpc.solve_mpc(state, goal, last_goal)
        v = (u_opt[0] + u_opt[1]) / 2
        omega= (-u_opt[0] + u_opt[1]) / mpc.L
        # Update state based on dynamics
        state = mpc.robot_dynamics(state, u_opt).full().flatten() * mpc.dt + state

        #print(f"u_opt: {u_opt}")
        #print(f"v: {v}, omega: {omega}")

        speed = v
        heading = omega
        return speed, heading




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
        self.current_goal = None # get by clicking on the map in rViz

        self.navigator = Navigator()

        # Subscriber / Publisher / Timer
        # Fill in the initialization member variables that you need
        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_pose_cbk, 10)
        self.task_allocator_timer = self.create_timer(0.2, self.task_allocator_timer_cb)
        self.path_follow_timer = self.create_timer(0.1, self.path_follow_timer_cb)

        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/waypoint_path', 10)

    def goal_pose_cbk(self, pose_msg: PoseStamped):
        self.current_goal = pose_msg.pose.position.x, pose_msg.pose.position.y

    def task_allocator_timer_cb(self):
        if self.navigator.state == NavigationStatus.IDLE:
            # Stop the robot
            idle_twist = Twist()
            idle_twist.linear.x = 0.
            idle_twist.angular.z = 0.
            self.vel_pub.publish(idle_twist)

            if self.current_goal is not None:
                self.get_logger().info(f"New goal: {self.current_goal}")
                # Send goal to navigator
                if self.navigator.send_goal(self.robot_current_position, self.current_goal):
                    self.navigator.state = NavigationStatus.NAVIGATING
                    
                    self.path_pub.publish(self.navigator.path)
                else:
                    self.get_logger().info("Failed to send goal to navigator")
                    self.current_goal = None
        elif self.navigator.state == NavigationStatus.NAVIGATING:
            # TODO
            # if finished,
            if self.current_goal == None:
                self.navigator.path.poses = []
                self.current_goal = None
                self.navigator.state = NavigationStatus.IDLE
                self.get_logger().info("Goal reached")
                return
            # self.current_goal = None
            # self.navigator.state = NavigationStatus.IDLE

    def path_follow_timer_cb(self):
        self.get_logger().info('Task2 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function
        if self.navigator.state != NavigationStatus.NAVIGATING:
            return
        if self.current_goal is None or len(self.navigator.path.poses) <= 1:
            return
        # TODO:
        tolerance = 0.06
        if np.sqrt((self.robot_current_position[0] - self.navigator.path.poses[1].pose.position.x)**2 + (self.robot_current_position[1] - self.navigator.path.poses[1].pose.position.y)**2) < tolerance:
            self.navigator.path.poses.pop(0)
            self.get_logger().info(f"Waypoint reached, {len(self.navigator.path.poses)} waypoints left")
        if len(self.navigator.path.poses) <= 1:
            self.navigator.path.poses = []
            self.current_goal = None
            self.navigator.state = NavigationStatus.IDLE
            self.get_logger().info("Goal reached")
            return
        # Follow the path (heading to waypoint[1] from waypoint[0])
        vx, omega = self.navigator.path_following(self.robot_pose_trans, self.navigator.path.poses[1], self.navigator.path.poses[0])
        self.get_logger().info(f"vx: {vx}, omega: {omega}")
        cmd_vel = Twist()
        cmd_vel.linear.x = vx
        cmd_vel.angular.z = omega
        self.vel_pub.publish(cmd_vel)
    
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