#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
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
from copy import copy

import casadi as ca # For MPC

# ===================== Map Processor ========================
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from graphviz import Graph
from nav_msgs.msg import MapMetaData
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
    def __init__(self, map_data, map_info: MapMetaData):
        self.map_im = map_data
        self.map_info = map_info
        self.limits = [map_info.origin.position.x, map_info.origin.position.x + map_info.width * map_info.resolution, map_info.origin.position.y, map_info.origin.position.y + map_info.height * map_info.resolution]
        self.image_array = self.__get_obstacle_map(self.map_im)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __get_obstacle_map(self, map_im):
        img_array = np.zeros_like(map_im, dtype=np.uint8)
        img_array[map_im == 100] = 255   # OCCUPIED -> black
        return img_array
    
    def get_real_world_pos(self, y_pixel_idx, x_pixel_idx):
        # Note: the pixel is in (y,x) and y-axis from top to bottom
        return self.limits[0] + x_pixel_idx * self.map_info.resolution, self.limits[2] + y_pixel_idx * self.map_info.resolution

    def get_closest_pixel(self, x_pos, y_pos):
        # Note: the pixel is in (y,x) and y-axis from top to bottom
        return round((y_pos - self.limits[2]) / self.map_info.resolution), round((x_pos - self.limits[0]) / self.map_info.resolution)

class MapProcessor():
    def __init__(self, map_data, map_info: MapMetaData):
        self.map = Map(map_data, map_info)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree()

        # Change these initialization if needed
        kr = self.rect_kernel(11,1) # Inflate a safety margin for turtlebot ( > 0.2m / 0.05)
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
                if self.map.image_array[i][j] == 255:
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

# ============================== Path Planner ================================
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
        print(f"via: {self.via}, sn: {sn.name}, en: {en.name}")
        current_node_key = self.via[en.name]
        path.append(en.name)
        dist = self.dist[en.name]
        while current_node_key != sn.name:
            path.append(current_node_key)
            current_node_key = self.via[current_node_key]
        path.append(sn.name)
        path.reverse()
        return path, dist

# ======================== Path Following Controller ========================
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time

class DifferentialDriveMPC:
    def __init__(self, dt=0.1, N=20, wheel_base=0.288, v_max=0.25, omega_max=0.5):
        """
        Initialize the MPC controller for a differential drive robot.
        
        Args:
            dt: Time step duration
            N: Prediction horizon
            wheel_base: Distance between the two wheels
            v_max: Maximum linear velocity
            omega_max: Maximum angular velocity    
        """
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
        R = np.diag([1.5, 1.5])  # Control effort weights
        S = 100 # Straight path line deviation weights

        start = start.reshape((3, 1))  # Shape (3, 1)

        # Add the initial state constraint
        self.opti.subject_to(self.X[:, 0] == start)

        # Loop over the prediction horizon to add dynamics and input constraints
        for t in range(self.N):
            # State error cost
            state_error = self.X[:, t] - goal
            objective += ca.mtimes([state_error.T, Q, state_error]) # i.e. err_x^2 + err_y^2 + err_theta^2

            # Control effort cost
            control_effort = self.U[:, t]
            objective += ca.mtimes([control_effort.T, R, control_effort])
            # self.opti.minimize(objective)

            # Path line deviation cost
            numerator = ((goal[1] - last_goal[1]) * self.X[0,t] - (goal[0] - last_goal[0]) * self.X[1,t] + goal[0] * last_goal[1] - goal[1] * last_goal[0])**2
            denominator = np.sqrt((goal[1] - last_goal[1])**2 + (goal[0] - last_goal[0])**2)
            if denominator < 1e-6:
                path_line_deviation = 0
            else:
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

# ================================ Navigator =================================
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

class Navigator():
    def __init__(self):
        self.current_goal_ = None
        self.map_processor_ = None
        self.path = Path() # path with waypoints
        self.path.header.frame_id = 'map'
    
    def send_goal(self, current_pos, goal, map_data, map_info):
        # if too close, don't change goal
        if self.current_goal_ and ((self.current_goal_[0] - goal[0])**2 + (self.current_goal_[1] - goal[1])**2) < 0.05**2:
            return False
        self.current_goal_ = goal
        # Initialize map processor
        self.map_processor_ = MapProcessor(map_data, map_info)
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
                pass
            if (f'{end_y_idx},{end_x_idx}' not in self.map_processor_.map_graph.g):
                pass
            return False
        
        solved = a_star_planner.solve(self.map_processor_.map_graph.g[f'{start_y_idx},{start_x_idx}'], self.map_processor_.map_graph.g[f'{end_y_idx},{end_x_idx}'], visualize=False)
        if not solved:
            #pass
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

        print(f"u_opt: {u_opt}")
        print(f"v: {v}, omega: {omega}")

        speed = v
        heading = omega
        return speed, heading



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

def mean_shift_clustering(points, bandwidth):
    centroids = []
    if len(points) > 3:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(points)
        centroids = ms.cluster_centers_
        centroids = [tuple(center) for center in centroids]
    else:
        centroids = points
        
    return centroids


def get_information_score(map_data, x, y, radius, target_val):
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
    # score: ratio of unknown cells in the circle
    return count / np.sum(circle_mask)

def check_satisfy_safety_margin(map_data, x, y, radius, target_val):
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
    if count > 0:
        return False
    return True

def extract_best_frontier_point(frontier_points, map_data, map_info, current_position):
    best_score = float('-inf')
    best_frontier = None
    for frontier in frontier_points:
        grid_x_idx, grid_y_idx = real_pose_to_map_grid_pos(frontier[0], frontier[1], map_info.origin, map_info.resolution)
        # info score is the number of unknown cells within a certain radius r
        info_score = get_information_score(map_data, grid_x_idx, grid_y_idx, 10, CellType.UNKNOWN)
        energy_cost = np.sqrt((frontier[0] - current_position[0])**2 + (frontier[1] - current_position[1])**2)
        c1 = 20.0 if energy_cost > 0.24 else 40.0
        score = c1 * info_score - 20.0 * energy_cost
        if score > best_score:
            best_score = score
            best_frontier = frontier
    
    return best_frontier

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
        self.frontier_explore_timer = self.create_timer(0.1, self.frontier_explore_timer_cb)
        self.global_frontier_explore_timer = self.create_timer(1, self.global_frontier_explore_timer_cb)
        self.filter_timer = self.create_timer(0.4, self.filter_timer_cb)
        self.replan_timer = self.create_timer(0.4, self.replan_timer_cb)
        self.task_allocator_timer = self.create_timer(0.2, self.task_allocator_timer_cb)
        self.path_follow_timer = self.create_timer(0.1, self.path_follow_timer_cb)
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
        self.goal_pub = self.create_publisher(Marker, '/goal', 10)
        self.path_pub = self.create_publisher(Path, '/waypoint_path', 10)

        self.odom_trans = np.eye(4)       # map (world) frame to odom frame
        self.robot_base_trans = np.eye(4) # odom frame to robot base frame
        self.robot_pose_trans = np.eye(4) # map (world) frame to robot base frame

        self.robot_current_position = None # in real world coord
        self.current_goal = None
        self.map: OccupancyGrid = None
        self.map_data: np.ndarray = None

        self.RRT = RRT(step_size=2.0) # default step size is 4m, TODO: Change according to map size
        self.local_frontier_points = []
        self.global_frontier_points = []
        self.clustered_frontiers = []

        self.state = ExplorationStatus.ROTATING_SCAN
        self.rotate_scan_time = 0.0
        self.rotate_scan_duration = 4.0
        self.map_updated = False

        self.navigator = Navigator()

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
        while iter < 2:
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

        self.map_updated = True
    
    def filter_timer_cb(self):
        # Remove invalid frontier (no longer a frontier), check the map value is still unexplored
        self.local_frontier_points = [frontier for frontier in self.local_frontier_points if self.map_data[real_pose_to_map_grid_pos(frontier[0], frontier[1], self.map.info.origin, self.map.info.resolution)[::-1]] == CellType.UNKNOWN]
        #self.global_frontier_points = [frontier for frontier in self.global_frontier_points if self.map_data[real_pose_to_map_grid_pos(frontier[0], frontier[1], self.map.info.origin, self.map.info.resolution)[::-1]] == CellType.UNKNOWN]

        # Filter: Mean shift clustering to find the goal position
        if len(self.global_frontier_points) == 0 and len(self.local_frontier_points) == 0:
            return
        self.clustered_frontiers = mean_shift_clustering(self.global_frontier_points + self.local_frontier_points, 0.4)
        self.global_frontier_points = []
        self.local_frontier_points = copy(self.clustered_frontiers) # Store the clustered frontiers as local frontiers
        
        # Ignore invalid frontier: occupied or very low information gain
        # TODO: Use threshold from cost map to filter
        ignore_idx = []
        for idx in range(len(self.clustered_frontiers)):
            grid_x_idx, grid_y_idx = real_pose_to_map_grid_pos(self.clustered_frontiers[idx][0], self.clustered_frontiers[idx][1], self.map.info.origin, self.map.info.resolution)
            # info score (gain) is the number of unknown cells within a certain radius r
            if get_information_score(self.map_data, grid_x_idx, grid_y_idx, 8, CellType.UNKNOWN) < 0.15:
                ignore_idx.append(idx)
            elif not check_satisfy_safety_margin(self.map_data, grid_x_idx, grid_y_idx, 5, CellType.OCCUPIED):
                ignore_idx.append(idx)
        self.clustered_frontiers = [frontier for idx, frontier in enumerate(self.clustered_frontiers) if idx not in ignore_idx]

    def replan_timer_cb(self):
        if self.state != ExplorationStatus.NAVIGATE_TO_FRONTIER:
            return
        # Check if the current goal is no longer a frontier
        # Check if the information score of the current goal is still high
        grid_x_idx, grid_y_idx = real_pose_to_map_grid_pos(self.current_goal[0], self.current_goal[1], self.map.info.origin, self.map.info.resolution)
        if get_information_score(self.map_data, grid_x_idx, grid_y_idx, 8, CellType.UNKNOWN) < 0.01:
            #self.get_logger().info("Replan: Current goal is no longer a frontier")
            self.state = ExplorationStatus.ROTATING_SCAN_DONE
            self.rotate_scan_time = 0.0
            self.current_goal = None
            return
    
    def task_allocator_timer_cb(self):
        # State routine and transition
        if self.state == ExplorationStatus.ROTATING_SCAN:
            if self.rotate_scan_time >= self.rotate_scan_duration:
                self.state = ExplorationStatus.ROTATING_SCAN_DONE
                self.rotate_scan_time = 0.0
                self.get_logger().info("Rotating scan done")
                self.map_updated = False
            else:
                self.rotate_scan_time += 0.2
                rotate_cmd = Twist()
                rotate_cmd.angular.z = 0.8
                self.vel_pub.publish(rotate_cmd)
        elif self.state == ExplorationStatus.ROTATING_SCAN_DONE:
            if not self.map_updated:
                return
            if len(self.clustered_frontiers) == 0:
                return
            # Score the frontiers and select the best one
            self.current_goal = extract_best_frontier_point(self.clustered_frontiers, self.map_data, self.map.info, self.robot_current_position)

            #self.navigator.path.header.stamp = self.get_clock().now().to_msg()
            send_goal_result = self.navigator.send_goal(self.robot_current_position, self.current_goal, self.map_data, self.map.info)
            if send_goal_result:
                # Publish for visualization
                self.path_pub.publish(self.navigator.path)

            self.get_logger().info(f"Best frontier: {self.current_goal}")
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = 0
            marker.ns = "current_goal"
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.10
            marker.scale.y = 0.10
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.2
            point = Point()
            point.x, point.y = self.current_goal[0], self.current_goal[1]
            marker.points.append(point)
            self.goal_pub.publish(marker)

            self.state = ExplorationStatus.NAVIGATE_TO_FRONTIER
        elif self.state == ExplorationStatus.NAVIGATE_TO_FRONTIER:
            # Navigate to the goal
            # TODO: Check if the goal is reached
            if np.sqrt((self.robot_current_position[0] - self.current_goal[0])**2 + (self.robot_current_position[1] - self.current_goal[1])**2) < 0.07:
                self.get_logger().info("Goal reached")
                self.state = ExplorationStatus.ROTATING_SCAN
                self.rotate_scan_time = 0.0
                self.current_goal = None
                
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = 0
                marker.ns = "current_goal"
                marker.action = Marker.DELETEALL
                self.goal_pub.publish(marker)
            else:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = 0
                marker.ns = "current_goal"
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.scale.x = 0.10
                marker.scale.y = 0.10
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.2
                point = Point()
                point.x, point.y = self.current_goal[0], self.current_goal[1]
                marker.points.append(point)
                self.goal_pub.publish(marker)
            # TODO: Select the best frontier as above
            pass
        else:
            self.get_logger().info("Invalid state")
    
    def path_follow_timer_cb(self):
        if self.state != ExplorationStatus.NAVIGATE_TO_FRONTIER:
            return
        if len(self.navigator.path.poses) <= 1:
            self.state = ExplorationStatus.ROTATING_SCAN
            return
        # Check if the current waypoint is reached
        if np.sqrt((self.robot_current_position[0] - self.navigator.path.poses[1].pose.position.x)**2 + (self.robot_current_position[1] - self.navigator.path.poses[1].pose.position.y)**2) < 0.06:
            self.navigator.path.poses.pop(0)
        if len(self.navigator.path.poses) <= 1:
            self.navigator.path.poses = []
            self.state = ExplorationStatus.ROTATING_SCAN
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
