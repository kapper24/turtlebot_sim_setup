#!/usr/bin/env python

import numpy
from torch._C import wait
from genpy.message import check_type
import rospy
import torch
import pyro
import pyro.distributions as dist
from datetime import datetime
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int8
from geometry_msgs.msg import PoseStamped
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

import time
from RobotExploration import RobotExploration

meter2pixel = int(rospy.get_param("/cognition/pixelsprmeter", 20))  # X pixel = 1 meter
robotRadius = rospy.get_param("/cognition/robotradius", 0.18 / meter2pixel)  # robot radius in meter
lidar_range = int(rospy.get_param("/cognition/pixellaserrange", 70))  # laser range in pixel
lidar_FOV = rospy.get_param("/cognition/laserfow", 6.28)  # laser field of view in rad
lidar_resolution = rospy.get_param("/cognition/laserresolution", 6.28/360)  # laser rotation resolution in rad
lidar_sigma_hit = rospy.get_param("/cognition/lasernoise", 0.1)  # sigma of Gaussian distribution of laser noise
d_min = robotRadius + rospy.get_param("/cognition/mindistance", 0.05)  # we add a small buffer of 5 cm - d_min = 0.25 m
p_z_g = None
intcheck = 1
sorted_map = numpy.zeros((int(100), int(100)))
pose = numpy.zeros((2, 1)) 
T_delta = 2

def map_callback(map_data):
    map_w = map_data.info.width
    map_h = map_data.info.height
    rawdata = map_data.data 
    sorted_map = numpy.empty((int(map_h), int(map_w)))
    for i in range(map_w): 
        for j in range(map_h): 
            if rawdata[i + j * map_w] == -1:
                sorted_map[map_h-1-j][i] = float(0.5)
            if rawdata[i + j * map_w] > 70:
                sorted_map[map_h-1-j][i] = float(1)
            else:
                sorted_map[map_h-1-j][i] = float(rawdata[i * map_w + j ]/100)
    #for i in range(map_w): 
    #    for j in range(map_h): 
    #        if rawdata[i + j * map_w] == -1:
    #            sorted_map[j][i] = float(0.5)
    #        else:
    #            sorted_map[j][i] = float(rawdata[i * map_w + j ]/100)
    
    

def position_callback(pos_data):
    pose[0] = float(pos_data.data[0]) 
    pose[1] = float(pos_data.data[1])
    
 


def listener():
    rospy.init_node('cognition', anonymous=True)
    #rospy.Subscriber("/map", OccupancyGrid, map_callback)
    #rospy.Subscriber("/obs0", Float64MultiArray, position_callback)
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()
    cognitive_exploration(client)
        
        
def euler_to_quaternion(yaw, pitch, roll):

        qx = numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) - numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)
        qy = numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2)
        qz = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2) - numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2)
        qw = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)

        return [qx, qy, qz, qw]

def cognitive_exploration(client):
    agent = RobotExploration(meter2pixel, lidar_range, lidar_FOV, lidar_resolution, lidar_sigma_hit, d_min)
    act = numpy.zeros(3)  # start by doing nothing!
    t = 0

    standard_diviation = torch.tensor(0.2 / 3)  # 99.7% of samples within a circle of 20 cm
    variance = standard_diviation * standard_diviation
    cov_s = variance * torch.eye(2)
    time_pr_iteration = []

    while not rospy.is_shutdown():
        map_data = rospy.wait_for_message("/map", OccupancyGrid)
        pos_data = rospy.wait_for_message("/obs0", Float64MultiArray)
        position_callback(pos_data)
        map_callback(map_data)
        position = numpy.array([pose[1][0], pose[0][0]])  # we only use the position not the heading
        print( "position" + str(position))

        map_grid_probabilities_np = sorted_map.copy()
        map_grid_probabilities = torch.from_numpy(map_grid_probabilities_np)
        map_grid_probabilities = torch.flip(map_grid_probabilities, [0])
        z_s_t = torch.tensor([position[0], position[1]], dtype=torch.float)

        def p_z_s_t():
            z_s_t_ = z_s_t.detach()
            cov_s_ = cov_s.detach()
            _p_z_s_t = dist.MultivariateNormal(z_s_t_, cov_s_)
            z_s = pyro.sample("z_s", _p_z_s_t)
            return z_s

            # make new plan
            # with contextlib.redirect_stdout(open(os.devnull, 'w')):
        tic = time.time()
        #z_a_tPlus_samples, z_s_tPlus_samples = agent.makePlan(t, T_delta, p_z_s_t, map_grid_probabilities, return_mode="raw_samples", p_z_g=p_z_g)
        z_a_tPlus, z_s_tPlus_ = agent.makePlan(t, T_delta, p_z_s_t, map_grid_probabilities, return_mode="mean", p_z_g=p_z_g)     
        #z_a_tPlus, z_s_tPlus_ = agent.calculate_mean_state_action_means(z_a_tPlus_samples, z_s_tPlus_samples)
        
        #act[0] = z_s_tPlus_[0][0]
        #act[1] = z_s_tPlus_[0][1]
        #z_s_tPlus_ = z_s_tPlus_samples[0]
        toc = time.time()
        time_pr_iteration.append(toc - tic)

            #convert plan to the format used by the simulator
        
        if torch.abs(z_a_tPlus[0][0]) < 0.001 or torch.abs(z_a_tPlus[0][1]) < 0.001:
            act[0] = 0
            act[1] = 0
        else:
            act[0] = z_a_tPlus[0][0]
            act[1] = z_a_tPlus[0][1]

       
        
        directionx = z_a_tPlus[0][0] 
        directiony = z_a_tPlus[0][1]
        direction_angle = numpy.arctan2(directiony, directionx)
        quat = euler_to_quaternion(direction_angle,0,0)
        
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "base_link"
        #goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = act[0]
        goal.target_pose.pose.position.y = act[1]
        
        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]
        
        client.send_goal(goal)
        print("z_s_tPlus_" + str(z_s_tPlus_))
        print("z_a_tPlus" + str(z_a_tPlus))
        print("goal" + str(goal.target_pose.pose.position.x) + " " + str(goal.target_pose.pose.position.y))

        
        i = 0
        while i < 10:
            rospy.sleep(1)
            a = client.get_state()
            if a == 3:
                i = 10
            if i == 9:
                client.cancel_goal()
            i = i+1
        

if __name__ == '__main__':
    listener()
    