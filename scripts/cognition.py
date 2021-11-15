#!/usr/bin/env python

import numpy
from torch._C import wait
import rospy
import torch
import pyro
import pyro.distributions as dist
from datetime import datetime
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64MultiArray
import time
from RobotExploration import RobotExploration


sorted_map = numpy.empty((int(1), int(1)))
pose = numpy.empty((2, 1))
T_delta = 2

def map_callback(map_data):
    map_w = map_data.info.width
    map_h = map_data.info.height
    rawdata = map_data.data 
    sorted_map = numpy.empty((int(map_h), int(map_w)))
    for i in range(map_w): 
        for j in range(map_h): 
            if rawdata[i + j * map_w] == -1:
                sorted_map[i][j] = float(0.5)
            else:
                sorted_map[i][j] = float(rawdata[i + j * map_w]/100)

def position_callback(pos_data):
    pose = numpy.empty((2, 1))
    pose[0] = float(pos_data.data[0]) 
    pose[1] = float(pos_data.data[1])


def listener():
    rospy.init_node('cognition', anonymous=True)
    rospy.Subscriber("/map", OccupancyGrid, map_callback)
    rospy.Subscriber("/obs0", Float64MultiArray, position_callback)
    
    
    
    cognitive_exploration()
        
        


def cognitive_exploration():
    meter2pixel = 10  # X pixel = 1 meter
    robotRadius = 0.5 / meter2pixel  # robot radius in meter
    lidar_range = 4  # laser range in pixel
    lidar_FOV = 6  # laser field of view in rad
    lidar_resolution = 0.1  # laser rotation resolution in rad
    lidar_sigma_hit = 0.1  # sigma of Gaussian distribution of laser noise
    d_min = robotRadius + 0.05  # we add a small buffer of 5 cm - d_min = 0.25 m
    agent = RobotExploration(meter2pixel, lidar_range, lidar_FOV, lidar_resolution, lidar_sigma_hit, d_min)
    T_delta = 2
    act = numpy.zeros(3)  # start by doing nothing!
    t = 0

    standard_diviation = torch.tensor(0.2 / 3)  # 99.7% of samples within a circle of 20 cm
    variance = standard_diviation * standard_diviation
    cov_s = variance * torch.eye(2)
    time_pr_iteration = []

    while not rospy.is_shutdown():
        position = [pose[0], pose[1]]  # we only use the position not the heading
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
        z_a_tPlus_samples, z_s_tPlus_samples = agent.makePlan(t, T_delta, p_z_s_t, map_grid_probabilities, return_mode="raw_samples")

        z_a_tPlus, z_s_tPlus_ = agent.calculate_mean_state_action_means(z_a_tPlus_samples, z_s_tPlus_samples)
        z_a_tPlus = z_a_tPlus_samples[0]
        z_s_tPlus_ = z_s_tPlus_samples[0]
        toc = time.time()
        time_pr_iteration.append(toc - tic)

            #convert plan to the format used by the simulator
        act[0] = z_a_tPlus[0].numpy()[0]
        act[1] = z_a_tPlus[0].numpy()[1]
        t += 1
        return act
#

if __name__ == '__main__':
    listener()
    