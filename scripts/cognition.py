#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid 
from std_msgs.msg import Float64MultiArray
import numpy
obs = list((2, 1))

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
    obs[1] = sorted_map

def position_callback(pos_data):
    pose = numpy.empty((2, 1))
    pose[0] = float(pos_data.data[0]) 
    pose[1] = float(pos_data.data[1])
    obs[0] = pose

def listener():

    rospy.init_node('cognition', anonymous=True)
    rospy.Subscriber("/map", OccupancyGrid, map_callback)
    rospy.Subscriber("/obs0", Float64MultiArray, position_callback)
    while not rospy.is_shutdown():
        print(obs[1])
        print(obs[0])
        
        
if __name__ == '__main__':
    listener()
    

def cognitive_exploration():
    a = 0