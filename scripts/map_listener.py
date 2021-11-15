#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid 
import numpy
    

def callback(data):
    map_w = data.info.width
    map_h = data.info.height
    rawdata = data.data 
    sorted_map = numpy.empty((int(map_h), int(map_w)))
    for i in range(map_w): 
        for j in range(map_h): 
            if rawdata[i + j * map_w] == -1:
                sorted_map[i][j] = float(0.5)
            else:
                sorted_map[i][j] = float(rawdata[i + j * map_w]/100)
    


    


def listener():

    rospy.init_node('map_listener', anonymous=True)
    rospy.Subscriber("/map", OccupancyGrid, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()