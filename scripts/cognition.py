#!/usr/bin/env python3

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
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from pathlib import Path
#for get pos
import tf
from tf import listener
import time
from RobotExploration import RobotExploration
path = Path('/home/melvin/turtle_ws/src/turtlebot_sim_setup/scripts')
meter2pixel = int(rospy.get_param("/cognition/pixelsprmeter", 20))  # X pixel = 1 meter
robotRadius = rospy.get_param("/cognition/robotradius", 0.18 / meter2pixel)  # robot radius in meter
lidar_range = int(rospy.get_param("/cognition/pixellaserrange", 60))  # laser range in pixel
lidar_FOV = rospy.get_param("/cognition/laserfow", 6.28)  # laser field of view in rad
lidar_resolution = rospy.get_param("/cognition/laserresolution", 6.28/360)  # laser rotation resolution in rad
lidar_sigma_hit = rospy.get_param("/cognition/lasernoise", 0)  # sigma of Gaussian distribution of laser noise
d_min = robotRadius + rospy.get_param("/cognition/mindistance", 0.05)  # we add a small buffer of 5 cm - d_min = 0.25 m
p_z_g = None
intcheck = 1
sorted_map = numpy.zeros((int(100), int(100)))
pose = numpy.zeros((2, 1)) 
T_delta = 3

#def markerpublisher(data, pos):
#    global marker_ests
#    #Publish it as a marker in rviz
#    marker_ests = MarkerArray()
#    marker_ests.markers = []
#    
#    k = 0
#    for i in data:
#        marker_est = Marker()
#        marker_est.header.frame_id = "map"
#        marker_est.header.stamp = rospy.Time.now()
#        marker_est.pose.position.x = float(pos[0]) + float(data[k][0][0])
#        marker_est.pose.position.y = float(pos[1]) + float(data[k][0][1])
#        marker_est.pose.position.z = 0
#        marker_est.pose.orientation.x = 0
#        marker_est.pose.orientation.y = 0
#        marker_est.pose.orientation.z = 0
#        marker_est.pose.orientation.w = 1
#        marker_est.ns = "est_pose_"+str(k)
#        marker_est.id = 42+k
#        marker_est.type = Marker.CUBE
#        marker_est.action = Marker.ADD
#        marker_est.color.r, marker_est.color.g, marker_est.color.b = (0, 255, 0)
#        marker_est.color.a = 0.5
#        marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (0.006, 0.006, 0.006)
#        marker_ests.markers.append(marker_est)
#        k+=1
#    marker_pub.publish(marker_ests)

def map_callback(map_data):
    map_w = map_data.info.width
    #print(str(map_w))
    map_h = map_data.info.height
    #print(str(map_h))
    rawdata = map_data.data 
    sorted_map = numpy.empty((int(map_h), int(map_w)))
    for i in range(map_w): 
        for j in range(map_h): 
            sorted_map[j][i] = rawdata[i + map_w * j ]/100.0
    return sorted_map
    #numpy.savetxt(path/'map_2',sorted_map,delimiter=',')       
  #  print("saved_map") 
          
def position_callback():
    try:
        (trans,rot) = tflistener.lookupTransform('/map', '/base_link', rospy.Time(0))
        pose[0] = float(trans[0]) 
        pose[1] = float(trans[1])
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print(" ")
   #     position_callback()
        

def listener():
    rospy.init_node('cognition', anonymous=True)
    #rospy.Subscriber("/map", OccupancyGrid, map_callback)
    #rospy.Subscriber("/obs0", Float64MultiArray, position_callback)
    
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()
    #for get pos
    print("1")
    global tflistener
    #global marker_pub
    #marker_pub = rospy.Publisher("/visualization_markerarray", MarkerArray, queue_size = 2)
    tflistener = tf.TransformListener()
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
        map_data = rospy.wait_for_message("/map1", OccupancyGrid)
        #pos_data = rospy.wait_for_message("/obs0", Float64MultiArray) for get position
        position_callback()

        map = map_callback(map_data)
        print(map)
        position = numpy.array([pose[0][0] + 10, pose[1][0] + 10])  # we only use the position not the heading
        print( "positionx" + str(position[0]) + "positiony" + str(position[1]))

        map_grid_probabilities_np = map.copy()
        map_grid_probabilities = torch.from_numpy(map_grid_probabilities_np)
       # print(map_grid_probabilities)
        map_grid_probabilities
       
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
        #print(z_a_tPlus)
        #markerpublisher(z_a_tPlus, z_s_t)
        #act[0] = z_s_tPlus_[0][0]
        #act[1] = z_s_tPlus_[0][1]
        #z_s_tPlus_ = z_s_tPlus_samples[0]
        toc = time.time()
        time_pr_iteration.append(toc - tic)

            #convert plan to the format used by the simulator
   
      
            
        act[0] = z_s_t[0] + z_a_tPlus[0][0] - 10
        act[1] = z_s_t[1] + z_a_tPlus[0][1] - 10
        #act[0] = z_s_tPlus_[0][0] - 10
        #act[1] = z_s_tPlus_[0][1] - 10

       
        
        directionx = z_a_tPlus[0][0] 
        directiony = z_a_tPlus[0][1]
        direction_angle = numpy.arctan2(directiony, directionx)
        quat = euler_to_quaternion(direction_angle,0,0)
        
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        #goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = act[0]
        goal.target_pose.pose.position.y = act[1]
        
        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]
        
        client.send_goal(goal)
        #print("z_s_tPlus_" + str(z_s_tPlus_))
        #print("z_a_tPlus" + str(z_a_tPlus))
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
    