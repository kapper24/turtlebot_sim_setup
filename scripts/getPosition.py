#!/usr/bin/env python

import rospy
import tf
from tf import listener
import numpy

def listener():
    rospy.init_node("tf_listener", anonymous = True)
    listener = tf.TransformListener()
    rate = rospy.Rate(10,0)
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            xPose = trans[0]
            yPose = trans[1]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    pose =  numpy.array(xPose, yPose)



if __name__ == "__main__":
    listener()