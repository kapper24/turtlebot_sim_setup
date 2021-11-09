#!/usr/bin/env python

import rospy
import tf
from tf import listener
import numpy

def listener():
    rospy.init_node("tf_listener", anonymous = True)
    listener = tf.TransformListener()
    pose = numpy.empty((2, 1))
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            xPose = trans[0]
            yPose = trans[1]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        pose[0] = float(xPose) 
        pose[1] = float(yPose)
        print(pose)




if __name__ == "__main__":
    listener()