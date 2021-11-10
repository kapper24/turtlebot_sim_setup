#!/usr/bin/env python

import rospy
from rospy.topics import Publisher
import tf
from tf import listener
from std_msgs.msg import Float64MultiArray

def listener():
    rospy.init_node("tf_listener", anonymous = True)
    pub = rospy.Publisher("/obs0", Float64MultiArray, queue_size=10)
    listener = tf.TransformListener()
    posmsg = Float64MultiArray() 
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            posmsg[0] = trans[0]
            posmsg[1] = trans[1]
            pub.publish(posmsg)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        







if __name__ == "__main__":
    listener()