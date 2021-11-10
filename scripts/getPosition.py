#!/usr/bin/env python

from os import posix_fadvise
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
            posx = float(trans[0])
            posy = float(trans[1])
            posmsg.data = {posx, posy}
            pub.publish(posmsg)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        







if __name__ == "__main__":
    listener()