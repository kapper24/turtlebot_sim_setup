import rospy
import tf
from tf import listener

def listener():
    rospy.init_node("tf_listener", anonymous = True)
    listener = tf.TransformListener()
    rate = rospy.Rate(10,0)
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

if __name__ == "__main__":
    listener()