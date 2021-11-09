import rospy
from nav_msgs.msg import OccupancyGrid 
    

def callback(data):
    map_w = data.info.width
    map_h = data.info.height
    rawdata = data.data 
    sorted_map = []
    for i in range(map_w): 
        for j in range(map_h): 

            float()
            if rawdata[i + j * map_w] == -1:
                sorted_map[i][j] = float(0.5)
            else:
                sorted_map[i][j] = float(rawdata[i + j * map_w]/100)
    print(sorted_map)


    


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('map_listener', anonymous=True)

    rospy.Subscriber("/map", OccupancyGrid, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()