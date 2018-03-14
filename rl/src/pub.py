#!/usr/bin/env python
# license removed for brevity
import rospy
from geometry_msgs.msg import Twist

def callback():
    pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    rospy.init_node('pub', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    count = 0
    msg = Twist()

    while not rospy.is_shutdown():
        # Put this in an init function
        if(count < 10):
            msg.linear.z = 0.5
            rospy.loginfo('Lift off')

        # Put this in a shutdown function
        elif(count > 100):
            rospy.loginfo('Stop')
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0

        else:
            rospy.loginfo('Move forward')
            msg.linear.x = 0.5
            msg.linear.y = 0.0
            msg.linear.z = 0.0

        pub_.publish(msg)
        count = count + 1
        rate.sleep()

if __name__ == '__main__':
    try:
        callback()
    except rospy.ROSInterruptException:
        pass