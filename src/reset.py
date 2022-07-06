#OLD RESET CODE
# import rospy
# from std_srvs.srv import Empty

# rospy.init_node('reset_world')

# rospy.wait_for_service('/gazebo/reset_world')
# reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

# reset_world()


import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

def main():
    rospy.init_node('set_pose')

    state_msg = ModelState()
    state_msg.model_name = 'mobile_base'
    state_msg.pose.position.x = 0
    state_msg.pose.position.y = 0
    state_msg.pose.position.z = 0.0
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 0

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass