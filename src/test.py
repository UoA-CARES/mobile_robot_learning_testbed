import rospy
from geometry_msgs.msg import Point, Twist
rospy.init_node('test_move', anonymous=False)
cmd_vel = rospy.Publisher("/cmd_vel_mux/input/navi", Twist, queue_size=10)

r = rospy.Rate(20)

while True:
    move_cmd = Twist()
    move_cmd.linear.x = 0.5
    move_cmd.linear.y = 0
    move_cmd.linear.z = 0
    move_cmd.angular.x = 0
    move_cmd.angular.y = 0
    move_cmd.angular.z = 0
    cmd_vel.publish(move_cmd)
    r.sleep()