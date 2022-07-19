
import random
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from time import sleep
from turtle import position
from torch import true_divide
import rospy
import math
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

# General Marker Pose Variable
MARKER_SEPERATION = 1
MARKER_SEPERATION_VARIATION_LIMIT = 0.05
MARKER_ROTATION_BASE = 0.2
MARKER_ROTATION_VARIATION_LIMIT = 0.1

# Max marker ID available in world
LAST_MARKER_ID = 30

#Default turn setup if no parameters provided
DEFAULT_INNER_TURN_RADIUS = 10
DEFAULT_MARKER_PAIRS_IN_TURN = 10
DEFAULT_TURN_ANGLE = 90


def SetStraight(Noise=False):
        for i in range(LAST_MARKER_ID + 1):

            if i % 2 == 1:
                if (i + 1) % 4 == 0:
                    z = 0.5
                else:
                    z = 0.1 
            else:
                if i % 4 == 0:
                    z = 0.5
                else:
                    z = 0.1

            state_msg = ModelState()
            state_msg.model_name = 'aruco_visual_marker_{}'.format(i)

            if Noise == True:
                rotation_variation   = (2 * random.random() - 1) * MARKER_ROTATION_VARIATION_LIMIT
                x_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
                y_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
            else:
                rotation_variation = 0
                x_position_variation = 0
                y_position_variation = 0
            
            if i == 0:
                state_msg.pose.position.x = ((LAST_MARKER_ID/2) + 3) * MARKER_SEPERATION
                state_msg.pose.position.y = 0
                state_msg.pose.position.z = 0.3
                state_msg.pose.orientation.x = 0
                state_msg.pose.orientation.y = -0.7
                state_msg.pose.orientation.z = 0
                state_msg.pose.orientation.w = 0.7
            elif i % 2 == 1: # odd markers left hand side
                state_msg.pose.position.x = (((i/2.0) + 0.5) * MARKER_SEPERATION) + x_position_variation
                state_msg.pose.position.y = MARKER_SEPERATION + y_position_variation
                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = MARKER_ROTATION_BASE + rotation_variation
                state_msg.pose.orientation.y = -0.7
                state_msg.pose.orientation.z = MARKER_ROTATION_BASE + rotation_variation
                state_msg.pose.orientation.w = 0.7
            else: # even markers right hand side
                state_msg.pose.position.x = (i/2.0) * MARKER_SEPERATION + x_position_variation
                state_msg.pose.position.y = -MARKER_SEPERATION + y_position_variation
                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = -MARKER_ROTATION_BASE + rotation_variation
                state_msg.pose.orientation.y = -0.7
                state_msg.pose.orientation.z = -MARKER_ROTATION_BASE + rotation_variation
                state_msg.pose.orientation.w = 0.7

            rospy.wait_for_service('/gazebo/set_model_state')
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

def SetRightCurve(turn_angle=DEFAULT_TURN_ANGLE, inner_turn_radius=DEFAULT_INNER_TURN_RADIUS, marker_pairs_in_turn=DEFAULT_MARKER_PAIRS_IN_TURN):

    outer_turn_radius = inner_turn_radius + (2 * MARKER_SEPERATION)
    turn_center_x = MARKER_SEPERATION
    turn_center_y = (MARKER_SEPERATION + inner_turn_radius)

    for i in range(LAST_MARKER_ID + 1):

        if i % 2 == 1:
            if (i + 1) % 4 == 0:
                z = 0.5
            else:
                z = 0.1 
        else:
            if i % 4 == 0:
                z = 0.5
            else:
                z = 0.1

        state_msg = ModelState()
        state_msg.model_name = 'aruco_visual_marker_{}'.format(i)
        
        if i > marker_pairs_in_turn*2 :
            quaternion = quaternion_from_euler(-math.radians(turn_angle), -math.pi/2, 0)
            if i % 2 == 1:
                x = outer_turn_radius
                y = -(((i-(marker_pairs_in_turn*2))/2) - 0.5)
                x_ = x* math.cos(math.radians(90-turn_angle)) - y*math.sin(math.radians(90-turn_angle))
                y_ = y* math.cos(math.radians(90-turn_angle)) + x*math.sin(math.radians(90-turn_angle)) 
                state_msg.pose.position.x = x_ + turn_center_x
                state_msg.pose.position.y = y_ - turn_center_y

                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = quaternion[0] + MARKER_ROTATION_BASE
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2] + MARKER_ROTATION_BASE
                state_msg.pose.orientation.w = quaternion[3]
            else:
                x = inner_turn_radius
                y = -((i-(marker_pairs_in_turn*2))/2) + 1
                x_ = x* math.cos(math.radians(90-turn_angle)) - y*math.sin(math.radians(90-turn_angle))
                y_ = y* math.cos(math.radians(90-turn_angle)) + x*math.sin(math.radians(90-turn_angle)) 
                state_msg.pose.position.x = x_ + turn_center_x
                state_msg.pose.position.y = y_ - turn_center_y

                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = quaternion[0] - MARKER_ROTATION_BASE
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2] - MARKER_ROTATION_BASE
                state_msg.pose.orientation.w = quaternion[3]
        else:
            if i == 0:
                quaternion = quaternion_from_euler(-math.radians(turn_angle), -math.pi/2, 0)
                x = ((outer_turn_radius + inner_turn_radius)/2) 
                y = -(((LAST_MARKER_ID-(marker_pairs_in_turn*2))/2) - 0.5) - 2.5
                x_ = x* math.cos(math.radians(90-turn_angle)) - y*math.sin(math.radians(90-turn_angle))
                y_ = y* math.cos(math.radians(90-turn_angle)) + x*math.sin(math.radians(90-turn_angle)) 
                state_msg.pose.position.x = x_ + turn_center_x
                state_msg.pose.position.y = y_ - turn_center_y

                state_msg.pose.position.z = 0.3
                state_msg.pose.orientation.x = quaternion[0]
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2]
                state_msg.pose.orientation.w = quaternion[3]
            elif i % 2 == 1:
                quaternion = quaternion_from_euler((-math.radians(turn_angle)) * ((i/2)-0.5) * (1/marker_pairs_in_turn), -math.pi/2, 0)
                state_msg.pose.position.x = (math.sin(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + turn_center_x
                state_msg.pose.position.y = (math.cos(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + -turn_center_y
                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = quaternion[0] + MARKER_ROTATION_BASE
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2] + MARKER_ROTATION_BASE
                state_msg.pose.orientation.w = quaternion[3]
            else:
                quaternion = quaternion_from_euler((-math.radians(turn_angle)) * ((i/2)-1) * (1/marker_pairs_in_turn), -math.pi/2, 0)
                state_msg.pose.position.x = (math.sin(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + turn_center_x
                state_msg.pose.position.y = (math.cos(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + -turn_center_y
                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = quaternion[0] - MARKER_ROTATION_BASE
                state_msg.pose.orientation.y = quaternion[1] 
                state_msg.pose.orientation.z = quaternion[2] - MARKER_ROTATION_BASE
                state_msg.pose.orientation.w = quaternion[3]

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)

def SetLeftCurve(turn_angle=DEFAULT_TURN_ANGLE, inner_turn_radius=DEFAULT_INNER_TURN_RADIUS, marker_pairs_in_turn=DEFAULT_MARKER_PAIRS_IN_TURN):

    outer_turn_radius = inner_turn_radius + (2 * MARKER_SEPERATION)

    turn_center_x = MARKER_SEPERATION
    turn_center_y = (-MARKER_SEPERATION - inner_turn_radius)

    for i in range(LAST_MARKER_ID + 1):
        state_msg = ModelState()
        state_msg.model_name = 'aruco_visual_marker_{}'.format(i)

        if i % 2 == 1:
            if (i + 1) % 4 == 0:
                z = 0.5
            else:
                z = 0.1 
        else:
            if i % 4 == 0:
                z = 0.5
            else:
                z = 0.1
        
        if i > marker_pairs_in_turn*2 :
            quaternion = quaternion_from_euler(math.radians(turn_angle), -math.pi/2, 0)
            if i % 2 == 1:
                x = inner_turn_radius
                y = (((i-(marker_pairs_in_turn*2))/2) - 0.5)
                x_ = x* math.cos(math.radians(270+turn_angle)) - y*math.sin(math.radians(270+turn_angle))
                y_ = y* math.cos(math.radians(270+turn_angle)) + x*math.sin(math.radians(270+turn_angle)) 
                state_msg.pose.position.x = x_ + turn_center_x
                state_msg.pose.position.y = y_ - turn_center_y

                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = quaternion[0] + MARKER_ROTATION_BASE
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2] + MARKER_ROTATION_BASE
                state_msg.pose.orientation.w = quaternion[3]
            else:
                x = outer_turn_radius
                y = ((i-(marker_pairs_in_turn*2))/2) - 1
                x_ = x* math.cos(math.radians(270+turn_angle)) - y*math.sin(math.radians(270+turn_angle))
                y_ = y* math.cos(math.radians(270+turn_angle)) + x*math.sin(math.radians(270+turn_angle)) 
                state_msg.pose.position.x = x_ + turn_center_x
                state_msg.pose.position.y = y_ - turn_center_y

                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = quaternion[0] - MARKER_ROTATION_BASE
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2] - MARKER_ROTATION_BASE
                state_msg.pose.orientation.w = quaternion[3]
                
            
        else:
            if i == 0:
                quaternion = quaternion_from_euler(math.radians(turn_angle), -math.pi/2, 0)
                x = ((outer_turn_radius + inner_turn_radius)/2) 
                y = (((LAST_MARKER_ID-(marker_pairs_in_turn*2))/2) - 0.5) + 2.5
                x_ = x* math.cos(math.radians(270+turn_angle)) - y*math.sin(math.radians(270+turn_angle))
                y_ = y* math.cos(math.radians(270+turn_angle)) + x*math.sin(math.radians(270+turn_angle)) 
                state_msg.pose.position.x = x_ + turn_center_x
                state_msg.pose.position.y = y_ - turn_center_y

                state_msg.pose.position.z = 0.3
                state_msg.pose.orientation.x = quaternion[0]
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2] 
                state_msg.pose.orientation.w = quaternion[3]
                
            elif i % 2 == 1:
                quaternion = quaternion_from_euler((math.radians(turn_angle)) * ((i/2)-0.5) * (1/marker_pairs_in_turn), -math.pi/2, 0)

                state_msg.pose.position.x = (math.sin(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + turn_center_x
                state_msg.pose.position.y = -(math.cos(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) - turn_center_y
                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = quaternion[0] + MARKER_ROTATION_BASE
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2] + MARKER_ROTATION_BASE
                state_msg.pose.orientation.w = quaternion[3]
            else:
                
                quaternion = quaternion_from_euler((math.radians(turn_angle)) * ((i/2)-1) * (1/marker_pairs_in_turn), -math.pi/2, 0)
                state_msg.pose.position.x = (math.sin(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + turn_center_x
                state_msg.pose.position.y = -(math.cos(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) - turn_center_y
                state_msg.pose.position.z = z
                state_msg.pose.orientation.x = quaternion[0] - MARKER_ROTATION_BASE
                state_msg.pose.orientation.y = quaternion[1]
                state_msg.pose.orientation.z = quaternion[2] - MARKER_ROTATION_BASE
                state_msg.pose.orientation.w = quaternion[3]
               

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)


# if __name__ == '__main__':

#     while True:
#         track_type = random.random()* 10
#         if track_type < 2:
#              SetStraight()
#         elif track_type < 6:
#             x = math.ceil(random.random()* 6) 
#             SetLeftCurve(15*x,10,math.floor(0.5*x)+4)
#         else:
#             x = math.ceil(random.random()* 6) 
#             SetRightCurve(15*x,10,math.floor(0.5*x)+4)
#         sleep(0.5)

