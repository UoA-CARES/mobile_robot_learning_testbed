
import random

from matplotlib import markers
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



class TrackGenerator():

    def __init__(self):
        self.origin_x = 0
        self.origin_y = 0
        self.origin_rotation = 0
        self.markers_x = []
        self.markers_y = []

    def SetStraight(self,segment_id,Noise=False):
            next_origin_x = 0
            next_origin_y = 0
            next_origin_rotation = 0

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
                if segment_id == 0 or i == 0:
                    state_msg.model_name = 'aruco_visual_marker_{}'.format(i)
                else:
                    state_msg.model_name = 'aruco_visual_marker_{}_{}'.format(i,segment_id-1)

                if Noise == True:
                    rotation_variation   = (2 * random.random() - 1) * MARKER_ROTATION_VARIATION_LIMIT
                    x_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
                    y_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
                else:
                    rotation_variation = 0
                    x_position_variation = 0
                    y_position_variation = 0
                
                quaternion = quaternion_from_euler(math.radians(self.origin_rotation), -math.pi/2, 0)
                if i == 0:
                    state_msg.pose.position.x = ((LAST_MARKER_ID/2) + 3) * MARKER_SEPERATION
                    state_msg.pose.position.y = 0
                    state_msg.pose.position.z = 0.3
                    state_msg.pose.orientation.x = quaternion[0]
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2]
                    state_msg.pose.orientation.w = quaternion[3]

                    next_origin_x = ((LAST_MARKER_ID/2) + 3) * MARKER_SEPERATION - 3
                    next_origin_y = 0
                    next_origin_rotation = 0

                elif i % 2 == 1: # odd markers left hand side
                    state_msg.pose.position.x = (((i/2.0) + 0.5) * MARKER_SEPERATION) + x_position_variation
                    state_msg.pose.position.y = MARKER_SEPERATION + y_position_variation
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] + MARKER_ROTATION_BASE + rotation_variation
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] + MARKER_ROTATION_BASE + rotation_variation
                    state_msg.pose.orientation.w = quaternion[3]
                else: # even markers right hand side
                    state_msg.pose.position.x = (i/2.0) * MARKER_SEPERATION + x_position_variation
                    state_msg.pose.position.y = -MARKER_SEPERATION + y_position_variation
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] - MARKER_ROTATION_BASE + rotation_variation
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] - MARKER_ROTATION_BASE + rotation_variation
                    state_msg.pose.orientation.w = quaternion[3]

                x_rotated = state_msg.pose.position.x* math.cos(math.radians(self.origin_rotation)) - state_msg.pose.position.y*math.sin(math.radians(self.origin_rotation))
                y_rotated = state_msg.pose.position.y* math.cos(math.radians(self.origin_rotation)) + state_msg.pose.position.x*math.sin(math.radians(self.origin_rotation)) 
                state_msg.pose.position.x = x_rotated + self.origin_x
                state_msg.pose.position.y = y_rotated + self.origin_y
                
                rospy.wait_for_service('/gazebo/set_model_state')
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state(state_msg)

            origin_x_rotated = next_origin_x* math.cos(math.radians(self.origin_rotation)) - next_origin_y*math.sin(math.radians(self.origin_rotation))
            origin_y_rotated = next_origin_y* math.cos(math.radians(self.origin_rotation)) + next_origin_x*math.sin(math.radians(self.origin_rotation)) 
            self.origin_x += origin_x_rotated
            self.origin_y += origin_y_rotated
            self.origin_rotation += next_origin_rotation

    def SetRightCurve(self,segment_id,turn_angle=DEFAULT_TURN_ANGLE, inner_turn_radius=DEFAULT_INNER_TURN_RADIUS, marker_pairs_in_turn=DEFAULT_MARKER_PAIRS_IN_TURN):

        next_origin_x = 0
        next_origin_y = 0
        next_origin_rotation = 0

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
            if segment_id == 0 or i == 0:
                state_msg.model_name = 'aruco_visual_marker_{}'.format(i)
            else:
                state_msg.model_name = 'aruco_visual_marker_{}_{}'.format(i,segment_id-1)
            
            if i > marker_pairs_in_turn*2 :
                quaternion = quaternion_from_euler(-math.radians(turn_angle) + math.radians(self.origin_rotation), -math.pi/2, 0)
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
                    quaternion = quaternion_from_euler(-math.radians(turn_angle)+ math.radians(self.origin_rotation), -math.pi/2, 0)
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

                    midpoint_x = ((outer_turn_radius + inner_turn_radius)/2) 
                    midpoint_y = y + 3
                    x__ = midpoint_x* math.cos(math.radians(90-turn_angle)) - midpoint_y*math.sin(math.radians(90-turn_angle))
                    y__ = midpoint_y* math.cos(math.radians(90-turn_angle)) + midpoint_x*math.sin(math.radians(90-turn_angle)) 

                    next_origin_x = x__ + turn_center_x
                    next_origin_y = y__ - turn_center_y
                    next_origin_rotation = -turn_angle

                elif i % 2 == 1:
                    quaternion = quaternion_from_euler(((-math.radians(turn_angle)) * ((i/2)-0.5) * (1/marker_pairs_in_turn))+ math.radians(self.origin_rotation), -math.pi/2, 0)
                    state_msg.pose.position.x = (math.sin(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + turn_center_x
                    state_msg.pose.position.y = (math.cos(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + -turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] + MARKER_ROTATION_BASE
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] + MARKER_ROTATION_BASE
                    state_msg.pose.orientation.w = quaternion[3]
                else:
                    quaternion = quaternion_from_euler(((-math.radians(turn_angle)) * ((i/2)-1) * (1/marker_pairs_in_turn))+ math.radians(self.origin_rotation), -math.pi/2, 0)
                    state_msg.pose.position.x = (math.sin(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + turn_center_x
                    state_msg.pose.position.y = (math.cos(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + -turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] - MARKER_ROTATION_BASE
                    state_msg.pose.orientation.y = quaternion[1] 
                    state_msg.pose.orientation.z = quaternion[2] - MARKER_ROTATION_BASE
                    state_msg.pose.orientation.w = quaternion[3]

            x_rotated = state_msg.pose.position.x* math.cos(math.radians(self.origin_rotation)) - state_msg.pose.position.y*math.sin(math.radians(self.origin_rotation))
            y_rotated = state_msg.pose.position.y* math.cos(math.radians(self.origin_rotation)) + state_msg.pose.position.x*math.sin(math.radians(self.origin_rotation)) 
            state_msg.pose.position.x = x_rotated + self.origin_x
            state_msg.pose.position.y = y_rotated + self.origin_y
            rospy.wait_for_service('/gazebo/set_model_state')
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

            if i != 0:
                self.markers_x.append(round(state_msg.pose.position.x,2))
                self.markers_y.append(round(state_msg.pose.position.y,2))

        origin_x_rotated = next_origin_x* math.cos(math.radians(self.origin_rotation)) - next_origin_y*math.sin(math.radians(self.origin_rotation))
        origin_y_rotated = next_origin_y* math.cos(math.radians(self.origin_rotation)) + next_origin_x*math.sin(math.radians(self.origin_rotation)) 
        self.origin_x += origin_x_rotated
        self.origin_y += origin_y_rotated
        self.origin_rotation += next_origin_rotation

    def SetLeftCurve(self,segment_id,turn_angle=DEFAULT_TURN_ANGLE, inner_turn_radius=DEFAULT_INNER_TURN_RADIUS, marker_pairs_in_turn=DEFAULT_MARKER_PAIRS_IN_TURN):

        next_origin_x = 0
        next_origin_y = 0
        next_origin_rotation = 0

        outer_turn_radius = inner_turn_radius + (2 * MARKER_SEPERATION)

        turn_center_x = MARKER_SEPERATION
        turn_center_y = (-MARKER_SEPERATION - inner_turn_radius)

        for i in range(LAST_MARKER_ID + 1):
            state_msg = ModelState()
            if segment_id == 0 or i == 0:
                state_msg.model_name = 'aruco_visual_marker_{}'.format(i)
            else:
                state_msg.model_name = 'aruco_visual_marker_{}_{}'.format(i,segment_id-1)

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
                quaternion = quaternion_from_euler(math.radians(turn_angle) + math.radians(self.origin_rotation), -math.pi/2, 0)
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
                    quaternion = quaternion_from_euler(math.radians(turn_angle)+ math.radians(self.origin_rotation), -math.pi/2, 0)
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

                    midpoint_x = ((outer_turn_radius + inner_turn_radius)/2) 
                    midpoint_y = y - 3
                    x__ = midpoint_x* math.cos(math.radians(270+turn_angle)) - midpoint_y*math.sin(math.radians(270+turn_angle))
                    y__ = midpoint_y* math.cos(math.radians(270+turn_angle)) + midpoint_x*math.sin(math.radians(270+turn_angle)) 

                    next_origin_x = x__ + turn_center_x
                    next_origin_y = y__ - turn_center_y
                    next_origin_rotation = turn_angle
                    
                elif i % 2 == 1:
                    quaternion = quaternion_from_euler(((math.radians(turn_angle)) * ((i/2)-0.5) * (1/marker_pairs_in_turn))+ math.radians(self.origin_rotation), -math.pi/2, 0)

                    state_msg.pose.position.x = (math.sin(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + turn_center_x
                    state_msg.pose.position.y = -(math.cos(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) - turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] + MARKER_ROTATION_BASE
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] + MARKER_ROTATION_BASE
                    state_msg.pose.orientation.w = quaternion[3]
                else:
                    
                    quaternion = quaternion_from_euler(((math.radians(turn_angle)) * ((i/2)-1) * (1/marker_pairs_in_turn))+ math.radians(self.origin_rotation), -math.pi/2, 0)
                    state_msg.pose.position.x = (math.sin(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + turn_center_x
                    state_msg.pose.position.y = -(math.cos(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) - turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] - MARKER_ROTATION_BASE
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] - MARKER_ROTATION_BASE
                    state_msg.pose.orientation.w = quaternion[3]
                
            x_rotated = state_msg.pose.position.x* math.cos(math.radians(self.origin_rotation)) - state_msg.pose.position.y*math.sin(math.radians(self.origin_rotation))
            y_rotated = state_msg.pose.position.y* math.cos(math.radians(self.origin_rotation)) + state_msg.pose.position.x*math.sin(math.radians(self.origin_rotation)) 
            state_msg.pose.position.x = x_rotated + self.origin_x
            state_msg.pose.position.y = y_rotated + self.origin_y
            rospy.wait_for_service('/gazebo/set_model_state')
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

            if i != 0:
                self.markers_x.append(round(state_msg.pose.position.x,2))
                self.markers_y.append(round(state_msg.pose.position.y,2))
        
        origin_x_rotated = next_origin_x* math.cos(math.radians(self.origin_rotation)) - next_origin_y*math.sin(math.radians(self.origin_rotation))
        origin_y_rotated = next_origin_y* math.cos(math.radians(self.origin_rotation)) + next_origin_x*math.sin(math.radians(self.origin_rotation)) 
        self.origin_x += origin_x_rotated
        self.origin_y += origin_y_rotated
        self.origin_rotation += next_origin_rotation

    def CycleAllTrackSegments(self,segment=0):
        self.markers_x = []
        self.markers_y = []
        for track_segment_id in range(-6,7):
            self.SetBySegmentId(track_segment_id, segment)
            sleep(2)

    def SetBySegmentId(self, track_segment_id, segment):
        if track_segment_id == 0:
            self.SetStraight(segment)
        elif track_segment_id < 0:
            track_segment_id *= -1
            self.SetLeftCurve(segment,15*track_segment_id,10,(math.floor((track_segment_id/6)*8))+2)
        else:
            self.SetRightCurve(segment,15*track_segment_id,10,(math.floor((track_segment_id/6)*8))+2)
 

    def SetRandomTrackSegment(self,segment=0):
        self.markers_x = []
        self.markers_y = []
        track_segment_id = math.ceil(random.random()* 13) - 7
        if track_segment_id == 0:
            self.SetStraight(segment)
        elif track_segment_id < 0:
            track_segment_id *= -1
            self.SetLeftCurve(segment,15*track_segment_id,10,(math.floor((track_segment_id/6)*8))+2)
        else:
            self.SetRightCurve(segment,15*track_segment_id,10,(math.floor((track_segment_id/6)*8))+2)

    def ResetOrigin(self,x=0,y=0,rotation=0):
        self.origin_x = x
        self.origin_y = y
        self.origin_rotation = rotation

    def SetTrack(self, seg1Id=6, seg2Id=-6):
        self.markers_x = []
        self.markers_y = []
        segment = 0
        self.SetBySegmentId(seg1Id,segment)
        segment = 1
        self.SetBySegmentId(seg2Id,segment)

    def GetMarkerPositions(self):
        return self.markers_x, self.markers_y


if __name__ == '__main__':

    trackGenerator = TrackGenerator()
    trackGenerator.SetTrack()
    #trackGenerator.GetMarkerPositions()
    # while True:
    #     trackGenerator.CycleAllTrackSegments()
    #     sleep(2)

