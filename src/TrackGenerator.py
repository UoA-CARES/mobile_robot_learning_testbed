"""
Authors:  Aakaash Salvaji, Harry Taylor, David Valencia, Trevor Gee, Henry Williams
The University of Auckland

Track Generator:
Generate Straight, Left, Right segments with constant inner turn radius of 10 units.
Track Segment IDs: -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6
NOTE: 
Negative IDs = Turn Left
ID number * 15 = Turn Angle
i.e. track segment ID 3 = 45 degree right turn & -5 = 75 degree left turn
"""
import random
from tf.transformations import quaternion_from_euler
from turtle import position #TODO: Do we need this?
from torch import true_divide #TODO: Do we need this?
import rospy
import math
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState


# Base seperation between markers ALONG straight track (2 times seperation = track width) in Gazebo units
MARKER_SEPERATION = 1
MARKER_SEPERATION_VARIATION_LIMIT = 0.05 # MAX Noise position +/- (units)

# Base inward rotation of markers in degrees
MARKER_ROTATION_BASE = 20 
MARKER_ROTATION_VARIATION_LIMIT = 15 # MAX Noise rotation +/- (degrees)

# Max marker ID available in world
LAST_MARKER_ID = 30

# Default turn setup if no parameters provided
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

    # Set straight segment 
    def SetStraight(self,segment_id,Noise=False):
            next_origin_x = 0
            next_origin_y = 0
            next_origin_rotation = 0

            # Alternate heights of markers to increase visibility
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
                # As only 2 sets of identical markers used, sets distinguished with "_0"
                if segment_id == 0 or i == 0:
                    state_msg.model_name = 'aruco_visual_marker_{}'.format(i)
                else:
                    state_msg.model_name = 'aruco_visual_marker_{}_{}'.format(i,segment_id-1)

                # If noise = True, add random variation to positon and rotation
                if Noise == True:
                    rotation_variation   = (2 * random.random() - 1) * MARKER_ROTATION_VARIATION_LIMIT
                    x_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
                    y_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
                else:
                    rotation_variation = 0
                    x_position_variation = 0
                    y_position_variation = 0
                
                # If marker ID = 0
                if i == 0:
                    quaternion = quaternion_from_euler(math.radians(self.origin_rotation), -math.pi/2, 0)
                    state_msg.pose.position.x = ((LAST_MARKER_ID/2) + 3) * MARKER_SEPERATION
                    state_msg.pose.position.y = 0
                    state_msg.pose.position.z = 0.6
                    state_msg.pose.orientation.x = quaternion[0]
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2]
                    state_msg.pose.orientation.w = quaternion[3]

                    next_origin_x = ((LAST_MARKER_ID/2) + 3) * MARKER_SEPERATION - 3
                    next_origin_y = 0
                    next_origin_rotation = 0
                    
                # IF odd markers place on left hand side
                elif i % 2 == 1: 
                    quaternion = quaternion_from_euler(math.radians(self.origin_rotation) + math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    state_msg.pose.position.x = (((i/2.0) + 0.5) * MARKER_SEPERATION) + x_position_variation
                    state_msg.pose.position.y = MARKER_SEPERATION + y_position_variation
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] 
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2]
                    state_msg.pose.orientation.w = quaternion[3]

                # If even markers place on right hand side
                else: 
                    quaternion = quaternion_from_euler(math.radians(self.origin_rotation) - math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    state_msg.pose.position.x = (i/2.0) * MARKER_SEPERATION + x_position_variation
                    state_msg.pose.position.y = -MARKER_SEPERATION + y_position_variation
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0]
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2]
                    state_msg.pose.orientation.w = quaternion[3]

                # Translate and rotate segment to START relative to the END of previous segment
                x_rotated = state_msg.pose.position.x* math.cos(math.radians(self.origin_rotation)) - state_msg.pose.position.y*math.sin(math.radians(self.origin_rotation))
                y_rotated = state_msg.pose.position.y* math.cos(math.radians(self.origin_rotation)) + state_msg.pose.position.x*math.sin(math.radians(self.origin_rotation)) 
                state_msg.pose.position.x = x_rotated + self.origin_x
                state_msg.pose.position.y = y_rotated + self.origin_y
                
                rospy.wait_for_service('/gazebo/set_model_state')
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state(state_msg)
                
                # If not zero marker, store position of marker
                if i != 0:
                    self.markers_x.append(round(state_msg.pose.position.x,2))
                    self.markers_y.append(round(state_msg.pose.position.y,2))

            # Update "origin" for next segment to build from
            origin_x_rotated = next_origin_x* math.cos(math.radians(self.origin_rotation)) - next_origin_y*math.sin(math.radians(self.origin_rotation))
            origin_y_rotated = next_origin_y* math.cos(math.radians(self.origin_rotation)) + next_origin_x*math.sin(math.radians(self.origin_rotation)) 
            self.origin_x += origin_x_rotated
            self.origin_y += origin_y_rotated
            self.origin_rotation += next_origin_rotation

    # Set right turn segment 
    def SetRightCurve(self,segment_id,turn_angle=DEFAULT_TURN_ANGLE, inner_turn_radius=DEFAULT_INNER_TURN_RADIUS, marker_pairs_in_turn=DEFAULT_MARKER_PAIRS_IN_TURN,Noise=False):

        next_origin_x = 0
        next_origin_y = 0
        next_origin_rotation = 0

        # Calculate turn radii and center of circle that turn is calculated from
        outer_turn_radius = inner_turn_radius + (2 * MARKER_SEPERATION)
        turn_center_x = MARKER_SEPERATION
        turn_center_y = (MARKER_SEPERATION + inner_turn_radius)
        
        for i in range(LAST_MARKER_ID + 1):

            # Alternate heights of markers to increase visibility
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
            # As only 2 sets of identical markers used, sets distinguished with "_0"
            if segment_id == 0 or i == 0:
                state_msg.model_name = 'aruco_visual_marker_{}'.format(i)
            else:
                state_msg.model_name = 'aruco_visual_marker_{}_{}'.format(i,segment_id-1)

            # If noise = True, add random variation to positon and rotation
            if Noise == True:
                rotation_variation   = (2 * random.random() - 1) * MARKER_ROTATION_VARIATION_LIMIT
                x_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
                y_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
            else:
                rotation_variation = 0
                x_position_variation = 0
                y_position_variation = 0
            
            # Set markers used in actual curve of turn
            if i <= marker_pairs_in_turn*2:
                # If marker ID = 0
                if i == 0:
                    quaternion = quaternion_from_euler(-math.radians(turn_angle)+ math.radians(self.origin_rotation), -math.pi/2, 0)
                    x = ((outer_turn_radius + inner_turn_radius)/2)
                    y = -(((LAST_MARKER_ID-(marker_pairs_in_turn*2))/2) - 0.5) - 2.5
                    x_ = x* math.cos(math.radians(90-turn_angle)) - y*math.sin(math.radians(90-turn_angle))
                    y_ = y* math.cos(math.radians(90-turn_angle)) + x*math.sin(math.radians(90-turn_angle)) 
                    state_msg.pose.position.x = x_ + turn_center_x
                    state_msg.pose.position.y = y_ - turn_center_y
                    state_msg.pose.position.z = 0.6
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
                
                # IF odd markers place on left hand side (outer turn radius)
                elif i % 2 == 1:
                    quaternion = quaternion_from_euler(((-math.radians(turn_angle)) * ((i/2)-0.5) * (1/marker_pairs_in_turn))+ math.radians(self.origin_rotation) + math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    state_msg.pose.position.x = (math.sin(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + turn_center_x
                    state_msg.pose.position.y = (math.cos(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + -turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] 
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2]
                    state_msg.pose.orientation.w = quaternion[3]

                # IF even markers place on right hand side (inner turn radius)
                else:
                    quaternion = quaternion_from_euler(((-math.radians(turn_angle)) * ((i/2)-1) * (1/marker_pairs_in_turn)) + math.radians(self.origin_rotation) - math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    state_msg.pose.position.x = (math.sin(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + turn_center_x
                    state_msg.pose.position.y = (math.cos(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + -turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0]
                    state_msg.pose.orientation.y = quaternion[1] 
                    state_msg.pose.orientation.z = quaternion[2]
                    state_msg.pose.orientation.w = quaternion[3]    

            # Set remaining marker pairs (out of 15 pairs per segment) in a straight after the turn
            elif i > marker_pairs_in_turn*2 :
                # IF odd markers place on left hand side
                if i % 2 == 1:
                    quaternion = quaternion_from_euler(-math.radians(turn_angle) + math.radians(self.origin_rotation) + math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    x = outer_turn_radius
                    y = -(((i-(marker_pairs_in_turn*2))/2) - 0.5)
                    x_ = x* math.cos(math.radians(90-turn_angle)) - y*math.sin(math.radians(90-turn_angle))
                    y_ = y* math.cos(math.radians(90-turn_angle)) + x*math.sin(math.radians(90-turn_angle)) 
                    state_msg.pose.position.x = x_ + turn_center_x
                    state_msg.pose.position.y = y_ - turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] 
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] 
                    state_msg.pose.orientation.w = quaternion[3]

                # IF even markers place on right hand side
                else:
                    quaternion = quaternion_from_euler(-math.radians(turn_angle) + math.radians(self.origin_rotation) - math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    x = inner_turn_radius
                    y = -((i-(marker_pairs_in_turn*2))/2) + 1
                    x_ = x* math.cos(math.radians(90-turn_angle)) - y*math.sin(math.radians(90-turn_angle))
                    y_ = y* math.cos(math.radians(90-turn_angle)) + x*math.sin(math.radians(90-turn_angle)) 
                    state_msg.pose.position.x = x_ + turn_center_x
                    state_msg.pose.position.y = y_ - turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] 
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2]
                    state_msg.pose.orientation.w = quaternion[3]

            # Translate and rotate segment to START relative to the END of previous segment
            x_rotated = state_msg.pose.position.x* math.cos(math.radians(self.origin_rotation)) - state_msg.pose.position.y*math.sin(math.radians(self.origin_rotation))
            y_rotated = state_msg.pose.position.y* math.cos(math.radians(self.origin_rotation)) + state_msg.pose.position.x*math.sin(math.radians(self.origin_rotation)) 
            state_msg.pose.position.x = x_rotated + self.origin_x + x_position_variation
            state_msg.pose.position.y = y_rotated + self.origin_y + y_position_variation
            rospy.wait_for_service('/gazebo/set_model_state')
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            
            # If not zero marker, store position of marker
            if i != 0:
                self.markers_x.append(round(state_msg.pose.position.x,2))
                self.markers_y.append(round(state_msg.pose.position.y,2))

        # Update "origin" for next segment to build from
        origin_x_rotated = next_origin_x* math.cos(math.radians(self.origin_rotation)) - next_origin_y*math.sin(math.radians(self.origin_rotation))
        origin_y_rotated = next_origin_y* math.cos(math.radians(self.origin_rotation)) + next_origin_x*math.sin(math.radians(self.origin_rotation)) 
        self.origin_x += origin_x_rotated
        self.origin_y += origin_y_rotated
        self.origin_rotation += next_origin_rotation

     # Set left turn segment 
    def SetLeftCurve(self,segment_id,turn_angle=DEFAULT_TURN_ANGLE, inner_turn_radius=DEFAULT_INNER_TURN_RADIUS, marker_pairs_in_turn=DEFAULT_MARKER_PAIRS_IN_TURN,Noise=False):

        next_origin_x = 0
        next_origin_y = 0
        next_origin_rotation = 0

        # Calculate turn radii and center of circle that turn is calculated from
        outer_turn_radius = inner_turn_radius + (2 * MARKER_SEPERATION)
        turn_center_x = MARKER_SEPERATION
        turn_center_y = (-MARKER_SEPERATION - inner_turn_radius)
        
        for i in range(LAST_MARKER_ID + 1):

            # Alternate heights of markers to increase visibility
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
            # As only 2 sets of identical markers used, sets distinguished with "_0"
            if segment_id == 0 or i == 0:
                state_msg.model_name = 'aruco_visual_marker_{}'.format(i)
            else:
                state_msg.model_name = 'aruco_visual_marker_{}_{}'.format(i,segment_id-1)

            # If noise = True, add random variation to positon and rotation
            if Noise == True:
                rotation_variation   = (2 * random.random() - 1) * MARKER_ROTATION_VARIATION_LIMIT
                x_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
                y_position_variation = (2 * random.random() - 1) * MARKER_SEPERATION_VARIATION_LIMIT
            else:
                rotation_variation = 0
                x_position_variation = 0
                y_position_variation = 0
            
            # Set markers used in actual curve of turn
            if i <= marker_pairs_in_turn*2:
                # If marker ID = 0
                if i == 0:
                    quaternion = quaternion_from_euler(math.radians(turn_angle)+ math.radians(self.origin_rotation), -math.pi/2, 0)
                    x = ((outer_turn_radius + inner_turn_radius)/2)
                    y = (((LAST_MARKER_ID-(marker_pairs_in_turn*2))/2) - 0.5) + 2.5
                    x_ = x* math.cos(math.radians(270+turn_angle)) - y*math.sin(math.radians(270+turn_angle))
                    y_ = y* math.cos(math.radians(270+turn_angle)) + x*math.sin(math.radians(270+turn_angle)) 
                    state_msg.pose.position.x = x_ + turn_center_x
                    state_msg.pose.position.y = y_ - turn_center_y
                    state_msg.pose.position.z = 0.6
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
                
                # IF odd markers place on left hand side (inner turn radius)
                elif i % 2 == 1:
                    quaternion = quaternion_from_euler(((math.radians(turn_angle)) * ((i/2)-0.5) * (1/marker_pairs_in_turn))+ math.radians(self.origin_rotation) + math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    state_msg.pose.position.x = (math.sin(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) + turn_center_x
                    state_msg.pose.position.y = -(math.cos(math.radians(((i/2)-0.5)*turn_angle/marker_pairs_in_turn)) * inner_turn_radius) - turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] 
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] 
                    state_msg.pose.orientation.w = quaternion[3]

                # IF even markers place on right hand side (outer turn radius)
                else: 
                    quaternion = quaternion_from_euler(((math.radians(turn_angle)) * ((i/2)-1) * (1/marker_pairs_in_turn))+ math.radians(self.origin_rotation) - math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    state_msg.pose.position.x = (math.sin(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) + turn_center_x
                    state_msg.pose.position.y = -(math.cos(math.radians(((i/2)-1)*turn_angle/marker_pairs_in_turn)) * outer_turn_radius) - turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] 
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] 
                    state_msg.pose.orientation.w = quaternion[3]
            
            # Set remaining marker pairs (out of 15 pairs per segment) in a straight after the turn
            elif i > marker_pairs_in_turn*2 :
                # IF odd markers place on left hand side
                if i % 2 == 1:
                    quaternion = quaternion_from_euler(math.radians(turn_angle) + math.radians(self.origin_rotation) + math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    x = inner_turn_radius
                    y = (((i-(marker_pairs_in_turn*2))/2) - 0.5)
                    x_ = x* math.cos(math.radians(270+turn_angle)) - y*math.sin(math.radians(270+turn_angle))
                    y_ = y* math.cos(math.radians(270+turn_angle)) + x*math.sin(math.radians(270+turn_angle)) 
                    state_msg.pose.position.x = x_ + turn_center_x
                    state_msg.pose.position.y = y_ - turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] 
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] 
                    state_msg.pose.orientation.w = quaternion[3]

                # IF even markers place on right hand side
                else:
                    quaternion = quaternion_from_euler(math.radians(turn_angle) + math.radians(self.origin_rotation) - math.radians(MARKER_ROTATION_BASE) + math.radians(rotation_variation), -math.pi/2, 0)
                    x = outer_turn_radius
                    y = ((i-(marker_pairs_in_turn*2))/2) - 1
                    x_ = x* math.cos(math.radians(270+turn_angle)) - y*math.sin(math.radians(270+turn_angle))
                    y_ = y* math.cos(math.radians(270+turn_angle)) + x*math.sin(math.radians(270+turn_angle)) 
                    state_msg.pose.position.x = x_ + turn_center_x
                    state_msg.pose.position.y = y_ - turn_center_y
                    state_msg.pose.position.z = z
                    state_msg.pose.orientation.x = quaternion[0] 
                    state_msg.pose.orientation.y = quaternion[1]
                    state_msg.pose.orientation.z = quaternion[2] 
                    state_msg.pose.orientation.w = quaternion[3]
                    
            # Translate and rotate segment to START relative to the END of previous segment
            x_rotated = state_msg.pose.position.x* math.cos(math.radians(self.origin_rotation)) - state_msg.pose.position.y*math.sin(math.radians(self.origin_rotation))
            y_rotated = state_msg.pose.position.y* math.cos(math.radians(self.origin_rotation)) + state_msg.pose.position.x*math.sin(math.radians(self.origin_rotation)) 
            state_msg.pose.position.x = x_rotated + self.origin_x + x_position_variation
            state_msg.pose.position.y = y_rotated + self.origin_y + y_position_variation
            rospy.wait_for_service('/gazebo/set_model_state')
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

            # If not zero marker, store position of marker
            if i != 0:
                self.markers_x.append(round(state_msg.pose.position.x,2))
                self.markers_y.append(round(state_msg.pose.position.y,2))

        # Update "origin" for next segment to build from
        origin_x_rotated = next_origin_x* math.cos(math.radians(self.origin_rotation)) - next_origin_y*math.sin(math.radians(self.origin_rotation))
        origin_y_rotated = next_origin_y* math.cos(math.radians(self.origin_rotation)) + next_origin_x*math.sin(math.radians(self.origin_rotation)) 
        self.origin_x += origin_x_rotated
        self.origin_y += origin_y_rotated
        self.origin_rotation += next_origin_rotation

    # Set segment by a given track segment ID (-6 to 6, see TOP of file for more information)
    def SetBySegmentId(self, track_segment_id, segment, noise=False):
        if track_segment_id == 0:
            self.SetStraight(segment,noise)
        elif track_segment_id < 0:
            track_segment_id *= -1
            self.SetLeftCurve(segment,15*track_segment_id,10,(math.floor((track_segment_id/6)*8))+2, noise)
        else:
            self.SetRightCurve(segment,15*track_segment_id,10,(math.floor((track_segment_id/6)*8))+2, noise)

    # Reset origin (where next segment will build relative to)
    def ResetOrigin(self,x=0,y=0,rotation=0):
        self.origin_x = x
        self.origin_y = y
        self.origin_rotation = rotation

    # Get marker positions and centerline (list of centerpoints calculated as midpoints of corresponding marker pairs) information
    def GetTrackInfo(self):
        centerline_x = []
        centerline_y = []
        
        for i in range(0,len(self.markers_x)-1,2):
            centerline_x.append((self.markers_x[i] + self.markers_x[i+1])/2)
            centerline_y.append((self.markers_y[i] + self.markers_y[i+1])/2)

        return self.markers_x, self.markers_y, centerline_x, centerline_y

    # Clear marker position arrays
    def ClearMarkerPositions(self):
        self.markers_x = []
        self.markers_y = []


