"""
Authors:  Aakaash Salvaji, Harry Taylor, David Valencia, Trevor Gee, Henry Williams
The University of Auckland

TD3 Environment
Task: Autonomous Control of a Turtlebot2 as a racecar
"""

import random
from sqlite3 import complete_statement
from turtle import position
import numpy as np
from torch import true_divide
import rospy
import math
from gym import spaces
from geometry_msgs.msg import Point, Twist
from cares_msgs.msg import ArucoMarkers
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState, GetModelState
from TrackGenerator import TrackGenerator
from cmath import inf

# Speeds for turtlebot
MAX_TURN_SPEED = 0.4 # rad/s
MAX_LINEAR_SPEED = 0.5 # m/s

ACTION_RATE = 5 # number of actions/sec

REWARD_SCALAR = 10 #max reward value given when angle = 0 and distance = 0
ANGLE_REWARD_DROPOFF = 6 #larger value = faster drop off of reward for increasing angle (frequency of cos wave)

class ArucoPositionInfo():
	def __init__(self):
		self.x = 100.0
		self.z = 100.0
		self.id = 0

class TD3_Env():

    def __init__(self):
        rospy.init_node('TD3_Env', anonymous=False)

        self.segmentList = []
        self.observation_space = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        self.action_space = spaces.Discrete(2)
        self.trackGenerator = TrackGenerator()
        self.completion = 0
        self.current_segment = 1
        self.no_of_segments = 2
        self.max_marker_id = 1
        self.noise = False

        # Tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)
        
        self.detector_sub = rospy.Subscriber("/camera/markers", ArucoMarkers, self.MarkerCallback)
        self.cmd_vel = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=10)

        self.markers = ArucoMarkers()
        self.num_of_markers = 0

        self.current_state = [ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo()]
        self.current_reward = 0	

        self.no_marker_count = 0
        self.finish_line_detected = False
        self.state_updated = False

        self.r = rospy.Rate(ACTION_RATE)

    # Store track information and return absolute marker positions (without noise added although noise added on actual track)
    def SetTrackSegmentList(self, segmentList, noise):
        self.trackGenerator.ResetOrigin()
        self.segmentList = segmentList
        self.no_of_segments = len(segmentList)
        self.trackGenerator.ClearMarkerPositions()
        self.noise = noise

        for i in range(self.no_of_segments):
            segmentId = self.segmentList[i]
            self.trackGenerator.SetBySegmentId(segmentId,i%2)
        
        markers_x, markers_y, self.centerline_x, self.centerline_y = self.trackGenerator.GetTrackInfo()

        return markers_x, markers_y

    # Set first 2 segments of track
    def SetFirstTwoSegments(self):
        self.trackGenerator.ResetOrigin()
        self.trackGenerator.SetBySegmentId(self.segmentList[0], 0, self.noise)
        self.trackGenerator.SetBySegmentId(self.segmentList[1], 1, self.noise)

    # Set next segment of track (called when robot crossed boundary between 2 segments)
    def SetNextSegment(self):
        
        if self.current_segment>=len(self.segmentList):
            self.trackGenerator.SetBySegmentId(0,self.current_segment%2, self.noise)
            return

        segmentId = self.segmentList[self.current_segment]
        self.trackGenerator.SetBySegmentId(segmentId,self.current_segment%2, self.noise)
            

    def shutdown(self):
        # Stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        # Sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)

    # Gets marker information from /camera/markers topic  
    def MarkerCallback(self, data):
        self.markers = data
        self.num_of_markers = len(self.markers.marker_ids)
        self.state_updated = True
        
    # Gets random action
    def generate_sample_act(self):
        # Get random value between -1 and 1 (normalised)
        return np.array([np.clip(random.uniform(-1,1), -1, 1)])

    # Reset track and robot
    def reset(self):

        self.SetFirstTwoSegments()
        self.ResetRobotPosition()  

        # Wait to detect that the state has been updated and visible to the camera 
        self.state_updated = False
        while not self.state_updated:
            pass
        
        # Reset variables
        self.no_marker_count = 0
        self.finish_line_detected = False
        self.completion = 0
        self.current_segment = 1
        self.max_marker_id = 1

        self.GetCurrentState() 
        return self.current_state

    # Get robot position in Gazebo, and calculate error (distance from centerline)
    def GetRobotPosition(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp1 = gms("mobile_base","map")
        x = round(resp1.pose.position.x,4)
        y = round(resp1.pose.position.y,4)
        error, completion = self.CalculateDistanceToCenterline(x,y)
        
        if self.finish_line_detected == True:
            self.completion = 100
        else:
            self.completion = completion

        return x, y, error

    # Calculates robots deviation from centerline by measuring distance to closest midpoint of markerpairs
    def CalculateDistanceToCenterline(self, x, y):
        min_distance_to_center = inf
        for j in range(0, len(self.centerline_x)):
            distance_to_centerline = math.sqrt(pow(x - self.centerline_x[j],2) + pow(y - self.centerline_y[j],2)) 
            if distance_to_centerline < min_distance_to_center:
                min_distance_to_center = distance_to_centerline
                completion =min(100,round(((j+1)/((len(self.centerline_x))))*100,1))

        return min_distance_to_center, completion

    # Check if atleast one odd and one even marker are visible
    def PairPresent(self):
        num_odd = 0
        num_even = 0

        for id in self.markers.marker_ids:
            if id % 2 == 0:
                num_even += 1
            else:
                num_odd += 1
        
        if num_odd == 0 or num_even == 0:
            return False
        
        return True

    def step(self, action):
        move_cmd = Twist()
        done = False
        self.GetCurrentState()
        # Get reward for current state
        self.current_reward = self.CalculateReward()

        # If 0 marker is detected (finish line), end episode
        if self.finish_line_detected == True:
                self.completion = 100
                done = True
        
        # If no pairs present for 10 steps, end episode
        if not self.PairPresent():
            self.no_marker_count += 1
            if self.no_marker_count > 10:
                done = True
        else:
            self.no_marker_count = 0

        if not done:
            # If boundary between 2 segments is crossed, set next segment in track sequence
            self.CheckSegmentCrossed()
            
            # take normalised action (between -1 and 1) and multiply by MAX_TURN_SPEED to get rotational velocity between -MAX_TURN_SPEED and +MAX_TURN_SPEED rad/s
            move_cmd.angular.z = action*MAX_TURN_SPEED
            # Constant forward velocity
            move_cmd.linear.x = MAX_LINEAR_SPEED # 0.5 m/s
            self.cmd_vel.publish(move_cmd)

        # If robot gets too close to a marker, end the episode to avoid collision/going off track
        if self.current_state[1] <= 0.5:
            done = True

        self.r.sleep()
        x , y, error = self.GetRobotPosition()

        if self.completion == 100:
            done = True

        return self.current_state, self.current_reward, done, x, y, error, self.completion

    # Gets closest 6 marker positions (x,z) based on z (forward) distance, if 6 not present, fill state vector with (100,100)
    def GetCurrentState(self):
        saved_markers = 0

        marker_poses = self.markers.marker_poses
        marker_poses.sort(key=lambda x: x.position.z, reverse=False)
        
        current_state = []
        for i in range(self.num_of_markers):
            if saved_markers < 6:
                if self.markers.marker_ids[i] == 0:
                    continue

                current_state.append(marker_poses[i].position.x)
                current_state.append(marker_poses[i].position.z)
                saved_markers += 1

        while saved_markers < 6:
            # When no aruco marker present use 100 as x and z coordinates
            current_state.append(100)
            current_state.append(100)
            saved_markers += 1

        self.current_state = current_state

        # If 0 marker detected
        for marker in self.markers.marker_ids:
            if marker == 0:
                self.finish_line_detected = True

    # If boundary between 2 segments is crossed, set next segment in track sequence
    def CheckSegmentCrossed(self):
        markers = np.array(self.markers.marker_ids)
        if len(markers) > 0:
            max_id = max(markers)
            if self.max_marker_id > 25 and max_id < 15:
                self.current_segment += 1
                self.SetNextSegment()
                self.max_marker_id = max_id
            else:
                if max_id <= 30:
                    self.max_marker_id = max(self.max_marker_id, max_id)
            
            
            for i in range(len(markers)):
                markers[i] = markers[i] + 30*(self.current_segment-1)

                if max_id > 20:
                    if markers[i] < 10 + 30*(self.current_segment-1):
                        markers[i] = markers[i] + 30*(self.current_segment+1)

            min_id = min(markers)


    # Find lowest pair (one odd and one even marker)
    def FindLowestOddAndEven(self):
		# If first_marker_index and second_marker_index >= 0 then pair found
        lowest_odd = -1
        lowest_even = -1

        for index in range(self.num_of_markers - 1):
            marker_id = self.markers.marker_ids[index]
            if marker_id % 2 == 1:
                lowest_odd = index
                break

        for index in range(self.num_of_markers - 1):
            marker_id = self.markers.marker_ids[index]
            if marker_id % 2 == 0:
                lowest_even = index
                break
        
        return lowest_odd, lowest_even

    # Calculate Reward
    def CalculateReward(self):
        odd, even = self.FindLowestOddAndEven()
        # If no pair, then return 0 reward
        if(odd < 0 or even < 0):
            return 0

        # Calculate midpoint and angle to midpoint to determine reward
        midpoint = self.CalculateMidPoint(self.markers.marker_poses[odd],self.markers.marker_poses[even])
        angleToTarget = self.CalculateAngleTo(midpoint)
        angle_reward = REWARD_SCALAR*math.cos(angleToTarget * ANGLE_REWARD_DROPOFF)

        # Min reward = 0
        if angle_reward < 0:
            angle_reward = 0

        return angle_reward

    # Calculate midpoint coordinates between 2 points
    def CalculateMidPoint(self, mk1, mk2):
        midpoint = Point()
        midpoint.x = (mk1.position.x + mk2.position.x)/2
        midpoint.y = (mk1.position.y + mk2.position.y)/2
        midpoint.z = (mk1.position.z + mk2.position.z)/2
        return midpoint
	
    # Calculate angle difference between robot forward direction and a given point in radians
    def CalculateAngleTo(self, targetPosition):
        return math.atan(abs(targetPosition.x/targetPosition.z))

    # Reset robot position (with noise if self.noise = True)
    def ResetRobotPosition(self):
        if self.noise == True:
            robot_position_variation = random.random() - 0.5
        else:
            robot_position_variation = 0

        state_msg = ModelState()
        state_msg.model_name = 'mobile_base'
        state_msg.pose.position.x = 1 
        state_msg.pose.position.y = robot_position_variation
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)



