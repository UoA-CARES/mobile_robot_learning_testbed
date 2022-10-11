"""
Authors:  Aakaash Salvaji, Harry Taylor, David Valencia, Trevor Gee, Henry Williams
The University of Auckland

TD3 Real Environment
Task: Autonomous Control of a Turtlebot2 as a racecar in the real world
"""
import random
from sqlite3 import complete_statement
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from time import sleep
from turtle import position
import numpy as np
from torch import true_divide
import rospy
import math
from gym import spaces
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, Twist, Quaternion
from cares_msgs.msg import ArucoMarkers
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState, GetModelState
from cmath import inf

# Speeds for turtlebot
MAX_TURN_SPEED = 0.1 #0.15
MAX_LINEAR_SPEED = 0.2 #0.25

ACTION_RATE = 5 # number of actions/sec

REWARD_SCALAR = 10 #max reward value given when angle = 0 and distance = 0
ANGLE_REWARD_DROPOFF = 6 #larger value = faster drop off of reward for increasing angle (frequency of cos wave)

class ArucoPositionInfo():
	def __init__(self):
		self.x = 100.0
		self.z = 100.0
		self.id = 0

class TD3_Real_Env():

    def __init__(self):
        rospy.init_node('TD3_Real_Env', anonymous=False)

        self.observation_space = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        self.action_space = spaces.Discrete(2)
      
        self.completion = 0
        # Tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)
        
        self.detector_sub = rospy.Subscriber("/camera/markers", ArucoMarkers, self.MarkerCallback)
        self.cmd_vel = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=10)


        self.markers = ArucoMarkers()
        self.action = Twist()
        self.num_of_markers = 0

        self.current_state = [ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo()]
        self.current_action = Twist()
        self.current_reward = 0	

        self.no_marker_count = 0
        self.finish_line_detected = False
        self.state_updated = False

        self.r = rospy.Rate(ACTION_RATE)



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

    def reset(self, Noise=False):
        self.no_marker_count = 0

        self.GetCurrentState() #get new state and store in self.current_state variable
        return self.current_state
    
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
                done = True

        # If no pairs present for 10 steps, end episode
        if not self.PairPresent():
            self.no_marker_count += 1
            print(self.no_marker_count)
            if self.no_marker_count > 10:
                done = True
        else:
            self.no_marker_count = 0

        if not done:
            move_cmd.angular.z = action*MAX_TURN_SPEED

        #constant forward velocity
        move_cmd.linear.x = MAX_LINEAR_SPEED
        self.cmd_vel.publish(move_cmd)
        self.r.sleep()
        x = 0
        y = 0
        error = 0
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
            #when no aruco marker present use 100 as x and z coordinates
            current_state.append(100)
            current_state.append(100)
            saved_markers += 1

        self.current_state = current_state

        for marker in self.markers.marker_ids:
            if marker == 0:
                self.finish_line_detected = True


    def FindLowestOddAndEven(self):
		#if first_marker_index and second_marker_index >= 0 then pair found
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


    def CalculateReward(self):
        odd, even = self.FindLowestOddAndEven()
        #print(str(self.markers.marker_ids[odd]) + " " + str(self.markers.marker_ids[even]))
        if(odd < 0 or even < 0):
            return 0

        midpoint = self.CalculateMidPoint(self.markers.marker_poses[odd],self.markers.marker_poses[even])
        angleToTarget = self.CalculateAngleTo(midpoint)
        angle_reward = REWARD_SCALAR*math.cos(angleToTarget * ANGLE_REWARD_DROPOFF)

        if angle_reward < 0:
            angle_reward = 0

        return angle_reward

    
    def CalculateMidPoint(self, mk1, mk2):
        midpoint = Point()
        midpoint.x = (mk1.position.x + mk2.position.x)/2
        midpoint.y = (mk1.position.y + mk2.position.y)/2
        midpoint.z = (mk1.position.z + mk2.position.z)/2
        return midpoint
		
    def CalculateAngleTo(self, targetPosition):
		#calculates absolute angle deviation from facing target position in RADIANS
        return math.atan(abs(targetPosition.x/targetPosition.z))


