import random
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
from tracks import SetStraight, SetRightCurve, SetLeftCurve

MARKER_POSITION_BASE = 1
MARKER_POSITION_VARIATION_LIMIT = 0.05

MARKER_ROTATION_BASE = 0.2
MARKER_ROTATION_VARIATION_LIMIT = 0.1

INNER_TURN_RADIUS = 10
OUTER_TURN_RADIIS = INNER_TURN_RADIUS + (2 * MARKER_POSITION_BASE)

MARKER_PAIRS_IN_TURN = 15

LAST_MARKER_ID = 30

TURN_ANGLE = 45

ACTION_RATE = 5 # number of actions/sec

REWARD_SCALAR = 10 #max reward value given when angle = 0 and distance = 0
ANGLE_REWARD_DROPOFF = 6 #larger value = faster drop off of reward for increasing angle (frequency of cos wave)
DISTANCE_REWARD_DROPOFF = 4 #larger value = faster drop off of reward for increasing distance (frequency of cos wave)
ANGLE_TO_DISTANCE_REWARD_PRIORITY = 0.15 #ratio of total reward that is prioritized from angle reward (distance reward is 1 - ANGLE_TO_DISTANCE_REWARD_PRIORITY)

class ArucoPositionInfo():
	def __init__(self):
		self.x = 100.0
		self.z = 100.0
		self.id = 0


class FSAE_Env():

    def __init__(self):
        #rospy.init_node('reset_world')
        rospy.init_node('FSAE_Env', anonymous=False)

        #observation shape = (12,)
        self.observation_space = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        self.action_space = spaces.Discrete(2)
        #self.bumped = False

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
        # Sleep just makes sure TurtleBot receives the stop command
        # prior to shutting
        # down the script
        rospy.sleep(1)
        
    def MarkerCallback(self, data):
        self.markers = data
        self.num_of_markers = len(self.markers.marker_ids)
        self.state_updated = True

    def reset(self, Noise=False):
        self.no_marker_count = 0
        self.finish_line_detected = False
        self.state_updated = False
        
        if Noise == False:
            self.ResetRobotPosition()
        else:
            self.ResetRobotPosition(Noise=True)
            SetRightCurve()
            #SetStraight(Noise=True)

        while not self.state_updated:
            pass

        self.GetCurrentState() #get new state and store in self.current_state variable
        return self.current_state

    def GetRobotPosition(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp1 = gms("mobile_base","map")
        x = round(resp1.pose.position.x,2)
        y = round(resp1.pose.position.y,2)
        return x, y


    def step(self, action):
        move_cmd = Twist()
        done = False
        prev_state = self.current_state #store prev state in variable
        action_taken = self.action # store previous action in variable
        self.GetCurrentState() #get new state and store in self.current_state variable
        self.action = action #store current action taken in self.action variable
        self.current_reward = self.CalculateCrossReward() #get reward for current state
        
        if self.finish_line_detected == True:
                done = True
                #self.current_reward += 1000

        if self.current_reward == 0:
            self.no_marker_count += 1
            if self.no_marker_count > 10:
                done = True

        # if self.bumped:
        #     done = True
        #     self.current_reward += 1000

        if not done:
            if action == 0:
                move_cmd.angular.z = 0.1
            else:
                move_cmd.angular.z = -0.1

        #constant forward velocity
        move_cmd.linear.x = 0.5
        self.cmd_vel.publish(move_cmd)
        self.r.sleep()
        x , y = self.GetRobotPosition()

        return self.current_state, self.current_reward, done, x, y

    def GetCurrentState(self):
        saved_markers = 0
        current_state = []
        for i in range(self.num_of_markers):
            if saved_markers < 6:
                if self.markers.marker_ids[i] == 0:
                    continue

                current_state.append(self.markers.marker_poses[i].position.x)
                current_state.append(self.markers.marker_poses[i].position.z)
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
                
    def FindLowestPair(self):
		#if first_marker_index and second_marker_index >= 0 then pair found
        self.first_marker_index = -1

        for index in range(self.num_of_markers - 1):
            first_marker_id = self.markers.marker_ids[index]
            if (first_marker_id % 2 == 1):
                if(self.markers.marker_ids[index + 1] == first_marker_id + 1):
                    self.first_marker_index = index
                    #print("ids = " + str(self.markers.marker_ids[index]) + " & " + str(self.markers.marker_ids[index+1]))
                    break

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
        self.FindLowestPair()
        if(self.first_marker_index < 0):
            return 0

        midpoint = self.CalculateMidPoint(self.markers.marker_poses[self.first_marker_index],self.markers.marker_poses[self.first_marker_index + 1])
        angleToTarget = self.CalculateAngleTo(midpoint)
        angle_reward = REWARD_SCALAR*math.cos(angleToTarget * ANGLE_REWARD_DROPOFF)

        if angle_reward < 0:
            angle_reward = 0

        return angle_reward

    def CalculateCrossReward(self):
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


    def ResetRobotPosition(self, Noise=False):
        if Noise == True:
            robot_position_variation = random.random() - 0.5
        else:
            robot_position_variation = 0

        state_msg = ModelState()
        state_msg.model_name = 'mobile_base'
        state_msg.pose.position.x = -1.5 #-1.5
        state_msg.pose.position.y = robot_position_variation
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)



if __name__ == '__main__':
    env = FSAE_Env()
    env.reset()



