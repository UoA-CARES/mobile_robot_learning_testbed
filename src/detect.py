#! /usr/bin/env python
from re import A
from turtle import backward, distance
import rospy
import math
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, Twist
from cares_msgs.msg import ArucoMarkers

REWARD_SCALAR = 10 #max reward value given when angle = 0 and distance = 0
ANGLE_REWARD_DROPOFF = 6 #larger value = faster drop off of reward for increasing angle (frequency of cos wave)
DISTANCE_REWARD_DROPOFF = 4 #larger value = faster drop off of reward for increasing distance (frequency of cos wave)
ANGLE_TO_DISTANCE_REWARD_PRIORITY = 0.15 #ratio of total reward that is prioritized from angle reward (distance reward is 1 - ANGLE_TO_DISTANCE_REWARD_PRIORITY)

class ArucoPositionInfo():
	def __init__(self):
		self.x = 100.0
		self.z = 100.0
		self.id = 0

class Detector():
	def __init__(self):
		# Initiliaze
		rospy.init_node('Detector', anonymous=False)

		# Tell user how to stop TurtleBot
		rospy.loginfo("To stop TurtleBot CTRL + C")

		# What function to call when you ctrl + c    
		rospy.on_shutdown(self.shutdown)
		
		self.detector_sub = rospy.Subscriber("/camera/markers", ArucoMarkers, self.MarkerCallback)
		self.teleop_sub = rospy.Subscriber("/cmd_vel_mux/input/teleop", Twist, self.TeleopCallback)
		self.key_pressed_sub = rospy.Subscriber("/key_pressed", Bool, self.KeyPressedCallback)


		self.markers = ArucoMarkers()
		self.action = Twist()
		self.num_of_markers = 0
		self.key_pressed = False
		self.count = 0

		#store previous iteration values
		self.prev_marker_positions = []
		self.prev_action = Twist()
		self.prev_reward = 0

		self.current_marker_positions = [ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo(), ArucoPositionInfo()]
		self.current_action = Twist()
		self.current_reward = 0	

		#reset data.txt file (clear all data)
		#file = open("/home/aakaash/fsae/src/fsae_control/data.txt","w")
		#file.close()

		r = rospy.Rate(10)
		
		self.OpenFile()

		# As long as you haven't ctrl + c keeping doing...
		while not rospy.is_shutdown():
			self.GetCurrentData()
			self.AppendDataToFile()
			#self.CalculateReward()
			r.sleep()


	def shutdown(self):
		# Stop turtlebot
		rospy.loginfo("Stop TurtleBot")
		self.CloseFile()
		# Sleep just makes sure TurtleBot receives the stop command
		# prior to shutting
		# down the script
		rospy.sleep(1)
		
	def MarkerCallback(self, data):
		self.markers = data
		self.num_of_markers = len(self.markers.marker_ids)
		self.ReorderMarkers()

	def TeleopCallback(self, data):
		self.action = data

	def KeyPressedCallback(self, data):
		self.key_pressed = data.data
		
	def CalculateMidPoint(self, mk1, mk2):
		midpoint = Point()
		midpoint.x = (mk1.position.x + mk2.position.x)/2
		midpoint.y = (mk1.position.y + mk2.position.y)/2
		midpoint.z = (mk1.position.z + mk2.position.z)/2
		return midpoint
		
	def CalculateAngleTo(self, targetPosition):
		#calculates absolute angle deviation from facing target position in RADIANS
		return math.atan(abs(targetPosition.x/targetPosition.z))

	def CalculateDistanceTo(self, targetPosition):
		#calculates linear distance to target position
		return math.sqrt((targetPosition.x**2)+(targetPosition.z**2)) 

	def ReorderMarkers(self):
		
		# loop to access each array element
		for i in range(self.num_of_markers):
			# loop to compare array elements
			for j in range(0, self.num_of_markers - i - 1):
				# compare two adjacent elements
				# change > to < to sort in descending order
				if self.markers.marker_ids[j] > self.markers.marker_ids[j + 1]:

					#swap ids in marker_ids array
					temp_id = self.markers.marker_ids[j]
					self.markers.marker_ids[j] = self.markers.marker_ids[j+1]
					self.markers.marker_ids[j+1] = temp_id

					#swap poses in marker_poses array
					temp_pose = self.markers.marker_poses[j]
					self.markers.marker_poses[j] = self.markers.marker_poses[j+1]
					self.markers.marker_poses[j+1] = temp_pose

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

	def CalculateReward(self):
		self.FindLowestPair()

		if(self.first_marker_index < 0):
			return -10 * REWARD_SCALAR

		midpoint = self.CalculateMidPoint(self.markers.marker_poses[self.first_marker_index],self.markers.marker_poses[self.first_marker_index + 1])
		angleToTarget = self.CalculateAngleTo(midpoint)
		distanceToTarget = self.CalculateDistanceTo(midpoint)
		angle_reward = REWARD_SCALAR*math.cos(angleToTarget * ANGLE_REWARD_DROPOFF)
		
		# distance_reward = REWARD_SCALAR* math.cos(distanceToTarget * DISTANCE_REWARD_DROPOFF) dropoff = 1.5 (cos - gives negative values)
		# distance_reward = REWARD_SCALAR* math.exp(distanceToTarget * DISTANCE_REWARD_DROPOFF) dropoff = -0.5 (exp - decent performance)
		distance_reward = REWARD_SCALAR*1.5 - (distanceToTarget * DISTANCE_REWARD_DROPOFF) #dropoff = 2 (linear)
		
		# dont let distance reward go negative	
		if distance_reward < 0:
			distance_reward = 0	
		
		return (ANGLE_TO_DISTANCE_REWARD_PRIORITY*angle_reward) + ((1 - ANGLE_TO_DISTANCE_REWARD_PRIORITY)*distance_reward)
		
		#print ("distance = " + str(distance_reward))
		#print ("angle = " + str(angle_reward))
		#print ("total = " + str((ANGLE_TO_DISTANCE_REWARD_PRIORITY*angle_reward) + ((1 - ANGLE_TO_DISTANCE_REWARD_PRIORITY)*distance_reward)) + "\n")


	def AppendDataToFile(self):
		#print only if teleop keys pressed
		if(self.key_pressed and self.count > 5):
			data_line = ""
			#write marker x,z values
			for i in range(6):
				data_line += "%d %.3f %.3f " % (self.prev_marker_positions[i].id, self.prev_marker_positions[i].x, self.prev_marker_positions[i].z)
			#write teleop cmd values
			data_line += str(self.prev_action.linear.x) + " " + str(self.prev_action.angular.z) + " "
			#write reward
			data_line += str(self.current_reward) + "\n"
			#print(str(self.current_reward) + "\n")
			
			#data_line += str(self.current_reward - self.prev_reward) + "\n"
			#print(str(self.current_reward - self.prev_reward) + "\n")
			#write data to file	
			#print(data_line + "\n")
			self.AppendToFile(data_line)
		
		self.count+=1

		self.prev_action = self.current_action
		self.prev_marker_positions = self.current_marker_positions
		self.prev_reward = self.current_reward
		

	def OpenFile(self):
		#open file
		self.file_object = open("/home/aakaash/fsae/src/fsae_control/data.txt", "a")

	def CloseFile(self):
		self.file_object.write("End Episode\n")
		self.file_object.close()

	def AppendToFile(self, data):
		self.file_object.write(data)

	def GetCurrentData(self):
		saved_markers = 0
		marker_info = []
		for i in range(self.num_of_markers):
			if saved_markers < 6:
				marker_pos = ArucoPositionInfo()
				marker_pos.id = self.markers.marker_ids[i]
				marker_pos.x = self.markers.marker_poses[i].position.x
				marker_pos.z = self.markers.marker_poses[i].position.z
				marker_info.append(marker_pos)
				saved_markers += 1

		while saved_markers < 6:
			marker_pos = ArucoPositionInfo()
			marker_info.append(marker_pos)
			saved_markers += 1

		self.current_marker_positions = marker_info
		self.current_action = self.action
		self.current_reward = self.CalculateReward()

		

	
	def PrintAllMarkers(self, markerPositions):
		print("Number of Markers = %d" % self.num_of_markers)
		for i in range(self.num_of_markers):
			print("Marker %d : \nX = %.3f \nY = %.3f \nZ = %.3f \n\n\n" % (self.markers.marker_ids[i],self.markers.marker_poses[i].position.x,self.markers.marker_poses[i].position.y,self.markers.marker_poses[i].position.z))

if __name__ == '__main__':
	Detector()

