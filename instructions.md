<h1 align="center">
  <br>
Instructions To Set Up & Run The Code
  <br>
 </h1>

# SETUP
## Install ROS/Turtlebot2 Files
1. Download turtlebot_setup.zip file from: [LINK]
2. Unzip turtlebot_setup.zip file.
3. Open New Terminal
4. Go to turtlebot_setup directory:
> $ cd Downloads/turtlebot_setup	
5. Modify the scripts so they are executable:
> $ chmod +x setup.sh
6. Run the script:
> $ ./setup.sh

If setup successfully completed, you should be able to run Gazebo to see the robot and environment using:
> $ roslaunch turtlebot_gazebo turtlebot_world.launch

## Create Workspace

1. Open New Terminal
2. Setup working directory:
> $ mkdir Directory_Name\
> $ cd Directory_Name\
> $ mkdir src\
> $ cd src
3. Clone this repository:
> $ git clone https://github.com/UoA-CARES/mobile_robot_learning_testbed.git
4. Clone other required repositories:
> $ git clone https://github.com/UoA-CARES/cares_msgs \
> $ git clone https://github.com/maraatech/aruco_detector
5. Build and compile workspace:
> $ cd .. \
> $ catkin_make
6. Define source (so you don't have to do it everytime you open a terminal):
> $ echo "~/Directory_Name/devel/setup.bash" >> ~/.bashrc


## Add Models/World/Launch Files
1. Copy the mobile_robot.world file from the world folder into the folder Home/turtlebot2/src/turtlebot_simulator/turtlebot_gazebo/worlds
2. Copy the mobile_robot.launch file from the launch folder into the folder Home/turtlebot2/src/turtlebot_simulator/turtlebot_gazebo/launch
3. Open the Home folder in file explorer and show hidden files (CTRL+H)
4. Copy all folders in the aruco_marker_models into the folder Home/.gazebo/models 

# Training/Simulation Testing
1. Open New Terminal
2. Launch the mobile robot world:
> $ roslaunch turtlebot_gazebo mobile_robot.launch
3. Run the main.py script in the src folder either through VS Code or a new terminal.

