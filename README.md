<h1 align="center">
  <br>
Racing Towards Reinforcement Learning based control of an
Autonomous Formula SAE Car
  <br>
 </h1>

## Instructions To Run The Code
Please refer to [instructions.md](https://github.com/UoA-CARES/mobile_robot_learning_testbed/blob/main/instructions.md) for in-depth instructions on how to run the code and replicate our experiments.

## State-Space
Represented by x (lateral) and z (forward) displacements of 6 closest (sorted by z displacement) Aruco markers relative to robot position.

## Action-Space
Linear Speed = Constant 0.5 m/s for both models.

DQN - Discrete Action Space +/- 0.2 rad/s rotatational speed.
TD3 - Continuous Action Space from +0.4 rad/s to -0.4 rad/s rotational speed.

## Results

Both models trained and tested in simulation. Example of simulation POV of robot shown below.

![](https://github.com/UoA-CARES/mobile_robot_learning_testbed/blob/main/images/simulation-view.png)

Results of training:
![](https://github.com/UoA-CARES/mobile_robot_learning_testbed/blob/main/images/results.png)

## Video
A video showing the robot executing the tasks during the training process can be found at [this youtube link.](https://youtu.be/TYCrK8-dnqE)

## Citation
If you use either the code or data in your paper, please kindly star this repo and cite our paper
Cite this paper as: 
Coming soon

## Contact
Please feel free to contact us or open an issue if you have questions or need additional explanations.
