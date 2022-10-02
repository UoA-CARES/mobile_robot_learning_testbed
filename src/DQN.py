"""
Authors: Aakaash Salvaji, Harry Taylor
The University of Auckland

DQN Model
Task: Autonomous Control of a Turtlebot2 as a racecar
NOTE: Run Training with specified FOLDER name then run testing using same FOLDER name
"""

from re import L #TODO: Do we need this?
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import DQN_env
import os

env = DQN_env.DQN_Env()

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Folder name which to create/load Models and Plots, generated from training
FOLDER = "DQN_TEST"  

# Frequency at which to generate plot of average rewards and save model
PLOT_FREQ = 500
# Number of episodes to average rewards by for plotting
AVERAGE_REWARD_LAST_N_EPISODES = 100
# Frequency at which to save robot path to file
PATH_POS_FREQ = 50



EPISODES = 15000
STEPS = 10000
LEARNING_RATE = 0.0001
MEM_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.95
# Training parameters
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.9999
EXPLORATION_MIN = 0.001 
# Testing parameters TODO: Change parameters implicitly
EXPLORATION_MAX = 0.0
EXPLORATION_DECAY = 0.9999
EXPLORATION_MIN = 0.0 

FC1_DIMS = 1024
FC2_DIMS = 512

DEVICE = torch.device("cuda")

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = action_space

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self):
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network()

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return env.action_space.sample()
        
        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        # Take 10 samples to learn from to increase speed of learning
        for i in range(10):
            states, actions, rewards, states_, dones = self.memory.sample()
            states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
            actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
            dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
            batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

            q_values = self.network(states)
            next_q_values = self.network(states_)
            
            predicted_value_of_now = q_values[batch_indices, actions]
            predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
            
            q_target = rewards + GAMMA * predicted_value_of_future * dones

            loss = self.network.loss(q_target, predicted_value_of_now)
            self.network.optimizer.zero_grad()
            loss.backward()
            self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate

#Function to train DQN Model, Noise = track noise in training
def train(episodes, steps, folder=FOLDER, noise=False):

    best_reward = 0
    rewards = []
    episode_number = []
    average_reward= []

    agent = DQN_Solver()

    # Creates plots & models folder, on level above current folder, and creates subfolders to store plots and models of this training
    # Ensure new folder name is given to each training session, as data will be appended/overwritten if using existing folder name
    if not os.path.exists(f"../plots/{folder}"):
        os.makedirs(f"../plots/{folder}")
    
    if not os.path.exists(f"../models/{folder}/best"):
        os.makedirs(f"../models/{folder}/best")

    for episode in range(1, episodes+1):

        # Give 2 random "segmentId"s which determine shape of training track section for each episode (see TrackGenerator for more on segmentIDs)
        seg1Id = math.ceil(random.random()* 13) - 7
        seg2Id = math.ceil(random.random()* 13) - 7
        trackSegmentList = [seg1Id, seg2Id]
        
        # Stores track information in environment, and returns marker postions
        markers_x, markers_y = env.SetTrackSegmentList(trackSegmentList,noise)
        seg1Id = trackSegmentList[0]
        seg2Id = trackSegmentList[1]


        state = env.reset()
        
        # Arrays to store robot position at each time step to write to path.txt file
        robot_x = []
        robot_y = []

        state = np.reshape(state, [1, observation_space])
        episode_reward = 0
        sum_error = 0

        for step in range(1, steps+1):
            action = agent.choose_action(state)
            state_, reward, done, x, y, error, completion  = env.step(action) 
            state_ = np.reshape(state_, [1, observation_space])
            agent.memory.add(state, action, reward, state_, done)
            agent.learn()
            state = state_
            episode_reward += reward
            sum_error += error

            robot_x.append(x)
            robot_y.append(y)

            if done or step == steps:
                # Store best episode (highest reward) model (overwritten)
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(agent.network.state_dict(),f"../models/{folder}/best/{folder}_dqn_best.pth")
                    file_object = open(f"../models/{folder}/best/best.txt", "a")
                    file_object.write(f"Best Ep = {episode}\n")
                    file_object.close()
                
                average_ep_error =  sum_error/step

                # Write episode information to rewards.txt
                WriteEpisodeRewardToFile(folder, episode_reward, average_ep_error, completion, seg1Id, seg2Id, step)

                rewards.append(episode_reward)
                average_reward_current_episode = np.mean(rewards[-AVERAGE_REWARD_LAST_N_EPISODES:])
                episode_number.append(episode)
                average_reward.append(average_reward_current_episode)

                print(f"Ep {episode} Avg {average_reward_current_episode} Best {best_reward} Last {episode_reward} Epsilon {agent.returning_epsilon()}, Seg1 {seg1Id*15} Seg2 {seg2Id*15}, Completion {completion}%")

                # Save new model, and plot of average reward (from last AVERAGE_REWARD_LAST_N_EPISODES episodes) every PLOT_FREQ episodes to show training progression
                if episode % PLOT_FREQ == 0:
                    plt.figure(1)
                    plt.plot(episode_number, average_reward)
                    plt.title(f"DQN {episode} episodes")
                    plt.savefig(f"../plots/{folder}/{folder}_dqn_{episode}_episodes.png")

                    torch.save(agent.network.state_dict(),f"../models/{folder}/{folder}_dqn_{episode}.pth")
                
                # Append marker postions, and robot positions at each step showing path taken for every PATH_POS_FREQ episodes to path.txt
                if episode % PATH_POS_FREQ == 0:

                    file_object = open(f"../plots/{folder}/path.txt", "a")

                    file_object.write(f"Ep{episode} {seg1Id} {seg2Id} {step} {completion} {average_ep_error}\n")

                    for i in range(30*len(trackSegmentList)):
                        file_object.write(f"{i+1} {markers_x[i]} {markers_y[i]}\n")

                    for pos in range(len(robot_x)):
                        file_object.write(f"0 {robot_x[pos]} {robot_y[pos]}\n")

                    file_object.write(f"End Episode {episode}\n")

                    file_object.close()

                break
    plt.close()

#Function to test DQN Model, Noise = track noise in testing
def test(episodes,steps, track,folder=FOLDER,noise=False):

    agent = DQN_Solver()

    # Load model 5000 from folder created in training (CAN CHANGE EPISODE TO WHATEVER YOU WANT)
    agent.network.load_state_dict(torch.load(f"../models/{folder}/{folder}_dqn_5000.pth"))
    agent.network.eval()

    for episode in range(1, episodes+1):
        
        # Track shape passed in as variable and track information stored in environment, and returns marker postions
        trackSegmentList = track
        markers_x, markers_y = env.SetTrackSegmentList(trackSegmentList,noise)
        seg1Id = trackSegmentList[0]
        seg2Id = trackSegmentList[1]
        
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        episode_reward = 0
        sum_error = 0      

        # Arrays to store robot position at each time step to write to path.txt file
        robot_x = []
        robot_y = []

        for step in range(1, steps+1):
            action = agent.choose_action(state)
            state_, reward, done, x, y, error, completion  = env.step(action) 
            state_ = np.reshape(state_, [1, observation_space])
            episode_reward += reward
            state = state_
            sum_error += error
            print(f"-------Episode:{episode} Step:{step} Action:{action} Completion:{round(completion,1)}%---------")

            robot_x.append(x)
            robot_y.append(y)
            
            if done or step == steps:
                average_ep_error =  sum_error/step
                # Write episode information to test.txt
                WriteTestRewardsToFile(folder, episode_reward, average_ep_error, completion, seg1Id, seg2Id, step)
                
                # Append marker postions, and robot positions at each step showing path taken for every PATH_POS_FREQ episodes to test_path.txt
                file_object = open(f"../plots/{folder}/test_path.txt", "a")

                file_object.write(f"Ep{episode} {seg1Id} {seg2Id} {step} {completion} {average_ep_error}\n")

                for i in range(30*len(trackSegmentList)):
                    file_object.write(f"{i+1} {markers_x[i]} {markers_y[i]}\n")

                for pos in range(len(robot_x)):
                    file_object.write(f"0 {robot_x[pos]} {robot_y[pos]}\n")

                file_object.write(f"End Episode {episode}\n")

                file_object.close()

                break

        print("Episode total reward:", episode_reward)

# Funciton to append each training episodes information to rewards.txt
def WriteEpisodeRewardToFile(folder, reward, avg_error, completion, seg1, seg2, steps):
    reward_file = open(f"../plots/{folder}/rewards.txt", "a")
    reward_file.write(f"{reward} {avg_error} {completion} {seg1} {seg2} {steps}\n")
    reward_file.close()

# Funciton to append each testing episodes information to test.txt
def WriteTestRewardsToFile(folder, reward, avg_error, completion, seg1, seg2, steps):
    reward_file = open(f"../plots/{folder}/test.txt", "a")
    reward_file.write(f"{reward} {avg_error} {completion} {seg1} {seg2} {steps}\n")
    reward_file.close()
            
