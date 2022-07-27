import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import fsae_env
from time import sleep
from tracks import TrackGenerator

env = fsae_env.FSAE_Env()

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

PLOT_FREQ = 500
PATH_POS_FREQ = 100
AVERAGE_REWARD_LAST_N_EPISODES = 100

EPISODES = 10000
STEPS = 1000
LEARNING_RATE = 0.0001
MEM_SIZE = 50000 # 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_MAX = 1.0 #1.0
EXPLORATION_DECAY = 0.9999 # 0.999
EXPLORATION_MIN = 0.001 #0.001

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")

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



def trainStraight():

    best_reward = 0
    rewards = []
    episode_number = []
    average_reward= []

    agent = DQN_Solver()
    #agent.network.load_state_dict(torch.load("../../../../models/all_tracks/moving_cartpole_dqn_2000.pth"))
    #agent.network.eval()

    trackGenerator = TrackGenerator()

    for episode in range(1, EPISODES+1):
        state, seg1Id, seg2Id = env.reset(Noise=True)
        robot_x = []
        robot_y = []

        state = np.reshape(state, [1, observation_space])
        score = 0

        for step in range(1, STEPS):
            action = agent.choose_action(state)
            state_, reward, done, x, y  = env.step(action) 
            state_ = np.reshape(state_, [1, observation_space])
            agent.memory.add(state, action, reward, state_, done)
            agent.learn()
            state = state_
            score += reward

            

            robot_x.append(x)
            robot_y.append(y)

            if done or step == STEPS - 1:
                if score > best_reward:
                    best_reward = score

                WriteEpisodeRewardToFile(score)

                rewards.append(score)
                average_reward_current_episode = np.mean(rewards[-AVERAGE_REWARD_LAST_N_EPISODES:])
                episode_number.append(episode)
                average_reward.append(average_reward_current_episode)

                print("Ep {} Avg {} Best {} Last {} Epsilon {}, Seg1 {} Seg2 {}".format(episode, average_reward_current_episode, best_reward, score, agent.returning_epsilon(), seg1Id*15, seg2Id*15))

                if episode % PLOT_FREQ == 0:
                    plt.figure(1)
                    plt.plot(episode_number, average_reward)
                    plt.title("DQN Straight %s episodes" % str(episode))
                    plt.savefig("../../../../plots/double/moving_cartpole_dqn_{}_episodes.png".format(episode))

                    torch.save(agent.network.state_dict(),"../../../../models/double/moving_cartpole_dqn_{}.pth".format(episode))
                
                if episode % PATH_POS_FREQ == 0:
                    markers_x, markers_y = trackGenerator.GetMarkerPositions()

                    # write robot position path for each episode
                    file_object = open("../../../../plots/double/data.txt", "a")

                    file_object.write("Ep{} ".format(episode) + str(seg1Id) + " " + str(seg2Id) + "\n")

                    for i in range(len(markers_x)):
                        file_object.write(str(i+1) + " " + str(markers_x[i]) + " " + str(markers_y[i]) + "\n")

                    for pos in range(len(robot_x)):
                        file_object.write("0 " + str(robot_x[pos]) + " " + str(robot_y[pos]) + "\n")

                    file_object.write("End Episode {}\n".format(episode))

                    file_object.close()

                break


def WriteEpisodeRewardToFile(reward):
    reward_file = open("../../../../plots/double/rewards.txt", "a")
    reward_file.write(str(reward) + "\n")
    reward_file.close()
            

if __name__ == '__main__':
    trainStraight()
