"""
Description:
            MBPO, model based policy optimization with TD3
            Real Robot Gripper
            Task move the cube
            Using aruco markers for all the detection and measures
"""
from sqlite3 import complete_statement
import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tracks2 import TrackGenerator

from TD3_memory import MemoryClass
from TD3_networks import Actor_NN, Critic_NN, ModelNet_probabilistic_transition
from TD3_env import FSAE_Env
from time import sleep

#from main_rl_env_translation_v2 import RL_ENV

FOLDER = "TD3_TEST"

UPDATE_LATEST_PLOT_FREQ = 50
PLOT_FREQ = 500
PATH_POS_FREQ = 50
AVERAGE_REWARD_LAST_N_EPISODES = 100

EXPLORATION_EPISODES = 100
EPISODES = 15000
STEPS = 2000

class TD3_Agent:

    def __init__(self, env):

        self.env = env
        self.gamma      = 0.99
        self.tau        = 0.005
        self.batch_size = 32

        self.M = 300
        self.G = 10
        self.update_counter = 0
        self.policy_freq_update = 2

        self.max_memory_size_env   = 20_000
        self.max_memory_size_model = 50_000

        self.actor_learning_rate      = 1e-4
        self.critic_learning_rate     = 1e-3
        self.transition_learning_rate = 1e-3

        self.hidden_size_critic = [64, 64, 32]
        self.hidden_size_actor  = [64, 64, 32]

        self.hidden_size_network_model  = [256, 128, 64]

        self.num_states  = 12
        self.num_actions = 1
        #only used for model based
        self.num_states_training = 14  # 16 in total but remove the target point from the state for training trans model

        self.loss_transition_model_1 = []
        self.loss_transition_model_2 = []
        self.loss_transition_model_3 = []
        self.loss_transition_model_4 = []
        self.loss_transition_model_5 = []

        self.epsilon       = 1
        self.epsilon_min   = 0.001
        self.epsilon_decay = 0.0008

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(self.max_memory_size_env, self.max_memory_size_model)

        # ---------- Initialization and build the networks for TD3----------- #
        # Main networks
        self.actor     = Actor_NN(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_q1 = Critic_NN(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)
        self.critic_q2 = Critic_NN(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Target networks
        self.actor_target     = Actor_NN(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target_q1 = Critic_NN(self.num_states + self.num_actions, self.hidden_size_critic,
                                           self.num_actions)
        self.critic_target_q2 = Critic_NN(self.num_states + self.num_actions, self.hidden_size_critic,
                                           self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer    = optim.Adam(self.actor.parameters(),     lr=self.actor_learning_rate)
        self.critic_optimizer_1 = optim.Adam(self.critic_q1.parameters(), lr=self.critic_learning_rate)
        self.critic_optimizer_2 = optim.Adam(self.critic_q2.parameters(), lr=self.critic_learning_rate)


        # ---------- Initialization and build the networks for Model Learning ----------- #
        self.pdf_transition_model_1 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_2 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_3 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_4 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_5 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)

        self.pdf_transition_1_optimizer = optim.Adam(self.pdf_transition_model_1.parameters(),
                                                     lr=self.transition_learning_rate, )
        self.pdf_transition_2_optimizer = optim.Adam(self.pdf_transition_model_2.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_3_optimizer = optim.Adam(self.pdf_transition_model_3.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_4_optimizer = optim.Adam(self.pdf_transition_model_4.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_5_optimizer = optim.Adam(self.pdf_transition_model_5.parameters(),
                                                     lr=self.transition_learning_rate)

    def get_action_from_policy(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
            self.actor.train()
        return action[0]


    def add_real_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_environment_add(state, action, reward, next_state, done)

    def add_imagined_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_model_add(state, action, reward, next_state, done)

    def epsilon_greedy_function_update(self):
        # this is used for choose the sample from memory when dream samples are generated
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)


    def step_training(self, style="MFRL"):
        if style == "MFRL":
            if self.memory.len_env_buffer() <= self.batch_size:
                return
            else:
                self.update_weights(style)

        elif style == "MBRL":
            if self.memory.len_env_buffer() and self.memory.len_model_buffer() <= self.batch_size:
                return
            else:
                self.epsilon_greedy_function_update()
                self.update_weights(style)


    def update_weights(self, style):

        for it in range(1, self.G + 1):
            self.update_counter += 1
            if style == "MFRL":
                states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)

            elif style == "MBRL":
                # I did this because the model predict bad things at the beginning, so I choose and train the policy
                # from previous real experiences and then move to the "dream" experiences using epsilon greedy.
                if np.random.random() <= self.epsilon:
                    states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)
                    print("sample from env data set")
                else:
                    print("sample from model data set")
                    states, actions, rewards, next_states, dones = self.memory.sample_experience_from_model(self.batch_size)

            states  = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones   = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            states  = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones   = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            # ------- compute the target action
            next_actions = self.actor_target.forward(next_states)

            # add noise also here, paper mention this
            next_actions = next_actions.detach().numpy()  # tensor to numpy
            next_actions = next_actions + (np.random.normal(0, scale=0.2, size=self.num_actions))
            next_actions = np.clip(next_actions, -1, 1)
            next_actions = torch.FloatTensor(next_actions)

            # compute next targets values
            next_Q_vales_q1 = self.critic_target_q1.forward(next_states, next_actions)
            next_Q_vales_q2 = self.critic_target_q2.forward(next_states, next_actions)

            q_min = torch.minimum(next_Q_vales_q1, next_Q_vales_q2)

            Q_target = rewards + (self.gamma * (1 - dones) * q_min).detach()

            loss = nn.MSELoss()

            Q_vals_q1 = self.critic_q1.forward(states, actions)
            Q_vals_q2 = self.critic_q2.forward(states, actions)

            critic_loss_1 = loss(Q_vals_q1, Q_target)
            critic_loss_2 = loss(Q_vals_q2, Q_target)

            # Critic step Update
            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            # TD3 updates the policy (and target networks) less frequently than the Q-function
            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss
                actor_loss = - self.critic_q1.forward(states, self.actor.forward(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # ------------------------------------- Update target networks --------------- #

                # update the target networks using tao "soft updates"
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def transition_model_learn(self):
        if self.memory.len_env_buffer() <= self.batch_size:
            return
        else:
            input_state_minibatch  = []
            output_state_minibatch = []

            states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)
            # these values for states and next states include the target point but need to be removed

            states  = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones   = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            # Remove the target point from state and next state because there is no point in predict
            # the target value since it randomly changes in the environment.

            for state in states:
                input_state = state[:-2]
                input_state_minibatch.append(input_state)

            for next_state in next_states:
                output_state = next_state[:-2]
                output_state_minibatch.append(output_state)

            states      = np.array(input_state_minibatch)
            next_states = np.array(output_state_minibatch)

            states  = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones   = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            # ---- Transition Model---- #
            distribution_probability_model_1 = self.pdf_transition_model_1.forward(states, actions)
            distribution_probability_model_2 = self.pdf_transition_model_2.forward(states, actions)
            distribution_probability_model_3 = self.pdf_transition_model_3.forward(states, actions)
            distribution_probability_model_4 = self.pdf_transition_model_4.forward(states, actions)
            distribution_probability_model_5 = self.pdf_transition_model_5.forward(states, actions)

            # calculate the loss
            loss_neg_log_likelihood_1 = - distribution_probability_model_1.log_prob(next_states)
            loss_neg_log_likelihood_2 = - distribution_probability_model_2.log_prob(next_states)
            loss_neg_log_likelihood_3 = - distribution_probability_model_3.log_prob(next_states)
            loss_neg_log_likelihood_4 = - distribution_probability_model_4.log_prob(next_states)
            loss_neg_log_likelihood_5 = - distribution_probability_model_5.log_prob(next_states)

            loss_neg_log_likelihood_1 = torch.mean(loss_neg_log_likelihood_1)
            loss_neg_log_likelihood_2 = torch.mean(loss_neg_log_likelihood_2)
            loss_neg_log_likelihood_3 = torch.mean(loss_neg_log_likelihood_3)
            loss_neg_log_likelihood_4 = torch.mean(loss_neg_log_likelihood_4)
            loss_neg_log_likelihood_5 = torch.mean(loss_neg_log_likelihood_5)

            self.pdf_transition_model_1.train()
            self.pdf_transition_1_optimizer.zero_grad()
            loss_neg_log_likelihood_1.backward()
            self.pdf_transition_1_optimizer.step()

            self.pdf_transition_model_2.train()
            self.pdf_transition_2_optimizer.zero_grad()
            loss_neg_log_likelihood_2.backward()
            self.pdf_transition_2_optimizer.step()

            self.pdf_transition_model_3.train()
            self.pdf_transition_3_optimizer.zero_grad()
            loss_neg_log_likelihood_3.backward()
            self.pdf_transition_3_optimizer.step()

            self.pdf_transition_model_4.train()
            self.pdf_transition_4_optimizer.zero_grad()
            loss_neg_log_likelihood_4.backward()
            self.pdf_transition_4_optimizer.step()

            self.pdf_transition_model_5.train()
            self.pdf_transition_5_optimizer.zero_grad()
            loss_neg_log_likelihood_5.backward()
            self.pdf_transition_5_optimizer.step()

            self.loss_transition_model_1.append(loss_neg_log_likelihood_1.item())
            self.loss_transition_model_2.append(loss_neg_log_likelihood_2.item())
            self.loss_transition_model_3.append(loss_neg_log_likelihood_3.item())
            self.loss_transition_model_4.append(loss_neg_log_likelihood_4.item())
            self.loss_transition_model_5.append(loss_neg_log_likelihood_5.item())

            print("Loss:", loss_neg_log_likelihood_1.item(), loss_neg_log_likelihood_2.item(),
                           loss_neg_log_likelihood_3.item(), loss_neg_log_likelihood_4.item(),
                           loss_neg_log_likelihood_5.item())


    def generate_dream_samples(self):
        if self.memory.len_env_buffer() <= self.batch_size:
            return
        else:
            sample_batch = 1  # Fix to be =1 always
            for _ in range(1, self.M + 1):
                state, _, _, _, _ = self.memory.sample_experience_from_env(sample_batch)
                state = np.array(state)
                state = state[0]  # 16 values
                target_point = state[-2:]  # just target point only

                # Remove the target point from state
                state_input  = state[:-2]  # 14 values
                state_tensor = torch.FloatTensor(state_input)
                state_tensor = state_tensor.unsqueeze(0)  # torch.Size([1, 14])

                action = self.get_action_from_policy(state)  # array  of (4,)
                action = np.clip(action, -1, 1)
                noise  = (np.random.normal(0, scale=0.15, size=4))
                action = action + noise
                action = np.clip(action, -1, 1)

                action_tensor = torch.FloatTensor(action)
                action_tensor = action_tensor.unsqueeze(0)  # torch.Size([1, 4])

                self.pdf_transition_model_1.eval()
                with torch.no_grad():
                    function_generated_1    = self.pdf_transition_model_1.forward(state_tensor, action_tensor)
                    observation_generated_1 = function_generated_1.sample((14,))  # torch.Size([14, 1, 1])
                    observation_generated_1 = torch.reshape(observation_generated_1, (1, 14))  # torch.Size([1, 14])
                    observation_generated_1 = observation_generated_1.detach()
                    observation_generated_1 = observation_generated_1.numpy()
                    self.pdf_transition_model_1.train()

                self.pdf_transition_model_2.eval()
                with torch.no_grad():
                    function_generated_2    = self.pdf_transition_model_2.forward(state_tensor, action_tensor)
                    observation_generated_2 = function_generated_2.sample((14,))  # torch.Size([14, 1, 1])
                    observation_generated_2 = torch.reshape(observation_generated_2, (1, 14))  # torch.Size([1, 14])
                    observation_generated_2 = observation_generated_2.detach()
                    observation_generated_2 = observation_generated_2.numpy()
                    self.pdf_transition_model_2.train()

                self.pdf_transition_model_3.eval()
                with torch.no_grad():
                    function_generated_3    = self.pdf_transition_model_3.forward(state_tensor, action_tensor)
                    observation_generated_3 = function_generated_3.sample((14,))  # torch.Size([14, 1, 1])
                    observation_generated_3 = torch.reshape(observation_generated_3, (1, 14))  # torch.Size([1, 14])
                    observation_generated_3 = observation_generated_3.detach()
                    observation_generated_3 = observation_generated_3.numpy()
                    self.pdf_transition_model_3.train()

                self.pdf_transition_model_4.eval()
                with torch.no_grad():
                    function_generated_4    = self.pdf_transition_model_4.forward(state_tensor, action_tensor)
                    observation_generated_4 = function_generated_4.sample((14,))  # torch.Size([14, 1, 1])
                    observation_generated_4 = torch.reshape(observation_generated_4, (1, 14))  # torch.Size([1, 14])
                    observation_generated_4 = observation_generated_4.detach()
                    observation_generated_4 = observation_generated_4.numpy()
                    self.pdf_transition_model_4.train()

                self.pdf_transition_model_5.eval()
                with torch.no_grad():
                    function_generated_5    = self.pdf_transition_model_5.forward(state_tensor, action_tensor)
                    observation_generated_5 = function_generated_5.sample((14,))  # torch.Size([14, 1, 1])
                    observation_generated_5 = torch.reshape(observation_generated_5, (1, 14))  # torch.Size([1, 14])
                    observation_generated_5 = observation_generated_5.detach()
                    observation_generated_5 = observation_generated_5.numpy()
                    self.pdf_transition_model_5.train()


                model_choose = random.randint(1, 5)
                if model_choose   == 1:
                    next_state_imagined = observation_generated_1[0]
                elif model_choose == 2:
                    next_state_imagined = observation_generated_2[0]
                elif model_choose == 3:
                    next_state_imagined = observation_generated_3[0]
                elif model_choose == 4:
                    next_state_imagined = observation_generated_4[0]
                elif model_choose == 5:
                    next_state_imagined = observation_generated_5[0]

                # calculate the reward and distance based on the prediction and input not using a model-reward
                cube_position   = next_state_imagined[-2:]
                target_position = target_point

                imagined_distance_cube_goal = np.linalg.norm(cube_position - target_position)  # millimeters distance

                if imagined_distance_cube_goal <= 10:  # millimeters
                    done = True
                    reward_d = 100
                else:
                    done = False
                    reward_d = -imagined_distance_cube_goal

                # to complete the 16 values of the original state-space
                next_state = np.append(next_state_imagined, target_position)

                # add imagines experiences to model_imagined buffer
                self.add_imagined_experience_memory(state, action, reward_d, next_state, done)




def run_exploration(env, episodes, horizont, agent, noise=False):
    mode = "Exploration"
    for episode in range(1, episodes+1):
        seg1Id = math.ceil(random.random()* 13) - 7
        seg2Id = math.ceil(random.random()* 13) - 7
        trackSegmentList = [seg1Id, seg2Id]
        
        markers_x, markers_y = env.SetTrackSegmentList(trackSegmentList,noise)
        seg1Id = trackSegmentList[0]
        seg2Id = trackSegmentList[1]
        state = env.reset()
        state = np.array(state)

        for step in range(1, horizont):
            action   = env.generate_sample_act()
            next_state,reward,done, _,_,_, _= env.step(action)
            next_state = np.array(next_state)

            agent.add_real_experience_memory(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        
        print("---- Exploration {}/{} ----".format(episode, episodes))
    print(f"******* -----{episodes} for exploration ended-----********* ")


def run_training(env, episodes, horizont, agent, style, folder=FOLDER, track_noise=False):
    mode        = f"Training TD3 {style}"
    rewards     = []
    best_reward = 0
    rewards = []
    episode_number = []
    average_reward= []

    # TODO: CREATE FOLDER FOR PLOTS AND MODELS AUTOMATICALLY/REMOVE "../../.."
    if not os.path.exists("../../../../plots/{}".format(folder)):
        os.makedirs("../../../../plots/{}".format(folder))
    
    if not os.path.exists("../../../../models/{}/best".format(folder)):
        os.makedirs("../../../../models/{}/best".format(folder))

    for episode in range(1,episodes+1):
        
        seg1Id = math.ceil(random.random()* 13) - 7
        seg2Id = math.ceil(random.random()* 13) - 7
        trackSegmentList = [seg1Id, seg2Id]
        
        markers_x, markers_y = env.SetTrackSegmentList(trackSegmentList,track_noise)
        seg1Id = trackSegmentList[0]
        seg2Id = trackSegmentList[1]

        state = env.reset()
        state = np.array(state)
        robot_x = []
        robot_y = []

        episode_reward = 0
        sum_error = 0

        for step in range(1, horizont+1):
            
            action   = agent.get_action_from_policy(state)
            noise    = np.random.normal(0, scale=0.1, size=1)
            action   = action + noise
            action   = np.clip(action, -1, 1)

            next_state,reward,done,x, y, error, completion= env.step(action)
            next_state = np.array(next_state)

            agent.add_real_experience_memory(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            sum_error += error

            robot_x.append(x)
            robot_y.append(y)
            
            # learn
            if style == "MFRL":
                agent.step_training(style)

            elif style == "MBRL":
                agent.transition_model_learn()
                agent.generate_dream_samples()
                agent.step_training(style)
            
            
            if done or step == horizont:

                if step < 15:
                    completion = 0

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(agent.actor.state_dict(),"../../../../models/{}/best/actor.pth".format(folder))
                    torch.save(agent.critic_q1.state_dict(),"../../../../models/{}/best/critic_q1.pth".format(folder))
                    torch.save(agent.critic_q2.state_dict(),"../../../../models/{}/best/critic_q2.pth".format(folder))
                    file_object = open("../../../../models/{}/best/best.txt".format(folder), "a")
                    file_object.write("Best Ep = {}\n".format(episode))
                    file_object.close()
                
                average_ep_error =  sum_error/step
                WriteEpisodeRewardToFile(folder, episode_reward, average_ep_error, completion, seg1Id, seg2Id, step)

                rewards.append(episode_reward)
                average_reward_current_episode = np.mean(rewards[-AVERAGE_REWARD_LAST_N_EPISODES:])
                episode_number.append(episode)
                average_reward.append(average_reward_current_episode)

                print("Ep {} Avg {} Best {} Last {} , Seg1 {} Seg2 {}, Steps {}, Completion {}%".format(episode, average_reward_current_episode, best_reward, episode_reward, seg1Id*15, seg2Id*15, step, completion))
                
                
                if episode % PLOT_FREQ == 0:
                    plt.figure(1)
                    plt.plot(episode_number, average_reward)
                    plt.title("TD3 %s episodes" % str(episode))
                    plt.savefig("../../../../plots/{}/td3_{}_episodes.png".format(folder, episode))
                    #plt.savefig("../../../../plots/{}/latest_td3.png".format(folder, episode))

                    torch.save(agent.actor.state_dict(),"../../../../models/{}/td3_ep_{}_actor.pth".format(folder, episode))
                    torch.save(agent.critic_q1.state_dict(),"../../../../models/{}/td3_ep_{}_critic_q1.pth".format(folder, episode))
                    torch.save(agent.critic_q2.state_dict(),"../../../../models/{}/td3_ep_{}_critic_q2.pth".format(folder, episode))
                elif episode % UPDATE_LATEST_PLOT_FREQ == 0:
                    #plt.figure(1)
                    #plt.plot(episode_number, average_reward)
                    #plt.title("TD3 %s episodes" % str(episode))
                    #plt.savefig("../../../../plots/{}/latest_td3.png".format(folder, episode))
                    pass
                    
                
                if episode % PATH_POS_FREQ == 0:

                    # write robot position path for each episode
                    file_object = open("../../../../plots/{}/data.txt".format(folder), "a")

                    file_object.write("Ep{} ".format(episode) + str(seg1Id) + " " + str(seg2Id) + " " + str(step) + " " + str(completion) + " " + str(average_ep_error) + "\n")

                    for i in range(30*len(trackSegmentList)):
                        file_object.write(str(i+1) + " " + str(markers_x[i]) + " " + str(markers_y[i]) + "\n")

                    for pos in range(len(robot_x)):
                        file_object.write("0 " + str(robot_x[pos]) + " " + str(robot_y[pos]) + "\n")

                    file_object.write("End Episode {}\n".format(episode))

                    file_object.close()

                break

    print(f"******* -----{episodes} episodes for training ended-----********* ")
    plt.close()

    

def run_testing(env,episodes_test, horizont, agent, style, track, folder=FOLDER, noise=False):

    # TODO: CREATE FOLDER FOR PLOTS AND MODELS AUTOMATICALLY/REMOVE "../../.."
    if not os.path.exists("../../../../plots/{}".format(folder)):
        os.makedirs("../../../../plots/{}".format(folder))
    
    if not os.path.exists("../../../../models/{}/best".format(folder)):
        os.makedirs("../../../../models/{}/best".format(folder))
     
    #Load actor
    agent.actor.load_state_dict(torch.load("../../../../models/{}/td3_ep_2000_actor.pth".format(folder)))
    agent.actor.eval()
    #Load Critic Q1
    agent.critic_q1.load_state_dict(torch.load("../../../../models/{}/td3_ep_2000_critic_q1.pth".format(folder)))
    agent.critic_q1.eval()
    #Load Critic Q2
    agent.critic_q2.load_state_dict(torch.load("../../../../models/{}/td3_ep_2000_critic_q2.pth".format(folder)))
    agent.critic_q2.eval()

    mode        = f"Testing {style}"
    rewards     = []


    for episode in range(1, episodes_test+1):

        trackSegmentList = track
        markers_x, markers_y = env.SetTrackSegmentList(trackSegmentList, noise)

        seg1Id = trackSegmentList[0]
        seg2Id = trackSegmentList[1]

        state = env.reset()
        state = np.array(state)
        episode_reward = 0
        sum_error = 0
        robot_x = []
        robot_y = []

        for step in range(1, horizont+1):

            action = agent.get_action_from_policy(state)
            next_state,reward,done,x, y, error, completion= env.step(action)
            next_state = np.array(next_state)
            episode_reward += reward
            state = next_state
            sum_error += error
            print(f"-------Episode:{episode} Step:{step} Action:{action[0]} Completion:{round(completion,1)}%---------")

            robot_x.append(x)
            robot_y.append(y)

            if done or step == horizont:
                average_ep_error =  sum_error/step
                WriteTestRewardsToFile(folder, episode_reward, average_ep_error, completion, seg1Id, seg2Id, step)

                file_object = open("../../../../plots/{}/test_odom.txt".format(folder), "a")

                file_object.write("Ep{} ".format(episode) + str(seg1Id) + " " + str(seg2Id) + " " + str(step) + " " + str(completion) + " " + str(average_ep_error)+ "\n")

                for i in range(30*len(trackSegmentList)):
                    file_object.write(str(i+1) + " " + str(markers_x[i]) + " " + str(markers_y[i]) + "\n")

                for pos in range(len(robot_x)):
                    file_object.write("0 " + str(robot_x[pos]) + " " + str(robot_y[pos]) + "\n")

                file_object.write("End Episode {}\n".format(episode))

                file_object.close()

                break

        print("Episode total reward:", episode_reward)


def WriteEpisodeRewardToFile(folder,reward, avg_error, completion, seg1, seg2, steps):
    reward_file = open("../../../../plots/{}/rewards.txt".format(folder), "a")
    reward_file.write("{} {} {} {} {} {}\n".format(reward, avg_error, completion, seg1, seg2, steps))
    reward_file.close()

def WriteTestRewardsToFile(folder,reward, avg_error, completion, seg1, seg2, steps):
    reward_file = open("../../../../plots/{}/test.txt".format(folder), "a")
    reward_file.write("{} {} {} {} {} {}\n".format(reward, avg_error, completion, seg1, seg2, steps))
    reward_file.close()

def main_run(style="MFRL"):
    env   = FSAE_Env()
    agent = TD3_Agent(env)

    num_exploration_episodes  = EXPLORATION_EPISODES
    num_episodes_training     = EPISODES
    episode_horizont          = STEPS

    run_exploration(env, num_exploration_episodes, episode_horizont, agent)
    run_training(env, num_episodes_training, episode_horizont, agent, style)
    #run_testing(env, episode_horizont, agent, style)

def train(num_exploration_episodes=EXPLORATION_EPISODES,num_training_episodes=EXPLORATION_EPISODES,steps=STEPS,folder=FOLDER,noise=False):
    env   = FSAE_Env()
    agent = TD3_Agent(env)
    style = "MFRL" # model free RL
    run_exploration(env, num_exploration_episodes, steps, agent, noise)
    run_training(env, num_training_episodes, steps, agent, style, folder, noise)

def test(episodes,steps,folder,noise,track):
    env   = FSAE_Env()
    agent = TD3_Agent(env)
    style = "MFRL" # model free RL
    run_testing(env, episodes, steps, agent, style, track, folder, noise)


if __name__ == "__main__":
    #model = "MBRL"
    model = "MFRL" # model free RL
    main_run(model)
    #test(100,STEPS)

