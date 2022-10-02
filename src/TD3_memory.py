"""
Authors:  Aakaash Salvaji, Harry Taylor, David Valencia, Trevor Gee, Henry Williams
The University of Auckland

TD3 Memory
"""

from collections import deque
import random


class MemoryClass:

    def __init__(self, replay_max_size_env=10_000, replay_max_size_model=10_000):

        self.replay_max_size_env   = replay_max_size_env
        self.replay_max_size_model = replay_max_size_model

        self.replay_buffer_env   = deque(maxlen=replay_max_size_env)
        self.replay_buffer_model = deque(maxlen=replay_max_size_model)


    def replay_buffer_environment_add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.replay_buffer_env.append(experience)

    def replay_buffer_model_add(self, state, action, reward, next_state, done):
        experience_from_model = (state, action, reward, next_state, done)
        self.replay_buffer_model.append(experience_from_model)

    def sample_experience_from_env(self, batch_size):
        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []

        batch = random.sample(self.replay_buffer_env, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def sample_experience_from_model(self, batch_size):
        state_batch  = []
        action_batch = []
        reward_batch = []
        done_batch   = []
        next_state_batch = []

        batch_of_experiences = random.sample(self.replay_buffer_model, batch_size)

        for experience in batch_of_experiences:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def len_env_buffer(self):
        return len(self.replay_buffer_env)

    def len_model_buffer(self):
        return len(self.replay_buffer_model)
