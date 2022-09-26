import DQN as dqn
import TD3 as td3

EXPLORATION_EPISODES = 100
TRAIN_EPISODES = 5000
TEST_EPISODES = 100
STEPS = 2000
# Track sequence provided as an array of track segment Ids (see TrackGenetor file for more information)
TEST_TRACK = [0, 6, -6, 5, -2]

def main():
    # Example of how to call train functions for both DQN and TD3 models
    dqn.train(episodes=TRAIN_EPISODES, steps=STEPS, folder="DQN_Test", noise=False)
    td3.train(exploration_episodes=EXPLORATION_EPISODES, episodes=TRAIN_EPISODES, steps=STEPS, folder="TD3_Test", noise=False)

    #Example of how to call test functions for both DQN and TD3 models
    dqn.test(episodes=TEST_EPISODES, steps=STEPS, track=TEST_TRACK, folder="DQN_Test", noise=False)
    td3.test(episodes=TEST_EPISODES, steps=STEPS, track=TEST_TRACK, folder="TD3_Test", noise=False)

if __name__ == "__main__":
    main()