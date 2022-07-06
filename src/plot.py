import matplotlib.pyplot as plt


if __name__ == '__main__':
    #specify path to .txt file that contains collected data
    file_object = open("/home/aakaash/fsae/src/fsae_control/data.txt", "r")
    lines = file_object.readlines()

    episode_count = 0
    episode = [] 

    for line in lines:
        #splits by episode
        if(line == "End Episode\n"):
            episode_count += 1
            all_rewards = []
            #gets reward value for each step in episode
            for line in episode:
                values = line.split(' ')
                #reward = float(values[len(values)-1])
                #if reward < 0:
                    #reward = -1
                #all_rewards.append(reward)
                lin = float(values[len(values)-3])
                ang = float(values[len(values)-2])

                if ang == 0.0:
                    states = [0.0, 0.0, 1.0, 0.0, 0.0]
                elif lin == 0.0 and ang < 0.0:
                    states = [1.0, 0.0, 0.0, 0.0, 0.0]
                elif lin == 0.0 and ang > 0.0:
                    states = [0.0, 0.0, 0.0, 0.0, 1.0]
                elif lin > 0.0 and ang < 0.0:
                    states = [0.0, 1.0, 0.0, 0.0, 0.0]
                elif lin > 0.0 and ang > 0.0:
                    states = [0.0, 0.0, 0.0, 1.0, 0.0]
                else:
                    states = [0.0, 0.0, 0.0, 0.0, 0.0]


                print(values[len(values)-3] + " " + values[len(values)-2] + " " + str(states))

            '''
            #plot each episode reward on seperate plot
            plt.figure(episode_count)
            plt.plot(all_rewards)
            plt.title("Episode " + str(episode_count))
            plt.xlabel('Line')
            plt.ylabel('Reward')
            '''
            episode.clear()
            continue
        else:
            episode.append(line)
      
    plt.show()

    

