import pickle
import matplotlib.pyplot as plt

with open("CNNAgent/episode_rewards_3000N.txt", "rb") as fp:
        episode_rewards = pickle.load(fp)
        
with open("CNNAgent/moving_average_3000N.txt", "rb") as fp:
        moving_average = pickle.load(fp)
        
        
print("max reward: ", max(episode_rewards))
print("max moving average: ", max(moving_average))
print("max moving average index: ", moving_average.index(min(moving_average)))
print(episode_rewards[367])


for i in range(len(episode_rewards)):
        if episode_rewards[i] > 70:
                print("start good performance: ", i)
                break

plt.plot(episode_rewards, color="orange", label="Episode Rewards")
plt.plot(moving_average, color="green", label="Moving Average")
plt.ylabel('Reward')
plt.xlabel("Episode")
plt.show()