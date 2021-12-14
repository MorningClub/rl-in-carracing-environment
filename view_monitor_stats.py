import csv
import matplotlib.pyplot as plt

episode_rewards = []
moving_averages = []
score_holder = []

with open('monitor_file_A2C_3m_12envs_1000N.monitor.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0 or line_count == 1:
            line_count += 1
        elif len(row) == 0:
            line_count += 1
        else:
            episode_rewards.append(float(row[0]))
            score_holder.append(float(row[0]))
            if len(score_holder) > 100:
                del score_holder[0]
            moving_averages.append(sum(score_holder)/len(score_holder))
            

print(max(moving_averages))
print(moving_averages.index(max(moving_averages)))

plt.plot(episode_rewards, color="orange", label="Episode Rewards")
plt.plot(moving_averages, color="green", label="Moving Average")
plt.ylabel('Reward')
plt.xlabel("Episode")
plt.show()