import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    old_observation = env.reset()
    print(old_observation.shape)
    observation = []
    cnt = 0
    for i in range(len(old_observation[0])):
        current_row = []
        for j in range(len(old_observation[0])):
            current_row.append(int(sum(old_observation[i][j])/3))
            cnt +=1
        observation.append(current_row)

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    action_space    = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
        (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
        (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ]

    agent = Agent(n_actions=len(action_space), batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    agent.load_models()

    observation = env.reset()
    observation = agent.view(observation)
    done = False
    score = 0
    while not done:
        env.render()
        action, prob, val = agent.choose_action(observation)
        observation, reward, done, info = env.step(action_space[action])
        observation = agent.view(observation)

    env.close()
