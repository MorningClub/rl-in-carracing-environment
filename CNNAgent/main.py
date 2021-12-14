import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    N = 3000
    batch_size = 64
    n_epochs = 5
    alpha = 0.001
    action_space    = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
        (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
        (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ]
    agent = Agent(n_actions=len(action_space), batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=(96*96,))
    n_games = 1000

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []
    moving_score_average = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in tqdm(range(n_games)):
        observation = env.reset()
        observation = agent.view(observation)
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action_space[action])
            n_steps += 1
            score += reward
            agent.remember(observation.T, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = agent.view(observation_)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        moving_score_average.append(avg_score)


        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        if (i%50 == 0):
            agent.save_models(episode=str(i))

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

    with open("CNNAgent/episode_rewards_3000N.txt", "wb") as fp:
        pickle.dump(score_history, fp)
    with open("CNNAgent/moving_average_3000N.txt", "wb") as fp:
        pickle.dump(moving_score_average, fp)
    plt.plot(score_history, color="orange")
    plt.plot(moving_score_average, color="green")
    plt.ylabel('Score')
    plt.show()
