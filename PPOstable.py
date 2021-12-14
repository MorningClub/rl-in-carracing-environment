import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

#This file is used for training the PPO implementation
#from Stable Baselines3's

# Parallel environments
env = make_vec_env("CarRacing-v0", n_envs=1)
monitored_env = VecMonitor(env, filename="monitor_file_PPO_1m")

model = PPO("CnnPolicy", monitored_env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("PPO_model_1m")

#Render test of model in the environment
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
