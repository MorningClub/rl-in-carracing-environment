import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

#This file is used for training the A2C implementation
#from Stable Baselines3's

# Parallel environments
env = make_vec_env("CarRacing-v0", n_envs=12)
monitored_env = VecMonitor(env, filename="monitor_file_A2C_3m")

model = A2C("CnnPolicy", monitored_env, verbose=1, n_steps=100)
model.learn(total_timesteps=3000000)
model.save("A2C_3m")

#Render test of model in the environment
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
