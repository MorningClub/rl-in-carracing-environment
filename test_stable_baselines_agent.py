import gym

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

# Parallel environments
env = make_vec_env("CarRacing-v0", n_envs=1)

model = PPO.load("PPO_1m_default_settings")
#model = A2C.load("A2C_3m_12env_1000N.zip")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()