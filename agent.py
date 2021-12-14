import os
import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.modules import linear
import torch.optim as optim
from torch.distributions.categorical import Categorical
from actor import ActorNetwork
from critic import CriticNetwork
from rpbuffer import PPOMemory
from autoencoder_fra_vm import AE


import matplotlib.pyplot as plt


class Agent:
    def __init__(
            self,
            n_actions, 
            input_dims, 
            gamma=0.99, 
            alpha=0.001, 
            gae_lambda=0.95, 
            policy_clip=0.1, 
            batch_size=64, 
            N=2048, 
            n_epochs=10,
        ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.autoencoder = AE()
        self.autoencoder.load_state_dict(T.load("autoencoder_model_128_01"))
        self.autoencoder.eval()

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    
    def save_models(self, episode=""):
        print('...saving models...')
        self.actor.save_checkpoint(episode)
        self.critic.save_checkpoint(episode)

    def load_models(self):
        print('...loading models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def view(self, observation):
        observation = observation[:84]
        observation = np.array(observation)
        observation = observation.astype("float64")
        observation = observation/255
        state = T.tensor(np.array([observation.T]), dtype=T.float).to(self.actor.device)
        with T.no_grad():
            return self.autoencoder.encoder(state)[0][0]

    def choose_action(self, observation):
        state = observation.to(self.actor.device)

        value = self.critic(state)
        dist = self.actor(state)
        action = dist.sample()
        
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                        (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()
