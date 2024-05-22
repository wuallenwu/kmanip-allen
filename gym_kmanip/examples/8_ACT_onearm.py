#Action Chunking Transformer 
from collections import OrderedDict
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym

# choose your environment
ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME)
obs_dims = env.observation_space.shape[0]
act_dims = env.action_space.n
total_num_episodes = 10000

class Encoder:

class Decoder:


class Transformer:
  def forward(self):

  def encoder

#create and run policy
agent = REINFORCE(obs_dims, act_dims)
episode_rewards = []
for episode in range(total_num_episodes):
  obs, _ = env.reset(seed = 0)
  #run a single episode
  done = False
  score = 0
  while not done:
    action = agent.sample(obs)
    state, reward, done, _, _ = env.step(action)
    score += reward
    agent.rewards.append(reward)
    # done = x_out or angle_out or state[2]
    obs = state

  episode_rewards.append(score)
  # average_reward = []
  avg = 0
  if episode % 100 == 0:
    print(agent.alpha)
    if episode == 0: 
      # average_reward.append(score)
      print("Episode:", episode, "Average Reward:", score)
    else: 
      avg = np.mean(episode_rewards[episode - 100 : episode])
      # average_reward.append(avg)
      # print(average_reward)
      print("Episode:", episode, "Average Reward:", avg)
  agent.update()
  agent.alpha += episode * 1e-9

average_reward = [np.mean(episode_rewards[i:i+50]) for i in range(0,len(episode_rewards),50)]

#display learning over episodes
print(average_reward)
xs = [x for x in range(len(average_reward))]
plt.plot(xs, average_reward)
plt.show()


