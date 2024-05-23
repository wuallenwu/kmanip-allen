# WIP - REINFORCE for one arm touching block

from collections import OrderedDict
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
import gym_kmanip
# choose your environment
# ENV_NAME: str = "KManipSoloArm"
ENV_NAME: str = "KManipSoloArmQPos"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME)

total_num_episodes = int(5e4)  # Total number of episodes
obs_space_dims, action_space_dims = 23, 11 #23 observation space dims since we don't care about cube orientation

plt.rcParams["figure.figsize"] = (10, 5)
torch.set_printoptions(precision=8)

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  
        hidden_space2 = 16  

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        return action_means, action_stddevs

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6   # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        flattened = np.concatenate(list(state.values()))
        flattened = flattened[:23]
        state = torch.tensor(flattened, dtype=torch.float64)
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        # mean and standard deviation and sample an action
        distribs = [Normal(action_means[i] + self.eps, action_stddevs[i] + self.eps) for i in range(len(action_means))]
        actions = [(distrib.sample()) for distrib in distribs]
        log_probs = [distribs[i].log_prob(actions[i]) for i in range(len(distribs))]
        log_prob_sum = torch.sum(torch.stack(log_probs))
        actions = torch.tensor([action.item() for action in actions], dtype = torch.float64)
        self.probs.append(log_prob_sum)
        actions = actions.numpy()

        keys = ["grip_r", "q_pos"]
        vals = [actions[:1], actions[1:]]

        actiondict = OrderedDict()
        for i in range (2):
            actiondict[keys[i]] = vals[i]
        return actiondict

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)
        # print(deltas[0])

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas, strict = True):
            loss += log_prob * delta * (-1)
        # print(loss)
        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in self.net.named_parameters():
          
        # print('Grad Sum', torch.sum([torch.norm(param.grad) ])
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
        

        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

rewards_over_seeds = []

for seed in [0]:  
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        start_time = time.time()
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, _ = env.reset(seed=seed)
        # print(obs)
        # print(env.observation_space)
        # breakpoint()
        done = False
        rewards = []
        
        while not done:
            action = agent.sample_action(obs)
            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, _ = env.step(action)

            rewards.append(reward)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated 
        agent.update()
        reward_over_episodes.append(np.average(rewards))
        if episode % 100 == 0:
            print("Episode:", episode, "Average Reward:", np.average(rewards))

#display learning over episodes
xs = [x for x in range(len(reward_over_episodes))]
plt.plot(xs, reward_over_episodes)
plt.show()