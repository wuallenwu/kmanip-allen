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
# env = gym.make("InvertedPendulum-v4")

total_num_episodes = int(5e3)  # Total number of episodes
obs_space_dims, action_space_dims = 23, 11

plt.rcParams["figure.figsize"] = (10, 5)
torch.set_printoptions(precision=8) 
loss_fn = nn.CrossEntropyLoss()

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

        hidden_space1 = 32  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

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
        # print (action_means)
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
        # print(state)
        # print(np.dstack([state]))
        # # print(np.array([state]))
        # print(state)
        # print(np.concatenate(list(state.values())))
        flattened = np.concatenate(list(state.values()))
        flattened = flattened[:23]
        # print(flattened)
        state = torch.tensor(flattened, dtype=torch.float64)
        action_means, action_stddevs = self.net(state)
        # print(action_means)
        # print(action_stddevs)
        # breakpoint()

        # create a normal distribution from the predicted
        # mean and standard deviation and sample an action
        distribs = [Normal(action_means[i] + self.eps, action_stddevs[i] + self.eps) for i in range(len(action_means))]
        # print(distrib)
        # action = distrib.sample()
        actions = [(distrib.sample()) for distrib in distribs]
        log_probs = [distribs[i].log_prob(actions[i]) for i in range(len(distribs))]
        log_prob_sum = torch.sum(torch.stack(log_probs))
        # print(probs)
        # print(loss)
        # breakpoint()
        # # print(log_prob)
        # # breakpoint()
        # log_prob = 0
        # for i in range(len(distribs)):
        #     prob = distribs[i].log_prob(actions[i])
        #     log_prob += prob.item()
        # log_prob = torch.sum
        # print(actions)
        # breakpoint()
        actions = torch.tensor([(distrib.sample()).item() for distrib in distribs], dtype = torch.float64)
        # print(actions)
        # breakpoint()
        self.probs.append(log_prob_sum)
        # print(actions)
        actions = actions.numpy()
        # print(actions)

        keys = ["grip_r", "q_pos"]
        vals = [actions[:1], actions[1:]]

        actiondict = OrderedDict()
        for i in range (2):
            actiondict[keys[i]] = vals[i]
        # print(actiondict)
        # breakpoint()
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

        loss = 0
        # minimize -1 * prob * reward obtained
        # print(self.probs)
        # print(len(self.probs))
        # breakpoint()
        for log_prob, delta in zip(self.probs, deltas, strict = True):
            loss += log_prob.mean() * delta * (-1)
        # print(loss)
        # log_probs = torch.stack(self.probs)
        # loss = torch.mean(log_probs) * (sum(self.rewards))
        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("updating")
        # print(self.net)

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

rewards_over_seeds = []

for seed in [0]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        start_time = time.time()
        # print(episode)
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, _ = env.reset(seed=seed)
        # print(obs)
        # breakpoint()
        # print(env.observation_space)

        # print(obs)
        done = False
        rewards = []
        # counter = 0
        while not done:
            # print(counter)
            # counter += 1
            action = agent.sample_action(obs)
            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            # wrapped_env.step(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            # breakpoint()
            # print(action)
            # print(obs)
            # breakpoint()
            # print(reward)
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
        # print("Time Elapsed:", time.time() - start_time)


    #     reward_over_episodes.append(env.return_queue[-1])
    #     agent.update()

    #     if episode % 1 == 0:
    #         avg_reward = np.mean(wrapped_env.return_queue)
    #         print("Episode:", episode, "Average Reward:", avg_reward)

    # rewards_over_seeds.append(reward_over_episodes)

# plotting learning trajectory
# print(rewards)
print(len(reward_over_episodes))
average_reward = [np.mean(rewards[i:i+50]) for i in range(0,len(rewards),50)]

#display learning over episodes
# print(average_reward)
xs = [x for x in range(len(reward_over_episodes))]
plt.plot(xs, reward_over_episodes)
plt.show()

# rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
# df1 = pd.DataFrame(rewards_to_plot).melt()
# df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
# sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
# sns.lineplot(x="episodes", y="reward", data=df1).set(
#     title="REINFORCE for InvertedPendulum-v4"
# )
# plt.show()