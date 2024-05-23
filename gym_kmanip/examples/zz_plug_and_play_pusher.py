from typing import OrderedDict
import numpy as np
import pygame
from dm_control import viewer
import gymnasium as gym
from gymnasium.utils.play import play
import gym_kmanip

# choose your environment
ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmQPos"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
# ENV_NAME: str = 'CartPole-v1'

def arr_to_actdict(actions):
  keys = ["eer_pos", "eer_orn", "grip_r"]
  vals = [np.array(actions[:3:]), np.array(actions[3:7:]), np.array(actions[7])]

  actiondict = OrderedDict()
  for i in range (3):
      actiondict[keys[i]] = vals[i]
  return actiondict

play(gym.make("CarRacing-v2", render_mode="rgb_array"), keys_to_action={  
                                               "w": np.array([0, 0.7, 0]),
                                               "a": np.array([-1, 0, 0]),
                                               "s": np.array([0, 0, 1]),
                                               "d": np.array([1, 0, 0]),
                                               "wa": np.array([-1, 0.7, 0]),
                                               "dw": np.array([1, 0.7, 0]),
                                               "ds": np.array([1, 0, 1]),
                                               "as": np.array([-1, 0, 1]),
                                              }, noop=np.array([0,0,0]))