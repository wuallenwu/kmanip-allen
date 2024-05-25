from typing import OrderedDict
import numpy as np
import pygame
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

SENSITIVITY = 1 #higher = bigger actions
SEED = 0 #consistent seed for testing

def arr_to_actdict(actions):
  actions = actions * SENSITIVITY
  keys = ["eer_pos", "eer_orn", "grip_r"]
  vals = [np.array(actions[:3:]), np.array(actions[3:6:]), np.array(actions[6])]

  actiondict = OrderedDict()
  for i in range (3):
      actiondict[keys[i]] = vals[i]
  return actiondict

play(gym.make(ENV_NAME, render_mode="rgb_array"), keys_to_action={  
                                               "q": arr_to_actdict([0.1, 0, 0, 0, 0, 0, 0]), #position
                                               "w": arr_to_actdict([-0.1, 0, 0, 0, 0, 0, 0]),
                                               "a": arr_to_actdict([0, 0.1, 0, 0, 0, 0, 0]),
                                               "s": arr_to_actdict([0, -0.1, 0, 0, 0, 0, 0]),
                                               "z": arr_to_actdict([0, 0, 0.1, 0, 0, 0, 0]),
                                               "x": arr_to_actdict([0, 0, -0.1, 0, 0, 0, 0]),
                                               "e": arr_to_actdict([0, 0, 0, 0.1, 0, 0, 0]), #orientation
                                               "r": arr_to_actdict([0, 0, 0, -0.1, 0, 0, 0]),
                                               "d": arr_to_actdict([0, 0, 0, 0, 0.1, 0, 0]),
                                               "f": arr_to_actdict([0, 0, 0, 0, -0.1, 0, 0]),
                                               "c": arr_to_actdict([0, 0, 0, 0, 0, 0.1, 0]),
                                               "v": arr_to_actdict([0, 0, 0, 0, 0, -0.1, 0]), 
                                               "g": arr_to_actdict([0, 0, 0, 0, 0, 0, 0.1]), #gripper
                                               "h": arr_to_actdict([0, 0, 0, 0, 0, 0, -0.1]),
                                              }, noop=np.array([0, 0, 0, 0, 0, 0, 0]), seed = SEED)