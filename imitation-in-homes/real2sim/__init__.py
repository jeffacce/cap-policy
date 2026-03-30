import gymnasium as gym
from .tasks.pick import SimPick
from .evaluate import eval_sim

gym.register(id="RUM-Pick-v0", entry_point=SimPick)
