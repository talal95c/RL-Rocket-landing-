import gymnasium as gym
import torch as th
import stable_baselines3 as sb3

print("Gymnasium version:", gym.__version__)
print("PyTorch version:", th.__version__)
print("Stable-Baselines3 version:", sb3.__version__)

env = gym.make("LunarLander-v3")
observation, info = env.reset(seed=42)

print("Observation :", observation)
print("Forme de l'observation :", observation.shape)
print("Espace d'observation :", env.observation_space)
print("Espace d'actions :", env.action_space)

env.close()