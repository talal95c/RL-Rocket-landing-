import gymnasium  as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

model = PPO.load("model/ppo_lunar")
env= gym.make("LunarLander-v3")
moyenne,ecart_type= evaluate_policy(model,env,n_eval_episodes=10,deterministic=True)
print(f"moyenne={moyenne},ecart type={ecart_type}")
env.close()