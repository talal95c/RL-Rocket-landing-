import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO            # PPO vient de SB3, pas de gymnasium

model = PPO.load("model/ppo_lunar")
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env,
                  video_folder="videos",
                  name_prefix="ppo_landing",
                  episode_trigger=lambda e: True)   # filme chaque épisode

try:
    for episode in range(3):                 # 3 épisodes : 0, 1, 2
        observation, info = env.reset(seed=episode)
        score, frames, fini = 0, 0, False
        while not fini:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            frames += 1
            fini = terminated or truncated
        print(f"Episode={episode}, score={score:.1f}, frames={frames}")
finally:
    env.close()                              # écrit la dernière vidéo