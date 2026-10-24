import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
liste_score = []

try:
    for episode in range(5):                          # niveau 1 : dans le try
        observation, infos = env.reset(seed=episode)  # niveau 2 : dans le for
        score = 0
        frames = 0
        end = False

        while not end:                                # niveau 2
            action = env.action_space.sample()        # niveau 3 : dans le while
            observation, reward, terminated, truncated, infos = env.step(action)
            score += reward
            frames += 1
            if episode == 0 and frames <= 5:
                print(f"frame={frames} action={action} reward={reward:.2f} obs={np.round(observation, 2)}")
            end = terminated or truncated

        print(f"Episode {episode} : score={score:.1f}, frames={frames}")  # niveau 2 : fin de chaque épisode
        liste_score.append(score)                                          # niveau 2 : à chaque épisode

    print(f"Score moyen : {np.mean(liste_score):.1f}")  # niveau 1 : une fois, après les 5 épisodes
finally:
    env.close()