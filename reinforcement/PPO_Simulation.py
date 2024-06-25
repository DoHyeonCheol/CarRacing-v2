import gym
from stable_baselines3 import PPO
import numpy as np

env = gym.make('CarRacing-v2', render_mode="human")

model = PPO.load("./Training_test/Saved_Models/PPO_car_best_Model/best_model.zip")

obs, _ = env.reset()
episode_reward = 0

done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    # 종료 조건 확인
    if terminated or truncated or info.get('restart') or info.get('quit'):
        done = True

print(f'episode_reward: {np.array(episode_reward).mean()}')

env.close()
