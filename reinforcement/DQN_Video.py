import gym
from stable_baselines3 import DQN
import numpy as np
from gym.wrappers import RecordVideo

env = gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array")
env = RecordVideo(env, "./CarRacing_Video/videos_DQN/", episode_trigger=lambda x: True) # 비디오 녹화를 위한 래퍼 추가

model = DQN.load("./Training_DQN/Saved_Models/DQN_Model.zip")

obs, _ = env.reset()
episode_reward = 0

done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    if terminated or truncated or info.get('restart') or info.get('quit'):
        done = True

print(f'episode_reward: {np.array(episode_reward).mean()}')

env.close()
