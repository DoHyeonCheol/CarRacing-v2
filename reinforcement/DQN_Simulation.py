import gym
from stable_baselines3 import DQN

env = gym.make('CarRacing-v2', continuous=False, render_mode='human')
model = DQN.load("./Training_DQN_2/Saved_Models/DQN_Model.zip")

obs, _ = env.reset()
episode_reward = 0

done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    if terminated or truncated or info.get('restart') or info.get('quit'):
        done = True

print(f'episode_reward: {episode_reward}')

env.close()
