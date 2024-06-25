import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make('CarRacing-v2')

log_path = os.path.join('./Train_PPO1/Logs')
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
ppo_path = os.path.join('./Train_PPO1/Saved_Models/PPO_car_best_Model')
eval_env = model.get_env()
eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=ppo_path,
                             n_eval_episodes=5,
                             eval_freq=50000, verbose=1,
                             deterministic=True, render=False)
model.learn(total_timesteps=3000000, callback=eval_callback)
ppo_path = os.path.join('./Train_PPO1/Saved_Models/PPO_300k_Model_final')
model.save(ppo_path)
