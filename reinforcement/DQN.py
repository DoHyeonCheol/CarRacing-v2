import gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

def make_env():
    env = gym.make('CarRacing-v2', continuous=False)
    return env

env = make_env()

log_path = os.path.join('./Train_DQN1/Logs')
model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
dqn_path = os.path.join('./Train_DQN1/Saved_Models/DQN_car_best_Model')

def make_eval_env():
    eval_env = gym.make('CarRacing-v2', continuous=False)
    return eval_env

eval_env = make_eval_env()
eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=dqn_path,
                             n_eval_episodes=5,
                             eval_freq=50000, verbose=1,
                             deterministic=True, render=False)
model.learn(total_timesteps=3000000, callback=eval_callback)
dqn_path = os.path.join('./Train_DQN1/Saved_Models/DQN_300k_Model_final')
model.save(dqn_path)
