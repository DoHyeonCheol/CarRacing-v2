import gym
import numpy as np
import os
from stable_baselines3 import DQN

env = gym.make('CarRacing-v2', continuous=False, render_mode='human')
model = DQN.load("./Training_DQN_1/Saved_Models/DQN_Model.zip")

log_path = os.path.join('./Real/')
os.makedirs(log_path, exist_ok=True)

log_file_path = os.path.join(log_path, 'DQN.txt')

quit = False

max_episodes = 300

# 정규화를 위한 최소값과 최대값 설정
min_coord = -300
max_coord = 300

episode_count = 0
previous_velocity = 0.0

with open(log_file_path, 'w') as log_file:
    while episode_count < max_episodes:
        obs, _ = env.reset()
        episode_reward = 0
        total_reward = 0.0
        done = False
        steps = 0
        restart = False

        # 초기 좌표 설정
        initial_x, initial_y = env.car.hull.position

        # 좌표 기록을 위한 리스트 초기화
        coordinates = []

        while not done:
            # 관찰(observation)에서 자동차의 실제 평면 좌표 추출
            car = env.car
            x_coordinate, y_coordinate = car.hull.position

            # 초기 좌표 기준으로 상대 좌표 계산
            relative_x = x_coordinate - initial_x
            relative_y = y_coordinate - initial_y

            # 상대 좌표를 -1에서 1 사이의 값으로 정규화
            normalized_x = round((relative_x - min_coord) / (max_coord - min_coord) * 2 - 1, 5)
            normalized_y = round((relative_y - min_coord) / (max_coord - min_coord) * 2 - 1, 5)

            velocity = round(np.linalg.norm(car.hull.linearVelocity), 5)

            steering_angle = round(car.wheels[0].joint.angle, 5)

            acceleration = round(velocity - previous_velocity, 5)
            previous_velocity = velocity

            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            # 좌표를 리스트에 추가
            coordinates.append(f"{normalized_x}, {normalized_y}, {velocity}, {steering_angle}, {acceleration}")
            total_reward += reward
            steps += 1

            if terminated or truncated or info.get('restart') or info.get('quit'):
                done = True

        # total_reward가 800을 넘을 경우에만 로그에 기록하고 에피소드 카운트 증가
        if total_reward > 800:
            log_file.write('\n'.join(coordinates) + '\n')
            episode_count += 1

env.close()
