import gym
import numpy as np
import os
import pygame

# Initialize Pygame
pygame.init()

env = gym.make('CarRacing-v2', render_mode="human")
env._max_episode_steps = 999
obs = env.reset()

log_path = os.path.join('./Real/')
os.makedirs(log_path, exist_ok=True)
log_file_path = os.path.join(log_path, 'keyboard.txt')

quit = False
max_episodes = 20
episode_count = 0

# 정규화를 위한 최소값과 최대값 설정
min_coord = -300
max_coord = 300

def register_input():
    global quit
    event = pygame.event.poll()
    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        quit = True

previous_velocity = 0.0

# Open log file
with open(log_file_path, 'a') as log_file:
    while episode_count < max_episodes and not quit:
        done = False
        total_reward = 0.0
        steps = 0

        initial_x, initial_y = env.car.hull.position
        coordinates = []
        previous_velocity = 0.0

        while not done:
            env.render()
            register_input()

            if quit:
                break

            action = np.array([0.0, 0.0, 0.0])
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = -1.0
            if keys[pygame.K_RIGHT]:
                action[0] = +1.0
            if keys[pygame.K_UP]:
                action[1] = +1.0
            if keys[pygame.K_DOWN]:
                action[2] = +0.8

            car = env.car
            x_coordinate, y_coordinate = car.hull.position

            relative_x = x_coordinate
            relative_y = y_coordinate

            normalized_x = round((relative_x - min_coord) / (max_coord - min_coord) * 2 - 1, 5)
            normalized_y = round((relative_y - min_coord) / (max_coord - min_coord) * 2 - 1, 5)

            velocity = round(np.linalg.norm(car.hull.linearVelocity), 5)
            steering_angle = round(car.wheels[0].joint.angle, 5)
            acceleration = round(velocity - previous_velocity, 5)
            previous_velocity = velocity

            obs, reward, terminated, truncated, info = env.step(action)
            coordinates.append(f"{normalized_x}, {normalized_y}, {velocity}, {steering_angle}, {acceleration}")
            total_reward += reward
            steps += 1

            if terminated or truncated:
                done = True

        if total_reward > 800:
            log_file.write('\n'.join(coordinates) + '\n')
            episode_count += 1

        obs = env.reset()

env.close()
pygame.quit()