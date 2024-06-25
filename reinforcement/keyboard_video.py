import gym
import numpy as np
import pygame
from gym.wrappers import RecordVideo

# Initialize Pygame
pygame.init()

# Create CarRacing environment
env = gym.make('CarRacing-v2', render_mode="rgb_array")
env = RecordVideo(env, "./CarRacing_Video/videos_ppo/", episode_trigger=lambda x: True)  # Wrap the environment with RecordVideo


# Global variable for controlling the environment
quit = False

# Function to handle keyboard input
def register_input():
    global quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            quit = True

while True:
    obs, _ = env.reset()
    done = False

    while not done and not quit:
        env.render()  # This will also record the video
        register_input()

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

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated or info.get('restart')

    if quit:
        break

env.close()  # This will also close the video recording
pygame.quit()
