import gym
import numpy as np
import pygame

# Initialize Pygame
pygame.init()

# Create CarRacing environment
env = gym.make('CarRacing-v2', render_mode="human")

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
        env.render()
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

env.close()
pygame.quit()