# human_play.py
import pygame
import csv
import numpy as np
from environment import Environment  # Import the modified Environment class

def main():
    """
    Human-controlled gameplay program.
    Allows human to play the game using W, A, S, D keys and records gameplay data.
    Finishes after 50 episodes.
    """
    env = Environment()
    clock = pygame.time.Clock()
    running = True
    max_episodes = 50  # Set the maximum number of episodes
    current_episode = 1  # Initialize the current episode counter

    # Open a CSV file to record data
    with open('gameplay_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        header = ['car_x', 'car_y', 'car_velocity', 'car_angular_velocity', 'car_angle',
                  'ball_x', 'ball_y', 'car_ball_dist', 'ball_relative_position',
                  'action', 'reward']
        writer.writerow(header)

        while running:
            action = 8  # Default action is 'No Action'
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get keys
            keys = pygame.key.get_pressed()

            # Map keys to actions
            if keys[pygame.K_w] and keys[pygame.K_a]:
                action = 4  # Forward + Left
            elif keys[pygame.K_w] and keys[pygame.K_d]:
                action = 5  # Forward + Right
            elif keys[pygame.K_s] and keys[pygame.K_a]:
                action = 6  # Backward + Left
            elif keys[pygame.K_s] and keys[pygame.K_d]:
                action = 7  # Backward + Right
            elif keys[pygame.K_w]:
                action = 0  # Forward
            elif keys[pygame.K_s]:
                action = 1  # Backward
            elif keys[pygame.K_a]:
                action = 2  # Rotate Left
            elif keys[pygame.K_d]:
                action = 3  # Rotate Right
            else:
                action = 8  # No Action

            state = env.get_state()
            next_state, reward, done = env.step(action)
            env.render()

            # Record data
            writer.writerow(state.tolist() + [action, reward])

            clock.tick(60)

            if done:
                # Check if the maximum number of episodes has been reached
                if current_episode >= max_episodes:
                    print(f"Finished {max_episodes} episodes. Exiting the game.")
                    running = False
                else:
                    current_episode += 1  # Increment the episode counter
                    env.reset()

            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                running = False
                

    pygame.quit()

if __name__ == "__main__":
    main()
