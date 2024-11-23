from collections import deque
import pygame
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

pygame.init()
print("Pygame initialized.")

class Car:
    def __init__(self):
        self.x = random.uniform(100, 700)
        self.y = random.uniform(100, 700)
        self.angle = random.uniform(0, 2 * math.pi)
        self.vel = 0
        self.ang_vel = 0
        self.width = 40
        self.height = 20
        self.speed = 5
        self.angular_speed = 0.1

    def update(self, action):
        # Update car's position and angle based on action
        if action == 0:  # Forward
            self.vel = self.speed
        elif action == 1:  # Backward
            self.vel = -self.speed
        elif action == 2:  # Rotate Left
            self.ang_vel = -self.angular_speed
        elif action == 3:  # Rotate Right
            self.ang_vel = self.angular_speed
        elif action == 4:  # Forward + Left
            self.vel = self.speed
            self.ang_vel = -self.angular_speed
        elif action == 5:  # Forward + Right
            self.vel = self.speed
            self.ang_vel = self.angular_speed
        elif action == 6:  # Backward + Left
            self.vel = -self.speed
            self.ang_vel = -self.angular_speed
        elif action == 7:  # Backward + Right
            self.vel = -self.speed
            self.ang_vel = self.angular_speed
        else:
            self.vel = 0
            self.ang_vel = 0

        # Update position and angle
        self.angle += self.ang_vel
        self.x += self.vel * math.cos(self.angle)
        self.y += self.vel * math.sin(self.angle)

        # Keep within bounds
        self.x = max(0, min(self.x, 800))
        self.y = max(0, min(self.y, 800))

    def draw(self, display):
        # Create the base surface for the car
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        car_surface.fill((0, 0, 255))  # Blue color

        # Rotate the car surface
        rotated_image = pygame.transform.rotate(car_surface, -math.degrees(self.angle))

        # Correctly position the rotated image
        rotated_rect = rotated_image.get_rect(center=(self.x, self.y))

        # Draw the rotated car on the display
        display.blit(rotated_image, rotated_rect.topleft)


class Ball:
    def __init__(self):
        self.x = random.uniform(100, 700)
        self.y = random.uniform(100, 700)
        self.radius = 15
        self.vel_x = 0
        self.vel_y = 0

    def update(self):
        # Update ball's position
        self.x += self.vel_x
        self.y += self.vel_y

        # Dampen the velocity
        self.vel_x *= 0.98
        self.vel_y *= 0.98

        # Keep within bounds and bounce off walls
        if self.x <= self.radius or self.x >= 800 - self.radius:
            self.vel_x *= -1
        if self.y <= self.radius or self.y >= 800 - self.radius:
            self.vel_y *= -1

    def draw(self, display):
        # Draw the ball
        pygame.draw.circle(display, (255, 0, 0), (int(self.x), int(self.y)), self.radius)

class Goal:
    def __init__(self):
        self.x = random.uniform(100, 700)
        self.y = random.uniform(100, 700)
        self.radius = 30  # Double the radius of the ball

    def draw(self, display):
        # Draw the goal region
        pygame.draw.circle(display, (0, 255, 0), (int(self.x), int(self.y)), self.radius, 2)



class CarBallEnv:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.width = 800
        self.height = 800
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Car Ball Environment")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Reset environment state
        self.car = Car()
        self.ball = Ball()
        self.goal = Goal()
        self.done = False
        self.timestep = 0
        self.total_timesteps = 500  # Max timesteps per episode
        return self.get_state()

    def get_state(self):
        # Get the current state
        state = np.array([
            self.car.x,
            self.car.y,
            self.car.vel,
            self.car.ang_vel,
            self.car.angle,
            self.ball.x,
            self.ball.y,
            self.ball.radius,
            self.goal.x,
            self.goal.y,
            self.get_distance(self.car.x, self.car.y, self.ball.x, self.ball.y),
            self.get_distance(self.ball.x, self.ball.y, self.goal.x, self.goal.y) - self.goal.radius,
            self.relative_position()
        ], dtype=np.float32)
        return state

    def step(self, action):
        # Perform action
        self.car.update(action)
        self.ball.update()
        self.check_collision()
        self.timestep += 1
        reward = self.compute_reward()
        self.done = self.is_done()
        state = self.get_state()
        return state, reward, self.done, {}

    def compute_reward(self):
        ball_to_goal_distance = self.get_distance(self.ball.x, self.ball.y, self.goal.x, self.goal.y)
        car_to_ball_distance = self.get_distance(self.car.x, self.car.y, self.ball.x, self.ball.y)
        if self.ball_in_goal():
            return 100  # High reward for completing the goal
        else:
            # Reward for moving the ball closer to the goal
            return -0.01 * ball_to_goal_distance - 0.1 * car_to_ball_distance


    def is_done(self):
        if self.timestep >= self.total_timesteps:
            return True
        if self.ball_in_goal():
            return True
        # End if ball or car hasn't moved significantly
        if abs(self.ball.vel_x) < 0.01 and abs(self.ball.vel_y) < 0.01:
            return True
        return False


    def render(self):
        self.display.fill((200, 200, 200))  # Light gray background
        self.goal.draw(self.display)
        self.ball.draw(self.display)
        self.car.draw(self.display)
        pygame.display.flip()  # Update the entire display
        self.clock.tick(60)    # Control the frame rate


    def close(self):
        pygame.quit()

    def get_distance(self, x1, y1, x2, y2):
        # Euclidean distance
        return math.hypot(x2 - x1, y2 - y1)

    def relative_position(self):
        # Determine if the ball is to the left or right of the car's front
        dx = self.ball.x - self.car.x
        dy = self.ball.y - self.car.y
        angle_to_ball = math.atan2(dy, dx)
        angle_diff = angle_to_ball - self.car.angle
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize between -pi and pi
        return angle_diff

    def check_collision(self):
        # Check for collision between car and ball
        if self.get_distance(self.car.x, self.car.y, self.ball.x, self.ball.y) <= self.ball.radius + self.car.width / 2:
            # Simple collision response: transfer velocity from car to ball
            self.ball.vel_x += self.car.vel * math.cos(self.car.angle)
            self.ball.vel_y += self.car.vel * math.sin(self.car.angle)

    def ball_in_goal(self):
        # Check if the ball is in the goal region
        return self.get_distance(self.ball.x, self.ball.y, self.goal.x, self.goal.y) <= self.goal.radius


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._build_model()

    def _build_model(self):
        # Neural network model
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        # Train the model using experiences from memory
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f = target_f.clone().detach()
            target_f[action] = target
            self.optimizer.zero_grad()
            outputs = self.model(state)
            loss = F.mse_loss(outputs, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes, min_frame_duration_ms=33):
    env = CarBallEnv()
    state_size = 13  # Number of state parameters
    action_size = 8  # Number of possible actions
    agent = DQNAgent(state_size, action_size)
    rewards = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_start_time = pygame.time.get_ticks()  # Track the start time of the episode
        
        for time_t in range(env.total_timesteps):
            frame_start_time = pygame.time.get_ticks()  # Start of the frame

            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Replay agent memory
            agent.replay()

            # Check if the episode is done
            if done:
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward}")
                break

            # Ensure minimum frame duration
            elapsed_time = pygame.time.get_ticks() - frame_start_time
            if elapsed_time < min_frame_duration_ms:
                pygame.time.wait(min_frame_duration_ms - elapsed_time)

        # Track rewards for plotting or logging
        rewards.append(total_reward)

    env.close()
    return rewards


import matplotlib.pyplot as plt

if __name__ == "__main__":
    episodes = 1000
    rewards = train_agent(episodes)
    plt.plot(range(episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.show()
