# environment.py
import pygame
import numpy as np
import math

class Environment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 500))
        pygame.display.set_caption("Reinforcement Learning: Car and Ball")
        self.clock = pygame.time.Clock()

        # Goal position fixed
        self.goal_width, self.goal_height = 100, 200
        self.goal_x, self.goal_y = 750, 250

        # Load images
        self.car_image = pygame.image.load("car2.png")
        self.ball_image = pygame.image.load("ball.png")
        self.goal_image = pygame.image.load("goal2.png")
        self.background_image = pygame.image.load("field2.png")

        # Scale images
        self.car_image = pygame.transform.scale(self.car_image, (50, 25))
        self.ball_image = pygame.transform.scale(self.ball_image, (30, 30))
        self.goal_image = pygame.transform.scale(self.goal_image, (self.goal_width, self.goal_height))
        self.background_image = pygame.transform.scale(self.background_image, (800, 500))

        # Metrics
        self.episode = 0
        self.car_to_ball_reward = 0
        self.ball_to_goal_reward = 0
        self.step_reward = 0
        self.total_reward = 0
        self.kicked = False
        self.goal = False

        # Calculate initial distances
        self.goal_left = self.goal_x - self.goal_width // 2
        self.goal_right = self.goal_x + self.goal_width // 2
        self.goal_top = self.goal_y - self.goal_height // 2
        self.goal_bottom = self.goal_y + self.goal_height // 2

        self.reset()

    def reset(self):
        self.episode += 1
        # First, set the ball's position
        # self.ball_x, self.ball_y = self.randomize_ball()
        self.ball_x, self.ball_y = self.fixed_ball()
        # Now, randomize the car's position, ensuring it's not within 40 units of the ball
        self.car_x, self.car_y, self.car_angle = self.randomize_car()
        self.car_velocity = 0
        self.car_angular_velocity = 0
        self.ball_velocity_x = 0
        self.ball_velocity_y = 0
        self.kicked = False
        self.goal = False

        self.prev_car_ball_dist = self.get_distance(self.car_x, self.car_y, self.ball_x, self.ball_y)
        self.prev_ball_goal_dist = self.get_distance(self.ball_x, self.ball_y, self.goal_x, self.goal_y)

        self.car_to_ball_reward = 0
        self.ball_to_goal_reward = 0
        self.step_reward = 0
        self.total_reward = 0

        return self.get_state()

    def step(self, action):
        if not self.kicked:
            self.move_car(action)

        self.move_ball()
        self.handle_collisions()

        car_to_ball_reward_step = self.reward_car_to_ball() if not self.kicked else 0
        ball_to_goal_reward_step = self.reward_ball_to_goal() if self.kicked else 0

        self.car_to_ball_reward += car_to_ball_reward_step
        self.ball_to_goal_reward += ball_to_goal_reward_step
        self.step_reward = car_to_ball_reward_step + ball_to_goal_reward_step

        if self.kicked:
            self.car_velocity = 0
            self.car_angular_velocity = 0

        done = self.kicked and (abs(self.ball_velocity_x) < 0.1 and abs(self.ball_velocity_y) < 0.1 or self.goal)

        if done:
            self.total_reward = self.car_to_ball_reward + self.ball_to_goal_reward

        return self.get_state(), self.step_reward, done

    def reward_car_to_ball(self):
        car_ball_dist = self.get_distance(self.car_x, self.car_y, self.ball_x, self.ball_y)
        reward = 0

        if car_ball_dist > self.prev_car_ball_dist:
            reward -= 5  # Penalty for moving away from the ball
        elif car_ball_dist == self.prev_car_ball_dist:
            reward -= 0.1
        elif car_ball_dist < self.prev_car_ball_dist:
            reward += car_ball_dist * 0.01  # Reward for getting closer to the ball
            if car_ball_dist > 150:
                reward += -0.1  # Penalty for being far from the ball
            elif 100 < car_ball_dist <= 150:
                reward += 1  # Reward for getting closer to the ball
            elif 50 < car_ball_dist <= 100:
                reward += 2  # Reward for getting closer to the ball
            elif 35 < car_ball_dist <= 50:
                reward += 4  # Reward for getting closer to the ball
            elif car_ball_dist <= 35:
                reward += 15  # Reward for Kicking Ball
                print(f"Kicked the ball! Car-Ball Reward Step: {reward}")
                self.kicked = True
                angle = math.atan2(self.ball_y - self.car_y, self.ball_x - self.car_x)
                self.ball_velocity_x = math.cos(angle) * 7
                self.ball_velocity_y = math.sin(angle) * 7

        self.prev_car_ball_dist = car_ball_dist
        return reward

    def reward_ball_to_goal(self):
        ball_goal_dist = self.get_distance_to_outline_of_goal(self.ball_x, self.ball_y)
        reward = 0

        if ball_goal_dist < self.prev_ball_goal_dist:
            reward += (self.prev_ball_goal_dist - ball_goal_dist) * 0.5
        else:
            reward -= 10

        if self.check_goal():
            reward += 200
            print(f"Goal scored! Ball-Goal Reward Step: {reward}")
            self.ball_velocity_x = 0
            self.ball_velocity_y = 0
            self.goal = True

        self.prev_ball_goal_dist = ball_goal_dist
        return reward

    def check_goal(self):
        return self.goal_left <= self.ball_x <= self.goal_right and self.goal_top <= self.ball_y <= self.goal_bottom

    def randomize_car(self):
        while True:
            x = np.random.randint(300, 500)
            y = np.random.randint(150, 350)
            angle = np.random.uniform(0, 360)
            dist = self.get_distance(x, y, self.ball_x, self.ball_y)
            if dist >= 40:
                break
        return x, y, angle

    def randomize_ball(self):
        x = np.random.randint(250, 550)
        y = np.random.randint(150, 350)
        return x, y

    def fixed_ball(self):
        x = 400
        y = 250
        return x, y

    def get_distance(self, x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def get_distance_to_outline_of_goal(self, x, y):
        calc_x = self.goal_left if x < self.goal_x else (x if x < self.goal_right else self.goal_right)  # Clamp x
        calc_y = self.goal_top if y < self.goal_y else (y if y < self.goal_bottom else self.goal_bottom) # Clamp y

        return self.get_distance(x, y, calc_x, calc_y)

    def get_state(self):
        car_ball_dist = self.get_distance(self.car_x, self.car_y, self.ball_x, self.ball_y)
        return np.array([
            self.car_x, self.car_y, self.car_velocity, self.car_angular_velocity, self.car_angle,
            self.ball_x, self.ball_y, car_ball_dist, 1 if self.ball_x < self.car_x else -1
        ])

    def handle_collisions(self):
        if self.ball_x - 15 < 0 or self.ball_x + 15 > 800:
            self.ball_velocity_x = -self.ball_velocity_x
        if self.ball_y - 15 < 0 or self.ball_y + 15 > 500:
            self.ball_velocity_y = -self.ball_velocity_y

    def move_car(self, action):
        car_speed = 5
        car_angular_speed = 5

        if action == 0:  # Forward
            self.car_velocity = car_speed
            self.car_angular_velocity = 0
        elif action == 1:  # Backward
            self.car_velocity = -car_speed
            self.car_angular_velocity = 0
        elif action == 2:  # Rotate Left
            self.car_velocity = 0
            self.car_angular_velocity = -car_angular_speed
        elif action == 3:  # Rotate Right
            self.car_velocity = 0
            self.car_angular_velocity = car_angular_speed
        elif action == 4:  # Forward + Left
            self.car_velocity = car_speed
            self.car_angular_velocity = -car_angular_speed
        elif action == 5:  # Forward + Right
            self.car_velocity = car_speed
            self.car_angular_velocity = car_angular_speed
        elif action == 6:  # Backward + Left
            self.car_velocity = -car_speed
            self.car_angular_velocity = -car_angular_speed
        elif action == 7:  # Backward + Right
            self.car_velocity = -car_speed
            self.car_angular_velocity = car_angular_speed
        elif action == 8:  # No Action
            self.car_velocity = 0
            self.car_angular_velocity = 0
        else:
            self.car_velocity = 0
            self.car_angular_velocity = 0

        self.car_angle += self.car_angular_velocity
        self.car_x += math.cos(math.radians(self.car_angle)) * self.car_velocity
        self.car_y += math.sin(math.radians(self.car_angle)) * self.car_velocity

    def move_ball(self):
        self.ball_x += self.ball_velocity_x
        self.ball_y += self.ball_velocity_y
        self.ball_velocity_x *= 0.98  # Friction
        self.ball_velocity_y *= 0.98  # Friction

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.blit(self.background_image, (0, 0))
        rotated_goal = pygame.transform.rotate(self.goal_image, 0)
        self.screen.blit(rotated_goal, (self.goal_x - 50, self.goal_y - 100))
        self.screen.blit(self.ball_image, (self.ball_x - 15, self.ball_y - 15))
        rotated_car = pygame.transform.rotate(self.car_image, -self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rect.topleft)

        font = pygame.font.SysFont(None, 24)
        font2 = pygame.font.SysFont(None, 24,bold=True)

        # Display metrics
        episode_label = font2.render(f"Episode: {self.episode}", True, (0, 0, 0))
        step_label = font.render(f"Step Reward: {self.step_reward:.2f}", True, (0, 0, 0))
        car_to_ball_label = font.render(f"Car-Ball Reward: {self.car_to_ball_reward:.2f}", True, (0, 0, 0))
        ball_to_goal_label = font.render(f"Ball-Goal Reward: {self.ball_to_goal_reward:.2f}", True, (0, 0, 0))
        kicked_label = font.render(f"Kicked: {int(self.kicked)}", True, (55, 25, 100))
        goal_label = font.render(f"Goal: {int(self.goal)}", True, (25, 100, 80))

        # Render labels
        self.screen.blit(episode_label, (10, 10))
        self.screen.blit(step_label, (10, 40))
        self.screen.blit(car_to_ball_label, (10, 70))
        self.screen.blit(ball_to_goal_label, (10, 100))
        self.screen.blit(kicked_label, (10, 130))
        self.screen.blit(goal_label, (10, 160))

        pygame.display.flip()
        self.clock.tick(60)
