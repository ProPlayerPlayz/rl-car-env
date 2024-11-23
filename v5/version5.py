import pygame
import numpy as np
import math
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import matplotlib.pyplot as plt
import threading

class Environment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 500))
        pygame.display.set_caption("Reinforcement Learning: Car and Ball")
        self.clock = pygame.time.Clock()

        # Goal position fixed
        self.goal_width, self.goal_height = 100,200
        self.goal_x, self.goal_y = 727, 218

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
        self.car_x, self.car_y, self.car_angle = self.randomize_car()
        #self.ball_x, self.ball_y = self.randomize_ball()
        self.ball_x, self.ball_y = self.fixed_ball()
        if self.get_distance(self.car_x, self.car_y, self.ball_x, self.ball_y) < 50:
            self.car_x += 50
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

        reward += car_ball_dist * 0.2

        if car_ball_dist < self.prev_car_ball_dist:
            reward += 0.5           # Reward for moving towards the ball
        else:
            reward -= 0.5           # Penalty for moving away from the ball

        if car_ball_dist < 60:
            reward += 10            # Reward for being close to the ball
        elif car_ball_dist < 35:
            reward += 20            # Reward for being very close to the ball
            self.kicked = True
            angle = math.atan2(self.ball_y - self.car_y, self.ball_x - self.car_x)
            self.ball_velocity_x = math.cos(angle) * 7
            self.ball_velocity_y = math.sin(angle) * 7
            #print(f"Ball kicked! Car-Ball Reward Step: {reward}")

        self.prev_car_ball_dist = car_ball_dist
        return reward

    def reward_ball_to_goal(self):
        #ball_goal_dist = self.get_distance(self.ball_x, self.ball_y, self.goal_x, self.goal_y)
        ball_goal_dist = self.get_distance_to_edge_of_goal(self.ball_x, self.ball_y)
        reward = 0

        if ball_goal_dist < self.prev_ball_goal_dist:
            reward += (self.prev_ball_goal_dist - ball_goal_dist) * 0.1
        else:
            reward -= 10

        if self.check_goal():
            reward += 100
            print(f"Goal scored! Ball-Goal Reward Step: {reward}")
            self.ball_velocity_x = 0
            self.ball_velocity_y = 0
            self.goal = True

        self.prev_ball_goal_dist = ball_goal_dist
        return reward

    def check_goal(self):
        return self.goal_left <= self.ball_x <= self.goal_right and self.goal_top <= self.ball_y <= self.goal_bottom

    def randomize_car(self):
        x = np.random.randint(300, 500)
        y = np.random.randint(150, 350)
        angle = np.random.uniform(0, 360)
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
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def get_distance_to_edge_of_goal(self, x, y):
        if x < self.goal_left:
            return self.get_distance(x, y, self.goal_left, y)
        elif x > self.goal_right:
            return self.get_distance(x, y, self.goal_right, y)
        elif y < self.goal_top:
            return self.get_distance(x, y, x, self.goal_top)
        elif y > self.goal_bottom:
            return self.get_distance(x, y, x, self.goal_bottom)
        else:
            return 0


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
        elif action == 1:  # Backward
            self.car_velocity = -car_speed
        elif action == 2:  # Rotate Left
            self.car_angular_velocity = -car_angular_speed
        elif action == 3:  # Rotate Right
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
        else:  # No action
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
        self.screen.blit(rotated_goal, (self.goal_x - 50, self.goal_y - 75))
        self.screen.blit(self.ball_image, (self.ball_x - 15, self.ball_y - 15))
        rotated_car = pygame.transform.rotate(self.car_image, -self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rect.topleft)

        font = pygame.font.SysFont(None, 24)

        # Display metrics
        episode_label = font.render(f"Episode: {self.episode}", True, (0, 0, 0))
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
        self.clock.tick(240)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            tf.keras.Input(shape=(self.state_size,)),  # Define the input shape
            Dense(24, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, (1, self.state_size))  # Ensure correct shape
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, (1, self.state_size))  # Ensure correct shape
            next_state = np.reshape(next_state, (1, self.state_size))  # Ensure correct shape
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_rl(agent, env, episodes, done_event, render_interval=50):
    rewards = []  # Total reward per episode
    car_to_ball_rewards = []  # Car-to-ball rewards per episode
    ball_to_goal_rewards = []  # Ball-to-goal rewards per episode
    kicks = []  # Whether the ball was kicked in the episode
    goals = []  # Whether the ball reached the goal in the episode

    for e in range(episodes):
        env.episode = e + 1
        state = env.reset()
        state = np.reshape(state, (1, agent.state_size))  # Ensure proper shape
        total_reward = 0
        kicked = 0  # Flag for ball kicked
        goal = 0  # Flag for goal achieved

        for _ in range(700):  # Increased episode steps
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, (1, agent.state_size))  # Ensure proper shape

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Track if the ball was kicked
            if not kicked and env.kicked:
                kicked = 1

            # Track if a goal was scored
            if env.check_goal():
                goal = 1

            if done:
                break

            if e % render_interval == 0:
                env.render()

        # Append metrics
        rewards.append(total_reward)
        car_to_ball_rewards.append(env.car_to_ball_reward)
        ball_to_goal_rewards.append(env.ball_to_goal_reward)
        kicks.append(kicked)
        goals.append(goal)

        # Console log for each episode
        print(
            f"Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}, "
            f"Car-Ball-Reward: {env.car_to_ball_reward:.2f}, "
            f"Ball-Goal-Reward: {env.ball_to_goal_reward:.2f}, "
            f"Kicked: {kicked}, Goal: {goal}"
        )

        # Replay and train
        agent.replay(32)

        # Save the model every 100 episodes
        save_interval = 100
        if e % save_interval == 0 and e > 0:
            agent.model.save(f"model_{e}.h5")
            print(f"Model saved as model_{e}.h5")

    done_event.set()

    # Final Model Save
    agent.model.save("model_final.h5")

    # Plot Training Metrics
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 2, 1)
    plt.plot(rewards, label="Total Rewards")
    plt.title("Total Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(car_to_ball_rewards, label="Car-to-Ball Rewards")
    plt.title("Car-to-Ball Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(ball_to_goal_rewards, label="Ball-to-Goal Rewards")
    plt.title("Ball-to-Goal Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(kicks, label="Kicks")
    plt.title("Kicks per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Kicked (1 = Yes, 0 = No)")
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(goals, label="Goals")
    plt.title("Goals per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Goal (1 = Yes, 0 = No)")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Rendering function
def render_pygame(env, done_event):
    """
    Renders the Pygame environment.
    Keeps rendering until the `done_event` is set after training is complete.
    """
    while not done_event.is_set():
        env.render()

    pygame.quit()

if __name__ == "__main__":
    # Initialize environment and agent
    env = Environment()
    state_size = env.get_state().shape[0]
    action_size = 8  # Number of actions
    agent = DQNAgent(state_size, action_size)
    episodes = 100

    # Threading setup
    done_event = threading.Event()
    rl_thread = threading.Thread(target=train_rl, args=(agent, env, episodes, done_event))
    
    # Start training in a separate thread
    rl_thread.start()

    # Start rendering in the main thread
    render_pygame(env, done_event)