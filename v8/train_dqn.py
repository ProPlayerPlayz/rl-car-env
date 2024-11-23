# train_dqn.py
import threading
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from agent import DQNAgent
import preprocess  # Import the preprocess_data function
import pygame

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
        if e % save_interval == 0:
            agent.save_model(f"model_{e}.h5")
            print(f"Model saved as model_{e}.h5")

    done_event.set()

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
    action_size = 9  # Number of actions (0 to 8)
    agent = DQNAgent(state_size, action_size)
    episodes = 500

    # Pretraining phase
    states, actions_one_hot = preprocess.preprocess_data('gameplay_data.csv')
    print("Starting pretraining...")
    agent.pretrain(states, actions_one_hot, epochs=10, batch_size=32)
    agent.save_model('pretrained_model.h5')
    print("Pretraining completed and model saved as 'pretrained_model.h5'.")

    # Threading setup
    done_event = threading.Event()
    rl_thread = threading.Thread(target=train_rl, args=(agent, env, episodes, done_event))

    # Start training in a separate thread
    rl_thread.start()

    # Start rendering in the main thread
    render_pygame(env, done_event)
