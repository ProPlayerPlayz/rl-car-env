# agent.py
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            tf.keras.Input(shape=(self.state_size,)),  # Define the input shape
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
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
            target = self.model.predict(state, verbose=0)[0]
            if done:
                target[action] = reward
            else:
                t = self.model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            target = np.reshape(target, (1, self.action_size))
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def pretrain(self, states, actions_one_hot, epochs=10, batch_size=32):
        """
        Pretrain the model using supervised learning on the collected human gameplay data.
        """
        # Shuffle the data
        indices = np.arange(states.shape[0])
        np.random.shuffle(indices)
        states = states[indices]
        actions_one_hot = actions_one_hot[indices]

        self.model.fit(states, actions_one_hot, epochs=epochs, batch_size=batch_size, verbose=1)

    def save_model(self, name):
        """
        Saves the model to a file.
        """
        self.model.save(name)

    def load_model(self, name):
        """
        Loads the model from a file.
        """
        self.model = tf.keras.models.load_model(name)
