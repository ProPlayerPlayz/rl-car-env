# preprocess.py
import numpy as np
import csv

def preprocess_data(filename):
    """
    Preprocesses the recorded gameplay data.
    Reads the data from the CSV file and prepares states and actions for training.
    """
    states = []
    actions = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        for row in reader:
            state = [float(x) for x in row[:9]]
            action = int(row[9])

            states.append(state)
            actions.append(action)

    states = np.array(states)
    actions = np.array(actions)

    # Convert actions to one-hot encoding
    num_actions = 9  # Actions from 0 to 8
    actions_one_hot = np.zeros((len(actions), num_actions))
    actions_one_hot[np.arange(len(actions)), actions] = 1

    return states, actions_one_hot

if __name__ == "__main__":
    # Example usage
    states, actions_one_hot = preprocess_data('gameplay_data.csv')
    print("States shape:", states.shape)
    print("Actions shape:", actions_one_hot.shape)
