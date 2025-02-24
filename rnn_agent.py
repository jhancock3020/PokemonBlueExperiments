import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Define the RNN model
class RNNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNAgent, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# PyBoy Setup
pyboy = PyBoy("pokemon_blue.gb")
pyboy.set_emulation_speed(0)

# Load an existing save state
with open("pokemon_blue.gb.state", "rb") as f:
    pyboy.load_state(f)

# Action Mapping (Includes Releases)
actions = {
    0: WindowEvent.PRESS_ARROW_UP,
    1: WindowEvent.PRESS_ARROW_DOWN,
    2: WindowEvent.PRESS_ARROW_LEFT,
    3: WindowEvent.PRESS_ARROW_RIGHT,
}

release_actions = {
    0: WindowEvent.RELEASE_ARROW_UP,
    1: WindowEvent.RELEASE_ARROW_DOWN,
    2: WindowEvent.RELEASE_ARROW_LEFT,
    3: WindowEvent.RELEASE_ARROW_RIGHT,
}

# Hyperparameters
input_size = 10  # Extracted state data
hidden_size = 128
output_size = len(actions)
learning_rate = 0.001
num_episodes = 100
steps_per_episode = 100  # Limit steps per episode

# Initialize model and optimizer
model = RNNAgent(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Track player position
def get_player_position():
    return (pyboy.memory[0xD362], pyboy.memory[0xD361])  # X, Y coordinates

# Execute action with delays
def execute_action(action):
    pyboy.send_input(actions[action])  # Press key
    pyboy.tick(5)  # Wait (slows movement)

    pyboy.send_input(release_actions[action])  # Release key
    pyboy.tick(5)  # Additional delay after release

# Training Loop
for episode in range(num_episodes):
    hidden = model.init_hidden(1)
    prev_position = get_player_position()
    visited_positions = set()
    
    for step in range(steps_per_episode):
        # Simulated game state input (Replace with actual extraction logic)
        state = np.random.rand(1, 1, input_size)  
        state = torch.tensor(state, dtype=torch.float32)

        output, hidden = model(state, hidden.detach())
        action = torch.argmax(output).item()

        # Execute action with delay
        execute_action(action)

        # Get new position
        new_position = get_player_position()
        
        # Reward system
        reward = 0
        if new_position not in visited_positions:
            reward += 20  # Higher reward for exploring new areas
            visited_positions.add(new_position)
        elif new_position != prev_position:
            reward += 2  # Reward for moving
        else:
            reward -= 5  # Penalize standing still

        # Update previous position
        prev_position = new_position

        # Training
        target = torch.tensor([reward], dtype=torch.float32, requires_grad=False)
        loss = criterion(output, target.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % 10 == 0:  # Print every 10 episodes
        print(f"Episode {episode}, Loss: {loss.item()}")
        print(f"Total Reward: {reward}")

pyboy.stop()
