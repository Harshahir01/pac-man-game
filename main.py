import pygame
import sys
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 600, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption('Pac-Man')

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Set up the clock for managing the frame rate
clock = pygame.time.Clock()
fps = 10000git add .
git commit -m "Updated dependencies for TensorFlow, NumPy, and Pygame"
git push origin main0  # Increase the frame rate to make the game run faster

# Define the game board layout
board = [
    "####################",
    "#..................#",
    "#.####.#####.####..#",
    "#..................#",
    "#.####.#.###.#.###.#",
    "#......#.....#.....#",
    "####################"
]

# Convert the board to a list of lists for easier manipulation
board = [list(row) for row in board]

# Pac-Man initial position
pacman_x, pacman_y = 1, 1

# Enemy initial positions
enemies = [(18, 1), (18, 5), (1, 5)]

# Rewards count
rewards_collected = 0

def move_pacman(dx, dy):
    global pacman_x, pacman_y
    new_x = pacman_x + dx
    new_y = pacman_y + dy
    if 0 <= new_x < len(board[0]) and 0 <= new_y < len(board) and board[new_y][new_x] != '#':
        pacman_x = new_x
        pacman_y = new_y

def move_enemy(enemy):
    x, y = enemy
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(directions)
    for dx, dy in directions:
        new_x = x + dx
        new_y = y + dy
        if 0 <= new_x < len(board[0]) and 0 <= new_y < len(board) and board[new_y][new_x] != '#':
            return (new_x, new_y)
    return (x, y)

def draw_pacman():
    block_size = 30
    pygame.draw.circle(window, YELLOW, (pacman_x * block_size + block_size // 2, pacman_y * block_size + block_size // 2), block_size // 2)

def draw_enemy():
    block_size = 30
    for enemy in enemies:
        x, y = enemy
        pygame.draw.circle(window, RED, (x * block_size + block_size // 2, y * block_size + block_size // 2), block_size // 2)

def draw_board():
    block_size = 30
    for y, row in enumerate(board):
        for x, char in enumerate(row):
            if char == '#':
                color = BLUE
            elif char == '.':
                color = WHITE
            else:
                color = BLACK
            pygame.draw.rect(window, color, pygame.Rect(x * block_size, y * block_size, block_size, block_size))
            if char == '.':
                pygame.draw.circle(window, WHITE, (x * block_size + block_size // 2, y * block_size + block_size // 2), 5)

def draw_score(score):
    font = pygame.font.SysFont(None, 35)
    score_text = font.render(f"Score: {score}", True, WHITE)
    window.blit(score_text, (width - 150, 10))

def get_state():
    state = np.zeros((len(board), len(board[0])))
    for y in range(len(board)):
        for x in range(len(board[0])):
            if board[y][x] == '#':
                state[y, x] = -1
            elif board[y][x] == '.':
                state[y, x] = 1
    state[pacman_y, pacman_x] = 2
    for ex, ey in enemies:
        state[ey, ex] = -2
    return state.flatten()

# Create the DQN agent
state_size = len(get_state())  # Size of the flattened state representation
action_size = 4  # Pac-Man has 4 possible actions: up, down, left, right
agent = DQNAgent(state_size, action_size)

# Main game loop
running = True
batch_size = 32
while running:
    state = get_state()
    state = np.reshape(state, [1, state_size])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = agent.act(state)
    if action == 0:
        move_pacman(-1, 0)  # Move left
    elif action == 1:
        move_pacman(1, 0)  # Move right
    elif action == 2:
        move_pacman(0, -1)  # Move up
    elif action == 3:
        move_pacman(0, 1)  # Move down

    next_state = get_state()
    next_state = np.reshape(next_state, [1, state_size])

    reward = 0
    if board[pacman_y][pacman_x] == '.':
        reward = 1
        board[pacman_y][pacman_x] = ' '
        rewards_collected += 1

    done = False
    for ex, ey in enemies:
        if pacman_x == ex and pacman_y == ey:
            done = True
            reward = -10  # Penalize for being caught by an enemy

    if not done:
        enemies = [move_enemy(enemy) for enemy in enemies]

    agent.remember(state, action, reward, next_state, done)

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    state = next_state

    window.fill(BLACK)
    draw_board()
    draw_pacman()
    draw_enemy()
    draw_score(rewards_collected)
    pygame.display.flip()

    clock.tick(fps)

pygame.quit()
sys.exit()
