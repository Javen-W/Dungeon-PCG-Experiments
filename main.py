import pygame
import numpy as np
import neat
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import label
import os
from datetime import datetime

# Constants
GRID_SIZE = 20  # Use 20x20 for faster testing (change to 100x100 for final)
VIEW_SIZE = 5   # 5x5 local view for CNN input
TILE_TYPES = {'EMPTY': 0, 'WALL': 1, 'PATH': 2, 'START': 3, 'END': 4}
MAX_STEPS = GRID_SIZE * GRID_SIZE  # Maximum tiles to place
NUM_GENERATIONS = 100
SAVE_DIR = "maps"
os.makedirs(SAVE_DIR, exist_ok=True)

# Map class to manage tilemap state
class Map:
    def __init__(self, size, start_pos=None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # Initialize with EMPTY tiles
        self.start_pos = start_pos if start_pos else (size // 2, size // 2)
        self.grid[self.start_pos] = TILE_TYPES['START']
        self.last_pos = self.start_pos  # Last placed tile position
        self.step_count = 0

    def place_tile(self, pos, tile_type):
        if 0 <= pos[0] < self.size and 0 <= pos[1] < self.size:
            if tile_type != TILE_TYPES['START']:  # Prevent overwriting start
                self.grid[pos] = tile_type
                self.last_pos = pos
                self.step_count += 1
            return True
        return False

    def get_local_view(self, pos):
        """Extract a 5x5 view centered at pos, padded if near edges."""
        view = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=int)
        half_view = VIEW_SIZE // 2
        for i in range(VIEW_SIZE):
            for j in range(VIEW_SIZE):
                grid_i = pos[0] - half_view + i
                grid_j = pos[1] - half_view + j
                if 0 <= grid_i < self.size and 0 <= grid_j < self.size:
                    view[i, j] = self.grid[grid_i, grid_j]
        return view

    def get_valid_moves(self, pos):
        """Return valid cardinal direction moves (up, down, left, right)."""
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
        for di, dj in directions:
            new_pos = (pos[0] + di, pos[1] + dj)
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                moves.append(new_pos)
        return moves

# CNN for spatial feature extraction
class TileCNN(nn.Module):
    def __init__(self, input_channels=len(TILE_TYPES), output_size=32):
        super(TileCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Dynamic pooling
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Fitness function
def compute_fitness(tilemap, start_pos, end_pos):
    # Check if end tile exists
    if tilemap[end_pos[0], end_pos[1]] != TILE_TYPES['END']:
        return 0.0

    # Dijkstra's distance
    graph = (tilemap == TILE_TYPES['PATH']) | (tilemap == TILE_TYPES['START']) | (tilemap == TILE_TYPES['END'])
    dist_matrix = dijkstra(graph.astype(float), indices=None, directed=False)
    path_length = dist_matrix[end_pos[0], end_pos[1]]
    if not np.isfinite(path_length):
        return 0.0
    path_score = path_length / (GRID_SIZE * GRID_SIZE)

    # Room detection
    path_regions, num_rooms = label(tilemap == TILE_TYPES['PATH'])
    room_sizes = np.bincount(path_regions.flat)[1:]
    valid_rooms = np.sum((room_sizes >= 10) & (room_sizes <= 50))
    room_score = min(valid_rooms / 3, 1.0)  # Target 3 rooms

    # Tile balance
    path_ratio = np.sum(tilemap == TILE_TYPES['PATH']) / (GRID_SIZE * GRID_SIZE)
    balance_score = max(0, 1 - abs(path_ratio - 0.4) / 0.4)

    # Combine scores
    fitness = 0.5 * path_score + 0.3 * room_score + 0.2 * balance_score
    return fitness

# Visualization function
def visualize_map(tilemap, filename):
    pygame.init()
    cell_size = 20
    screen = pygame.display.set_mode((GRID_SIZE * cell_size, GRID_SIZE * cell_size))
    colors = {
        TILE_TYPES['EMPTY']: (255, 255, 255),  # White
        TILE_TYPES['WALL']: (0, 0, 0),        # Black
        TILE_TYPES['PATH']: (100, 100, 100),  # Gray
        TILE_TYPES['START']: (0, 255, 0),     # Green
        TILE_TYPES['END']: (255, 0, 0)        # Red
    }

    screen.fill((255, 255, 255))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pygame.draw.rect(screen, colors[tilemap[i, j]],
                           (j * cell_size, i * cell_size, cell_size, cell_size))
    pygame.display.flip()
    pygame.image.save(screen, filename)
    pygame.quit()

# Evaluation function for NEAT
def eval_genome(genome, config, cnn):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game_map = Map(GRID_SIZE)
    cnn.eval()

    for step in range(MAX_STEPS):
        # Normalize step count (0 to 1)
        step_count = step / MAX_STEPS

        # Get local view
        local_view = game_map.get_local_view(game_map.last_pos)
        local_view_onehot = np.zeros((VIEW_SIZE, VIEW_SIZE, len(TILE_TYPES)))
        for i in range(VIEW_SIZE):
            for j in range(VIEW_SIZE):
                local_view_onehot[i, j, local_view[i, j]] = 1
        local_view_tensor = torch.tensor(local_view_onehot, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2)

        # CNN feature extraction
        with torch.no_grad():
            features = cnn(local_view_tensor).numpy().flatten()

        # Combine CNN features with step count
        input_features = np.concatenate([features, [step_count]])

        # NEAT prediction
        output = net.activate(input_features)
        direction_probs = output[:4]  # Scores for up, down, left, right
        tile_probs = output[4:]  # Scores for tile types

        # Select direction using argmax with validation
        valid_moves = game_map.get_valid_moves(game_map.last_pos)
        if not valid_moves:
            break
        direction_scores = np.array(direction_probs)
        valid_indices = []
        valid_scores = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
        for idx, (di, dj) in enumerate(directions):
            new_pos = (game_map.last_pos[0] + di, game_map.last_pos[1] + dj)
            if new_pos in valid_moves:
                valid_indices.append(idx)
                valid_scores.append(direction_scores[idx])
        if not valid_scores:
            break
        direction_idx = valid_indices[np.argmax(valid_scores)]
        new_pos = valid_moves[valid_indices.index(direction_idx)]

        # Select tile type using argmax
        tile_type = np.argmax(tile_probs)

        # Place tile
        game_map.place_tile(new_pos, tile_type)

        # Stop if end tile is placed
        if tile_type == TILE_TYPES['END']:
            break

    # Compute fitness
    end_pos = game_map.last_pos if game_map.grid[game_map.last_pos] == TILE_TYPES['END'] else (0, 0)
    fitness = compute_fitness(game_map.grid, game_map.start_pos, end_pos)
    return fitness, game_map.grid


# Main training loop
def train_neat():
    # Initialize CNN
    cnn = TileCNN()

    # Load NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'neat.cfg')

    # Create population
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Best genome tracking
    best_genome = None
    best_fitness = -float('inf')

    def eval_genomes(genomes, config):
        nonlocal best_genome, best_fitness
        for genome_id, genome in genomes:
            fitness, tilemap = eval_genome(genome, config, cnn)
            genome.fitness = fitness
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome
                # Visualize and save best map
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                visualize_map(tilemap, f"{SAVE_DIR}/gen_{pop.generation}_fitness_{fitness:.2f}_{timestamp}.png")

    # Run NEAT
    winner = pop.run(eval_genomes, NUM_GENERATIONS)
    return winner


# Run the training
if __name__ == "__main__":
    # Train the model
    winner = train_neat()
    print("Training complete. Best fitness:", winner.fitness)