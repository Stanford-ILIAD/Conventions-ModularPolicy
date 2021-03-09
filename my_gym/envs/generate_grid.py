import numpy as np
import json

def generate_grid(grid_size, blocks, blocks_per_color, colors):
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    for color in colors:
        arr = np.random.choice(len(blocks), blocks_per_color, replace=False)
        for i in arr:
            h, w = blocks[i]

            while True:
                r, c = np.random.randint(grid_size - h + 1), np.random.randint(grid_size - w + 1)
                valid = True
                for i in range(r,r+h):
                    for j in range(c, c+w):
                        if grid[i][j]: 
                            valid = False
                if not valid:
                    continue
                for i in range(r,r+h):
                    for j in range(c, c+w):
                        grid[i][j] = color
                break;

    return grid

def generate_mask(grid_size, percent_mask):
    mask = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    grid_sq = grid_size * grid_size
    arr = np.random.choice(grid_sq, int(grid_sq * percent_mask), replace=False)

    for a in arr:
        r = a // grid_size;
        c = a % grid_size;
        mask[r][c] = 1;

    return mask

def make_grids(grid_size, vis1, vis2, blocks, blocks_per_color=1, colors=[2,3]):
    goal_grid = generate_grid(grid_size, blocks, blocks_per_color, colors)

    v, p = [vis1, vis2], [None, None]
    for i in range(2):
        if v[i] == 1:
            p[i] = generate_mask(grid_size, 0)
        elif v[i] == 2:
            p[i] = generate_mask(grid_size, 0.5)
        elif v[i] == 3:
            p[i] = generate_mask(grid_size, 1)
        elif v[i] == 4:
            p[i] = generate_mask(grid_size, float(np.random.uniform() < 0.5) )
        elif v[i] == 5:
            p[0] = generate_mask(grid_size, 0.5)
            p[1] = [[1 - x for x in r] for r in p[0]]
        else:
            raise ValueError('Visibility value is not an integer between [1,5].')
    p1_mask, p2_mask = p[0], p[1]

    goal_grid = np.array(goal_grid)
    p1_grid = (np.array(1)-p1_mask) * goal_grid + p1_mask # masked squares are denotes as 1
    p2_grid = (np.array(1)-p2_mask) * goal_grid + p2_mask

    return goal_grid, p1_grid, p2_grid
