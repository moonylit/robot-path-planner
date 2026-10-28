# Robot Path Planning Simulator

An interactive Python and Pygame project that visualizes a robot finding a route through a grid of obstacles with A* search or Uniform Cost Search (UCS).

## Group details

- **Project:** Robot Path Planning Simulator
- **Group members:** Add your group members' names and student IDs here.
- **Course / section:** Add course and section here.

## Requirements and run instructions

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python main.py
```

## How to use

- Left-click once to place the cyan start, then left-click again to place the red goal.
- After both markers are placed, left-click or click and drag to paint black obstacle cells.
- Right-click a cell to erase a wall; right-clicking a marker removes that marker.
- Press **A** to select A*, **U** to select UCS, **Space** to animate a search, and **C** to clear the board.
- The bottom banner reports the selected algorithm, explored nodes, path cost, and execution status.

## Algorithms

The grid uses four-directional movement, and every move has a cost of one. **Uniform Cost Search** expands the frontier cell with the lowest accumulated path cost, `g(n)`. **A*** orders frontier cells by `f(n) = g(n) + h(n)`, where `h(n)` is Manhattan distance to the goal. Manhattan distance is admissible for this movement model, so A* returns a shortest path while typically exploring fewer cells. The visualization colors frontier cells light blue, explored cells lavender, and the reconstructed route yellow.
