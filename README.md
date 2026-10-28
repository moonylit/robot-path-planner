# Robot Path Planning Simulator - INT3120 AI Mini Project

**Group 7 | Mohammad Ayaan & Sarath Mohanraj**

An interactive Pygame application for exploring informed and uninformed search on a 40 × 40 grid. Build obstacle maps, generate a random maze with a guaranteed clear corridor, compare A* and Uniform Cost Search, and inspect each search frontier as it expands.

## Project details

| Area | Implementation |
|---|---|
| Language | Python 3.10+ |
| GUI and rendering | Pygame 2.x, fixed 800 × 800 grid canvas, 60 FPS event loop |
| External dependencies | `pygame` only (`requirements.txt`) |
| Priority frontier | `heapq`, with a monotonic tie-break counter; insertion and extraction are O(log n) |
| Search bookkeeping | Dictionaries for best-known `g(n)` costs and parent pointers; sets for open and closed membership |
| Runtime measurement | `time.perf_counter()`; reported search runtime in milliseconds |
| Movement costs | Orthogonal 1.0; diagonal √2 (approximately 1.414), with corner cutting prevented |

## Algorithms

Both algorithms use the same priority-queue graph-search engine, cost model, and parent-pointer path reconstruction. Every move cost is positive. A cell is expanded at most once; improved frontier entries are pushed again and stale entries are skipped when popped.

### Uniform Cost Search (UCS)

UCS orders the frontier by accumulated cost from the start:

\[
f(n) = g(n)
\]

It is Dijkstra's algorithm on this grid. With equal orthogonal costs it expands an approximately circular wave; with diagonal movement, the wave reflects the lower diagonal cost. UCS is complete and returns a least-cost path when all edge costs are positive.

### A* Search

A* orders the frontier by:

\[
f(n) = g(n) + h(n)
\]

- **4-directional:** `h(n) = |Δrow| + |Δcolumn|` (Manhattan distance).
- **8-directional:** `h(n) = √(Δrow² + Δcolumn²)` (Euclidean distance).

### Heuristic admissibility and consistency

In 4-directional mode, any route must make at least `|Δrow| + |Δcolumn|` unit moves, so Manhattan distance never exceeds the remaining route cost. It is consistent because a single orthogonal move changes Manhattan distance by at most one, the move's cost.

In 8-directional mode, each move has Euclidean length equal to its cost (1 or √2). By the triangle inequality, straight-line distance cannot exceed the cost of any sequence of moves, and the heuristic change across one move is no greater than that move's cost. Euclidean distance is therefore admissible and consistent. Preventing corner cutting removes some possible routes but does not invalidate this lower bound.

Both heuristics are zero at the goal. Thus A* with these heuristics returns an optimal least-cost path. UCS is equivalent to using `h(n) = 0`.

## Performance comparison

| Measure | A* | UCS |
|---|---|---|
| Frontier priority | `g(n) + h(n)` | `g(n)` |
| Theoretical search behavior | Uses goal direction; often expands fewer cells | Explores outward by cost without goal direction |
| Worst-case time | O((V + E) log V) with a binary heap | O((V + E) log V) with a binary heap |
| Worst-case memory | O(V + E) | O(V + E) |
| Empirical nodes and runtime | Use **X** on the same board; the HUD reports explored nodes and measured milliseconds | The same benchmark run reports UCS metrics after A* finishes |

Empirical results depend on the obstacle layout, movement mode, and machine. Press **X** to run both algorithms consecutively on the current board. The resulting metrics remain visible for direct comparison; each runtime measures the search iterator, not drawing or animation time.

## Controls

| Input | Action |
|---|---|
| First left click | Place cyan start |
| Second left click | Place red goal |
| Left click / drag | Paint charcoal walls (after endpoints are placed) |
| Right click / drag | Erase walls; right-clicking a marker removes it |
| **A** | Select A* |
| **U** | Select UCS |
| **D** | Toggle 4-directional / 8-directional movement |
| **M** | Generate a random obstacle map and carve a guaranteed corridor |
| **Space** | Animate the selected search |
| **X** | Benchmark A*, then UCS, on the current board |
| **1 / 2 / 3** | Set slow / medium / instant animation |
| **R** | Reset search overlays and path, preserving walls and endpoints |
| **C** | Clear the entire board, including walls and endpoints |

Search visualization uses light blue for the open set, orange for the closed set, and teal for the reconstructed shortest path. The bottom HUD displays algorithm, movement mode, animation speed, explored nodes, path cost, runtime, and status.

## Run the project

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python main.py
```

Set endpoints with two left clicks, add obstacles or press **M**, choose an algorithm and movement mode, then press **Space**. Press **X** to compare A* and UCS on the same obstacle map.
