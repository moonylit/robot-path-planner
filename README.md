# Robot Path Planning Simulator - INT3120 AI Mini Project

**Group 7 | Mohammad Ayaan & Sarath Mohanraj**

An interactive Pygame application for exploring informed and uninformed search on a 40 × 40 grid. The 800 × 800 grid sits beside a clickable dark control sidebar. Build obstacle and weighted terrain maps, generate a perfect maze with randomized depth-first search, compare A* and Uniform Cost Search, and watch a robot traverse the resulting route.

## Project details

| Area | Implementation |
|---|---|
| Language | Python 3.10+ |
| GUI and rendering | Pygame 2.x, 800 × 800 grid plus 350-pixel control sidebar, 60 FPS event loop |
| External dependencies | `pygame` only (`requirements.txt`) |
| Priority frontier | `heapq`, with a monotonic tie-break counter; insertion and extraction are O(log n) |
| Search bookkeeping | Dictionaries for best-known `g(n)` costs and parent pointers; sets for open and closed membership |
| Runtime measurement | `time.perf_counter()`; search compute time in milliseconds, excluding UI animation waits |
| Frontier memory | Tracks the maximum size of the unique open-set frontier during each search |
| Terrain costs | Normal 1.0; sand 3.0; water 5.0; walls are impassable |
| Movement costs | Orthogonal step = terrain cost; diagonal step = √2 × destination terrain cost, with corner cutting prevented |
| Procedural maps | Recursive-backtracker perfect maze and clustered weighted terrain scatter |

## Algorithms

Both algorithms use the same priority-queue graph-search engine, weighted destination-cell cost model, and parent-pointer path reconstruction. Every move cost is positive. A cell is expanded at most once; improved frontier entries are pushed again and stale entries are skipped when popped. The start cell has no entry cost; each move pays the destination cell's terrain cost.

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
- **8-directional:** `h(n) = max(Δrow, Δcolumn) + (√2 − 1) min(Δrow, Δcolumn)` (Octile distance).

Both distance estimates are multiplied by the minimum terrain cost, 1.0. This accounts for weighted cells without overestimating a route that could remain on normal terrain.

### Heuristic admissibility and consistency

In 4-directional mode, any route must make at least `|Δrow| + |Δcolumn|` unit moves. Each step costs at least 1.0, so scaled Manhattan distance never exceeds the remaining weighted route cost. A single orthogonal move changes Manhattan distance by at most one; the destination step costs at least 1.0, so the heuristic is consistent.

In 8-directional mode, Octile distance is the exact obstacle-free cost with orthogonal moves costing 1 and diagonals √2. It is also consistent across each allowed move. Terrain multipliers are at least 1, so scaling by the minimum terrain cost preserves the lower bound and consistency. Preventing corner cutting removes possible routes but does not invalidate the bound.

Both heuristics are zero at the goal. Thus A* with these heuristics returns an optimal least-cost path. UCS is equivalent to using `h(n) = 0`.

## Benchmark and robot traversal

Choose **Benchmark Both (X)** or press **X**. A* and then UCS run sequentially against the same captured grid, terrain, endpoints, and movement mode. The live analytics card compares explored nodes and percentage of search space saved by A*, measured execution time, peak open-set size, and final path costs. The card marks the optimal costs as matching when both algorithms find the same least-cost total. Search timing excludes the pauses used to animate frontier expansion.

After a normal search, or after both benchmark runs complete, a glowing robot icon moves along the reconstructed route. Its facing indicator follows the route direction, and the analytics card updates the accumulated terrain-weighted cost as it crosses cells.

## Performance comparison

| Measure | A* | UCS |
|---|---|---|
| Frontier priority | `g(n) + h(n)` | `g(n)` |
| Theoretical search behavior | Uses goal direction and can avoid costly terrain | Expands by accumulated terrain cost without goal direction |
| Worst-case time | O((V + E) log V) with a binary heap | O((V + E) log V) with a binary heap |
| Worst-case memory | O(V + E) | O(V + E) |
| Empirical nodes, runtime, frontier size | **Benchmark Both (X)** reports explored nodes, compute milliseconds, and peak open-set size | The same captured-board run reports UCS metrics after A* finishes |

Empirical results depend on the obstacle layout, movement mode, and machine. Press **X** to run both algorithms consecutively on the current board. The resulting metrics remain visible for direct comparison; each runtime measures the search iterator, not drawing or animation time.

## Controls

| Sidebar button | Action |
|---|---|
| **A* Search / Uniform Cost (UCS)** | Select the search algorithm |
| **Start / Goal, Wall, Sand, Water, Eraser** | Select the grid brush |
| **DFS Maze (M) / Terrain Map (T)** | Generate a map |
| **Run Search (Space) / Benchmark Both (X)** | Run one search or compare both |
| **Reset Path (R) / Clear All (C)** | Clear search overlays or reset the board |
| **Movement** | Toggle 4-directional / 8-directional movement |
| **1 Slow / 2 Normal / 3 Fast / 4 Instant** | Set search animation speed |

Keyboard shortcuts remain available: **A**, **U**, **W**, **S**, **E**, **D**, **M**, **T**, **Space**, **X**, **R**, **C**, and **1–4**.

On the grid, select **Start / Goal** and click to place the start then goal; subsequent clicks cycle their positions. Select a brush to paint cells by clicking or dragging. Right-click erases a cell or endpoint.

Search visualization uses light blue for the open set, orange for the closed set, and teal for the reconstructed lowest-cost path. Sand and water cells are tinted and labeled with their entry cost. The sidebar analytics shows comparison metrics and the robot's live accumulated path cost.

## Run the project

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python main.py
```

Set endpoints with the **Start / Goal** button, select a brush in the sidebar, and paint by clicking or dragging. Use the sidebar controls or keyboard to generate a perfect DFS maze, scatter weighted terrain, select movement and animation modes, run a search, or compare A* and UCS.
