# Robot Path Planning Simulator - INT3120 AI Mini Project

**Group 7 | Mohammad Ayaan & Sarath Mohanraj**

An interactive Pygame application for exploring informed and uninformed search on a 40 × 40 grid. Build obstacle and weighted terrain maps, generate a perfect maze with randomized depth-first search, compare A* and Uniform Cost Search, and inspect each search frontier as it expands.

## Project details

| Area | Implementation |
|---|---|
| Language | Python 3.10+ |
| GUI and rendering | Pygame 2.x, fixed 800 × 800 grid canvas, 60 FPS event loop |
| External dependencies | `pygame` only (`requirements.txt`) |
| Priority frontier | `heapq`, with a monotonic tie-break counter; insertion and extraction are O(log n) |
| Search bookkeeping | Dictionaries for best-known `g(n)` costs and parent pointers; sets for open and closed membership |
| Runtime measurement | `time.perf_counter()`; reported search runtime in milliseconds |
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

## Performance comparison

| Measure | A* | UCS |
|---|---|---|
| Frontier priority | `g(n) + h(n)` | `g(n)` |
| Theoretical search behavior | Uses goal direction and can avoid costly terrain | Expands by accumulated terrain cost without goal direction |
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
| **W / S / E** | Select wall, sand (cost 3), or water (cost 5) brush |
| **T** | Scatter clustered weighted sand and water terrain |
| **1 / 2 / 3 / 4** | Slow (15 steps/s), normal (60 steps/s), fast (up to 240 steps/s), instant |
| **R** | Reset search overlays and path, preserving walls and endpoints |
| **C** | Clear the entire board, including walls and endpoints |

Search visualization uses light blue for the open set, orange for the closed set, and teal for the reconstructed lowest-cost path. Sand and water cells are tinted and labeled with their entry cost. The bottom HUD displays algorithm, movement mode, active brush, speed, explored nodes, path length in cells (including start and goal), total weighted cost, runtime, and status.

## Run the project

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python main.py
```

Set endpoints with two left clicks, select a brush with **W**, **S**, or **E**, and paint by clicking or dragging. Press **M** for a perfect DFS maze or **T** for weighted terrain patches. Choose an algorithm and movement mode, then press **Space**. Press **X** to compare A* and UCS on the same weighted map.
