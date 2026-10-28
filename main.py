"""Robot Path Planning Simulator — A* and Uniform Cost Search with Pygame."""

from __future__ import annotations

import heapq
import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Iterator

import pygame


# The canvas is exactly 800 x 800: 40 rows/columns at 20 pixels per cell.
ROWS = COLS = 40
CELL_SIZE = 20
GRID_WIDTH = COLS * CELL_SIZE
GRID_HEIGHT = ROWS * CELL_SIZE
HEADER_HEIGHT = 52
FOOTER_HEIGHT = 104
WIDTH = GRID_WIDTH
HEIGHT = HEADER_HEIGHT + GRID_HEIGHT + FOOTER_HEIGHT
FPS = 60

BG = (22, 29, 40)
PANEL = (29, 38, 52)
TEXT = (240, 244, 250)
MUTED = (163, 177, 197)
EMPTY = (250, 251, 253)
GRID_LINE = (217, 223, 231)
CYAN = (20, 190, 215)
RED = (229, 70, 78)
WALL = (30, 35, 43)
SAND = (241, 218, 150)
WATER = (119, 185, 229)
OPEN_COLOR = (166, 222, 248)
CLOSED_COLOR = (255, 166, 73)
PATH_COLOR = (37, 190, 166)
SPEED_INTERVALS = {1: 1000 / 15, 2: 1000 / 60, 3: 1000 / 240, 4: 0}
DIAGONAL_COST = math.sqrt(2)
TERRAIN_COSTS = {"normal": 1.0, "sand": 3.0, "water": 5.0}
BRUSHES = {"wall": "Wall [W]", "sand": "Sand Cost:3 [S]", "water": "Water Cost:5 [E]"}
Cell = tuple[int, int]


@dataclass
class Spot:
    """A grid cell and its persistent obstacle state."""

    row: int
    col: int
    wall: bool = False
    terrain: str = "normal"

    @property
    def position(self) -> Cell:
        return self.row, self.col


class Grid:
    """Owns cells and provides movement-aware neighbors."""

    def __init__(self) -> None:
        self.spots = [[Spot(r, c) for c in range(COLS)] for r in range(ROWS)]
        self.start: Cell | None = None
        self.goal: Cell | None = None

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < ROWS and 0 <= c < COLS

    def is_wall(self, cell: Cell) -> bool:
        return self.spots[cell[0]][cell[1]].wall

    def set_wall(self, cell: Cell, value: bool) -> None:
        if self.in_bounds(cell) and cell not in (self.start, self.goal):
            self.spots[cell[0]][cell[1]].wall = value
            self.spots[cell[0]][cell[1]].terrain = "normal"

    def set_terrain(self, cell: Cell, terrain: str) -> None:
        if self.in_bounds(cell) and cell not in (self.start, self.goal):
            spot = self.spots[cell[0]][cell[1]]
            spot.wall = terrain == "wall"
            spot.terrain = terrain if terrain in TERRAIN_COSTS else "normal"

    def terrain_cost(self, cell: Cell) -> float:
        return TERRAIN_COSTS[self.spots[cell[0]][cell[1]].terrain]

    def walls(self) -> set[Cell]:
        return {spot.position for row in self.spots for spot in row if spot.wall}

    def neighbors(self, cell: Cell, diagonal: bool) -> list[tuple[Cell, float]]:
        r, c = cell
        offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        if diagonal:
            offsets += [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        result: list[tuple[Cell, float]] = []
        for dr, dc in offsets:
            nxt = (r + dr, c + dc)
            if not self.in_bounds(nxt) or self.is_wall(nxt):
                continue
            if dr and dc:
                # Prevent squeezing diagonally between two blocked cells.
                if self.is_wall((r + dr, c)) or self.is_wall((r, c + dc)):
                    continue
                result.append((nxt, DIAGONAL_COST * self.terrain_cost(nxt)))
            else:
                result.append((nxt, self.terrain_cost(nxt)))
        return result

    def clear(self) -> None:
        for row in self.spots:
            for spot in row:
                spot.wall = False
                spot.terrain = "normal"
        self.start = self.goal = None


@dataclass
class SearchResult:
    path: list[Cell]
    explored: int
    cost: float | None
    runtime_ms: float


class SearchAlgorithm:
    """Shared priority-queue search engine; subclasses supply priority heuristic."""

    name = "Search"

    def heuristic(self, cell: Cell, goal: Cell, diagonal: bool) -> float:
        return 0.0

    def steps(self, grid: Grid, start: Cell, goal: Cell, diagonal: bool
              ) -> Iterator[tuple[set[Cell], set[Cell], int] | SearchResult]:
        started = time.perf_counter()
        counter = itertools.count()
        costs: dict[Cell, float] = {start: 0.0}
        came_from: dict[Cell, Cell] = {}
        closed: set[Cell] = set()
        open_set: set[Cell] = {start}
        frontier: list[tuple[float, int, Cell]] = [
            (self.heuristic(start, goal, diagonal), next(counter), start)
        ]

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current in closed:
                continue
            open_set.discard(current)
            closed.add(current)
            yield set(open_set), set(closed), len(closed)

            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                yield SearchResult(path, len(closed), costs[goal],
                                   (time.perf_counter() - started) * 1000)
                return

            for neighbor, move_cost in grid.neighbors(current, diagonal):
                if neighbor in closed:
                    continue
                new_cost = costs[current] + move_cost
                if new_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    came_from[neighbor] = current
                    priority = new_cost + self.heuristic(neighbor, goal, diagonal)
                    heapq.heappush(frontier, (priority, next(counter), neighbor))
                    open_set.add(neighbor)

        yield SearchResult([], len(closed), None, (time.perf_counter() - started) * 1000)


class UCS(SearchAlgorithm):
    """Uniform Cost Search: frontier priority is accumulated cost g(n)."""

    name = "UCS"


class AStar(SearchAlgorithm):
    """A*: Manhattan in 4-dir and Octile in 8-dir, scaled by min terrain cost."""

    name = "A*"

    def heuristic(self, cell: Cell, goal: Cell, diagonal: bool) -> float:
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        if diagonal:
            high, low = max(dr, dc), min(dr, dc)
            return (high + (DIAGONAL_COST - 1) * low) * min(TERRAIN_COSTS.values())
        return float(dr + dc) * min(TERRAIN_COSTS.values())


class MazeGenerator:
    """Generate a perfect maze by randomized DFS recursive backtracking."""

    @staticmethod
    def generate(grid: Grid, seed: int | None = None) -> None:
        rng = random.Random(seed)
        # Maze nodes occupy every other cell; intermediate cells are passages.
        for row in grid.spots:
            for spot in row:
                spot.wall = True
                spot.terrain = "normal"
        nodes = [(r, c) for r in range(0, ROWS, 2) for c in range(0, COLS, 2)]
        visited = {nodes[0]}
        grid.spots[0][0].wall = False
        stack = [nodes[0]]
        while stack:
            r, c = stack[-1]
            candidates = [(r + dr, c + dc, dr, dc) for dr, dc in
                          ((-2, 0), (0, 2), (2, 0), (0, -2))
                          if 0 <= r + dr < ROWS and 0 <= c + dc < COLS
                          and (r + dr, c + dc) not in visited]
            if not candidates:
                stack.pop()
                continue
            nr, nc, dr, dc = rng.choice(candidates)
            grid.spots[r + dr // 2][c + dc // 2].wall = False
            grid.spots[nr][nc].wall = False
            visited.add((nr, nc))
            stack.append((nr, nc))

        # Seed/goal cells are put on the connected maze tree so M always yields
        # a solvable demonstration regardless of the previous endpoint choice.
        grid.start = (0, 0)
        grid.goal = (ROWS - 2, COLS - 2)

    @staticmethod
    def scatter_terrain(grid: Grid, seed: int | None = None) -> None:
        """Scatter clustered weighted terrain while retaining all walls."""
        rng = random.Random(seed)
        for row in grid.spots:
            for spot in row:
                if not spot.wall:
                    spot.terrain = "normal"
        for _ in range(22):
            center_r, center_c = rng.randrange(ROWS), rng.randrange(COLS)
            radius_r, radius_c = rng.randrange(2, 6), rng.randrange(2, 6)
            terrain = rng.choice(("sand", "water"))
            for r in range(max(0, center_r - radius_r), min(ROWS, center_r + radius_r + 1)):
                for c in range(max(0, center_c - radius_c), min(COLS, center_c + radius_c + 1)):
                    cell = (r, c)
                    if cell not in (grid.start, grid.goal) and not grid.is_wall(cell):
                        if ((r - center_r) / radius_r) ** 2 + ((c - center_c) / radius_c) ** 2 <= 1:
                            grid.spots[r][c].terrain = terrain


class Renderer:
    """Draw the grid, animated search overlays, markers, legend, and HUD."""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.font = pygame.font.SysFont("segoeui", 16)
        self.small = pygame.font.SysFont("segoeui", 13)
        self.title = pygame.font.SysFont("segoeui", 20, bold=True)
        self.terrain_tints = {
            "sand": pygame.Surface((CELL_SIZE - 1, CELL_SIZE - 1), pygame.SRCALPHA),
            "water": pygame.Surface((CELL_SIZE - 1, CELL_SIZE - 1), pygame.SRCALPHA),
        }
        self.terrain_tints["sand"].fill((*SAND, 210))
        self.terrain_tints["water"].fill((*WATER, 210))

    def label(self, text: str, pos: tuple[int, int], color: tuple[int, int, int],
              font: pygame.font.Font | None = None) -> None:
        self.screen.blit((font or self.font).render(text, True, color), pos)

    def draw(self, app: "PathPlanningApp") -> None:
        screen = self.screen
        screen.fill(BG)
        self.label("ROBOT PATH PLANNING", (15, 12), TEXT, self.title)
        self.label("INT3120 AI  |  Group 7", (280, 19), MUTED, self.small)
        self.label("Brush W/S/E | Maze M | Terrain T | Benchmark X",
                   (388, 19), MUTED, self.small)

        board_y = HEADER_HEIGHT
        pygame.draw.rect(screen, EMPTY, (0, board_y, GRID_WIDTH, GRID_HEIGHT))
        for r in range(ROWS):
            for c in range(COLS):
                cell = (r, c)
                color = EMPTY
                if app.grid.is_wall(cell):
                    color = WALL
                elif cell in app.path:
                    color = PATH_COLOR
                elif cell in app.closed:
                    color = CLOSED_COLOR
                elif cell in app.open_set:
                    color = OPEN_COLOR
                elif app.grid.spots[r][c].terrain in self.terrain_tints:
                    color = None
                if color != EMPTY:
                    if color is not None:
                        pygame.draw.rect(screen, color, (c * CELL_SIZE + 1, board_y + r * CELL_SIZE + 1,
                                                         CELL_SIZE - 1, CELL_SIZE - 1))
                terrain = app.grid.spots[r][c].terrain
                if not app.grid.is_wall(cell) and terrain in self.terrain_tints and cell not in app.path and cell not in app.closed and cell not in app.open_set:
                    tile_pos = (c * CELL_SIZE + 1, board_y + r * CELL_SIZE + 1)
                    screen.blit(self.terrain_tints[terrain], tile_pos)
                    multiplier = "3" if terrain == "sand" else "5"
                    marker = self.small.render(multiplier, True, (75, 63, 39) if terrain == "sand" else (25, 55, 80))
                    screen.blit(marker, marker.get_rect(center=(tile_pos[0] + 9, tile_pos[1] + 9)))
        for c in range(COLS + 1):
            pygame.draw.line(screen, GRID_LINE, (c * CELL_SIZE, board_y),
                             (c * CELL_SIZE, board_y + GRID_HEIGHT))
        for r in range(ROWS + 1):
            pygame.draw.line(screen, GRID_LINE, (0, board_y + r * CELL_SIZE),
                             (GRID_WIDTH, board_y + r * CELL_SIZE))
        self.draw_marker(app.grid.start, CYAN, "S")
        self.draw_marker(app.grid.goal, RED, "G")

        fy = HEADER_HEIGHT + GRID_HEIGHT
        pygame.draw.rect(screen, PANEL, (0, fy, WIDTH, FOOTER_HEIGHT))
        pygame.draw.line(screen, (58, 73, 93), (0, fy), (WIDTH, fy), 2)
        cost = f"{app.path_cost:.3f}" if app.path_cost is not None else "—"
        runtime = f"{app.runtime_ms:.3f} ms" if app.runtime_ms is not None else "—"
        mode = "8-dir" if app.diagonal else "4-dir"
        brush = BRUSHES[app.brush]
        self.label(f"{app.algorithm.name}  |  {mode}  |  Speed {app.speed}  |  Brush: {brush}",
                   (14, fy + 10), TEXT, self.small)
        path_length = str(app.path_length) if app.path_length is not None else "—"
        self.label(f"Nodes {app.explored}  |  Path length {path_length} cells  |  Total cost {cost}  |  Runtime {runtime}",
                   (14, fy + 33), TEXT, self.small)
        status_color = (116, 224, 168) if "found" in app.status.lower() else MUTED
        self.label(f"STATUS  {app.status}", (14, fy + 56), status_color, self.small)
        self.draw_legend(fy + 81)

        if app.benchmark_results:
            a = app.benchmark_results.get("A*")
            u = app.benchmark_results.get("UCS")
            if a and u:
                summary = f"Benchmark  A*: {a.explored} nodes / {a.runtime_ms:.2f} ms    UCS: {u.explored} nodes / {u.runtime_ms:.2f} ms"
                self.label(summary, (14, fy + 56), (255, 220, 125), self.small)

    def draw_marker(self, cell: Cell | None, color: tuple[int, int, int], text: str) -> None:
        if cell is None:
            return
        r, c = cell
        rect = pygame.Rect(c * CELL_SIZE + 2, HEADER_HEIGHT + r * CELL_SIZE + 2,
                           CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        marker = self.small.render(text, True, (255, 255, 255))
        self.screen.blit(marker, marker.get_rect(center=rect.center))

    def draw_legend(self, y: int) -> None:
        items = [(CYAN, "Start"), (RED, "Goal"), (WALL, "Wall"),
                 (SAND, "Sand ×3"), (WATER, "Water ×5"),
                 (OPEN_COLOR, "Open"), (CLOSED_COLOR, "Closed"), (PATH_COLOR, "Path")]
        x = 14
        for color, label in items:
            pygame.draw.rect(self.screen, color, (x, y + 2, 12, 12), border_radius=2)
            self.label(label, (x + 17, y), MUTED, self.small)
            x += 82


class PathPlanningApp:
    """Application controller for editing, running, and benchmarking searches."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Robot Path Planning Simulator — INT3120")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.grid = Grid()
        self.algorithms: dict[str, SearchAlgorithm] = {"A*": AStar(), "UCS": UCS()}
        self.algorithm: SearchAlgorithm = self.algorithms["A*"]
        self.diagonal = False
        self.speed = 2
        self.brush = "wall"
        self.reset_path()
        self.benchmark_results: dict[str, SearchResult] = {}
        self.benchmark_queue: list[str] = []

    def reset_path(self) -> None:
        self.open_set: set[Cell] = set()
        self.closed: set[Cell] = set()
        self.path: set[Cell] = set()
        self.explored = 0
        self.path_cost: float | None = None
        self.path_length: int | None = None
        self.runtime_ms: float | None = None
        self.status = "Place start and goal"
        self.searching = False
        self.search_iterator = None
        self.last_advance = 0
        self.step_accumulator = 0.0
        self.batch_limit = 1

    def reset_path_only(self) -> None:
        self.reset_path()
        self._benchmark_active = False
        self.benchmark_results = {}
        self.benchmark_queue = []
        self.status = "Path reset"

    def board_cell(self, pos: tuple[int, int]) -> Cell | None:
        x, y = pos[0] // CELL_SIZE, (pos[1] - HEADER_HEIGHT) // CELL_SIZE
        if HEADER_HEIGHT <= pos[1] < HEADER_HEIGHT + GRID_HEIGHT and 0 <= x < COLS and 0 <= y < ROWS:
            return y, x
        return None

    def start_search(self, algorithm: SearchAlgorithm | None = None) -> None:
        if self.grid.start is None or self.grid.goal is None:
            self.status = "Place start and goal before searching"
            return
        if algorithm is not None:
            self.algorithm = algorithm
        self._benchmark_active = False
        self.benchmark_results = {}
        self.benchmark_queue = []
        self.reset_path()
        self.status = f"Searching with {self.algorithm.name}..."
        self.search_iterator = self.algorithm.steps(self.grid, self.grid.start, self.grid.goal,
                                                    self.diagonal)
        self.searching = True
        self.last_advance = pygame.time.get_ticks()

    def start_benchmark(self) -> None:
        if self.grid.start is None or self.grid.goal is None:
            self.status = "Place start and goal before benchmarking"
            return
        self.benchmark_results = {}
        self.benchmark_queue = ["A*", "UCS"]
        self._benchmark_active = True
        self.run_next_benchmark()

    def run_next_benchmark(self) -> None:
        if not self.benchmark_queue:
            self.searching = False
            self.search_iterator = None
            self.status = "Benchmark complete"
            return
        name = self.benchmark_queue.pop(0)
        self.algorithm = self.algorithms[name]
        self.reset_path()
        self.status = f"Benchmarking {name}..."
        self.search_iterator = self.algorithm.steps(self.grid, self.grid.start, self.grid.goal,
                                                    self.diagonal)  # type: ignore[arg-type]
        self.searching = True
        self.last_advance = pygame.time.get_ticks()

    def advance_search(self) -> None:
        if not self.searching or self.search_iterator is None:
            return
        now = pygame.time.get_ticks()
        interval = SPEED_INTERVALS[self.speed]
        elapsed = max(0, now - self.last_advance)
        self.last_advance = now
        if self.speed == 4:
            self.batch_limit = 100000
        else:
            self.step_accumulator += elapsed
            self.batch_limit = int(self.step_accumulator / interval)
            if self.batch_limit == 0:
                return
            self.step_accumulator -= self.batch_limit * interval
        for _ in range(self.batch_limit):
            try:
                step = next(self.search_iterator)
            except StopIteration:
                self.finish_search(None)
                break
            if isinstance(step, SearchResult):
                self.finish_search(step)
                break
            self.open_set, self.closed, self.explored = step

    def finish_search(self, result: SearchResult | None) -> None:
        self.searching = False
        if result is not None:
            self.explored = result.explored
            self.path = set(result.path)
            self.path_cost = result.cost
            self.path_length = len(result.path) if result.path else None
            self.runtime_ms = result.runtime_ms
            self.status = "Path found" if result.path else "No path found"
            if self.benchmark_queue or self._benchmark_active:
                self.benchmark_results[self.algorithm.name] = result
        self.search_iterator = None
        if self.benchmark_queue:
            self.run_next_benchmark()
        elif self._benchmark_active:
            self._benchmark_active = False
            self.status = "Benchmark complete"

    @property
    def _benchmark_active(self) -> bool:
        return getattr(self, "benchmark_active", False)

    @_benchmark_active.setter
    def _benchmark_active(self, value: bool) -> None:
        self.benchmark_active = value

    def clear_board(self) -> None:
        self.grid.clear()
        self.reset_path_only()
        self.status = "Board cleared"

    def edit_cell(self, event: pygame.event.Event) -> None:
        cell = self.board_cell(event.pos)
        if cell is None:
            return
        self.reset_path_only()
        if event.button == 3:
            self.grid.set_terrain(cell, "normal")
            if cell == self.grid.start:
                self.grid.start = None
            if cell == self.grid.goal:
                self.grid.goal = None
            self.status = "Cell erased"
        elif event.button == 1:
            if self.grid.start is None:
                self.grid.start = cell
                self.status = "Place goal"
            elif self.grid.goal is None and cell != self.grid.start:
                self.grid.goal = cell
                self.status = "Ready to search"
            else:
                self.grid.set_terrain(cell, self.brush)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.algorithm = self.algorithms["A*"]
                    self.reset_path_only()
                elif event.key == pygame.K_u:
                    self.algorithm = self.algorithms["UCS"]
                    self.reset_path_only()
                elif event.key == pygame.K_d:
                    self.diagonal = not self.diagonal
                    self.reset_path_only()
                    self.status = "8-directional movement" if self.diagonal else "4-directional movement"
                elif event.key == pygame.K_m:
                    MazeGenerator.generate(self.grid)
                    self.reset_path_only()
                    self.status = "Perfect DFS maze generated"
                elif event.key == pygame.K_t:
                    MazeGenerator.scatter_terrain(self.grid)
                    self.reset_path_only()
                    self.status = "Weighted terrain scattered"
                elif event.key == pygame.K_w:
                    self.brush = "wall"
                    self.status = "Wall brush selected"
                elif event.key == pygame.K_s:
                    self.brush = "sand"
                    self.status = "Sand brush selected (cost 3)"
                elif event.key == pygame.K_e:
                    self.brush = "water"
                    self.status = "Water brush selected (cost 5)"
                elif event.key == pygame.K_SPACE:
                    self.benchmark_active = False
                    self.start_search()
                elif event.key == pygame.K_x:
                    self.benchmark_active = True
                    self.start_benchmark()
                elif event.key == pygame.K_c:
                    self.clear_board()
                elif event.key == pygame.K_r:
                    self.reset_path_only()
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    self.speed = event.key - pygame.K_0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.edit_cell(event)
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                cell = self.board_cell(event.pos)
                if cell is not None and self.grid.start is not None and self.grid.goal is not None:
                    self.grid.set_terrain(cell, self.brush)
                    self.reset_path_only()
            elif event.type == pygame.MOUSEMOTION and event.buttons[2]:
                cell = self.board_cell(event.pos)
                if cell is not None and cell not in (self.grid.start, self.grid.goal):
                    spot = self.grid.spots[cell[0]][cell[1]]
                    if spot.wall or spot.terrain != "normal":
                        self.grid.set_terrain(cell, "normal")
                        self.reset_path_only()
        return True

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.advance_search()
            self.renderer.draw(self)
            self.clock.tick(FPS)
        pygame.quit()


def main() -> None:
    PathPlanningApp().run()


if __name__ == "__main__":
    main()
