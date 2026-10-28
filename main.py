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


# The canvas is exactly 800 x 800, with a 350-pixel interactive sidebar.
ROWS = COLS = 40
CELL_SIZE = 20
GRID_WIDTH = COLS * CELL_SIZE
GRID_HEIGHT = ROWS * CELL_SIZE
SIDEBAR_WIDTH = 350
WIDTH = GRID_WIDTH + SIDEBAR_WIDTH
HEIGHT = GRID_HEIGHT
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

    def clone(self) -> "Grid":
        duplicate = Grid()
        duplicate.start, duplicate.goal = self.start, self.goal
        for r in range(ROWS):
            for c in range(COLS):
                duplicate.spots[r][c].wall = self.spots[r][c].wall
                duplicate.spots[r][c].terrain = self.spots[r][c].terrain
        return duplicate


@dataclass
class SearchResult:
    path: list[Cell]
    explored: int
    cost: float | None
    runtime_ms: float
    peak_frontier: int


@dataclass
class SearchProgress:
    open_set: set[Cell]
    closed: set[Cell]
    explored: int
    runtime_ms: float
    peak_frontier: int


class SearchAlgorithm:
    """Shared priority-queue search engine; subclasses supply priority heuristic."""

    name = "Search"

    def heuristic(self, cell: Cell, goal: Cell, diagonal: bool) -> float:
        return 0.0

    def steps(self, grid: Grid, start: Cell, goal: Cell, diagonal: bool
              ) -> Iterator[SearchProgress | SearchResult]:
        counter = itertools.count()
        costs: dict[Cell, float] = {start: 0.0}
        came_from: dict[Cell, Cell] = {}
        closed: set[Cell] = set()
        open_set: set[Cell] = {start}
        peak_frontier = 1
        frontier: list[tuple[float, int, Cell]] = [
            (self.heuristic(start, goal, diagonal), next(counter), start)
        ]
        search_runtime = 0.0

        while frontier:
            phase_started = time.perf_counter()
            _, _, current = heapq.heappop(frontier)
            if current in closed:
                search_runtime += time.perf_counter() - phase_started
                continue
            open_set.discard(current)
            closed.add(current)
            search_runtime += time.perf_counter() - phase_started
            yield SearchProgress(set(open_set), set(closed), len(closed),
                                 search_runtime * 1000, peak_frontier)

            if current == goal:
                phase_started = time.perf_counter()
                path = [goal]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                search_runtime += time.perf_counter() - phase_started
                yield SearchResult(path, len(closed), costs[goal],
                                   search_runtime * 1000, peak_frontier)
                return

            phase_started = time.perf_counter()
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
                    peak_frontier = max(peak_frontier, len(open_set))
            search_runtime += time.perf_counter() - phase_started

        yield SearchResult([], len(closed), None, search_runtime * 1000, peak_frontier)


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


@dataclass
class Button:
    key: str
    label: str
    rect: pygame.Rect


class Renderer:
    """Draw the 800px grid, clickable dark sidebar, analytics, and robot."""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.small = pygame.font.SysFont("segoeui", 13)
        self.body = pygame.font.SysFont("segoeui", 15)
        self.title = pygame.font.SysFont("segoeui", 21, bold=True)
        self.section_font = pygame.font.SysFont("segoeui", 12, bold=True)
        self.buttons: dict[str, Button] = {}
        self._build_buttons()
        self.terrain_tints = {
            "sand": pygame.Surface((CELL_SIZE - 1, CELL_SIZE - 1), pygame.SRCALPHA),
            "water": pygame.Surface((CELL_SIZE - 1, CELL_SIZE - 1), pygame.SRCALPHA),
        }
        self.terrain_tints["sand"].fill((*SAND, 210))
        self.terrain_tints["water"].fill((*WATER, 210))
        self.robot_glow = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(self.robot_glow, (32, 222, 204, 35), (20, 20), 19)
        pygame.draw.circle(self.robot_glow, (32, 222, 204, 70), (20, 20), 13)

    def _build_buttons(self) -> None:
        x0, w, gap = GRID_WIDTH + 16, SIDEBAR_WIDTH - 32, 8

        def add(key: str, label: str, x: int, y: int, width: int, height: int = 28) -> None:
            self.buttons[key] = Button(key, label, pygame.Rect(x, y, width, height))

        half = (w - gap) // 2
        add("algo_astar", "A* Search", x0, 69, half)
        add("algo_ucs", "Uniform Cost (UCS)", x0 + half + gap, 69, half)
        add("brush_start", "Start / Goal", x0, 123, half)
        add("brush_wall", "Wall", x0 + half + gap, 123, half)
        add("brush_sand", "Sand (3×)", x0, 157, half)
        add("brush_water", "Water (5×)", x0 + half + gap, 157, half)
        add("brush_eraser", "Eraser", x0, 191, w)
        add("maze", "DFS Maze (M)", x0, 244, half)
        add("terrain", "Terrain Map (T)", x0 + half + gap, 244, half)
        add("run", "Run Search (SPACE)", x0, 297, w)
        add("benchmark", "Benchmark Both (X)", x0, 331, w)
        add("reset", "Reset Path (R)", x0, 365, half)
        add("clear", "Clear All (C)", x0 + half + gap, 365, half)
        add("movement", "4-Directional", x0, 419, w)
        speed_gap = 5
        speed_w = (w - speed_gap * 3) // 4
        for i, label in enumerate(("1 Slow", "2 Normal", "3 Fast", "4 Instant"), 1):
            add(f"speed_{i}", label, x0 + (i - 1) * (speed_w + speed_gap), 473, speed_w)

    def label(self, text: str, pos: tuple[int, int], color: tuple[int, int, int],
              font: pygame.font.Font | None = None) -> None:
        self.screen.blit((font or self.body).render(text, True, color), pos)

    def draw(self, app: "PathPlanningApp") -> None:
        screen = self.screen
        screen.fill(BG)
        pygame.draw.rect(screen, EMPTY, (0, 0, GRID_WIDTH, GRID_HEIGHT))
        for r in range(ROWS):
            for c in range(COLS):
                cell = (r, c)
                terrain = app.grid.spots[r][c].terrain
                color = None
                if app.grid.is_wall(cell):
                    color = WALL
                elif cell in app.path:
                    color = PATH_COLOR
                elif cell in app.closed:
                    color = CLOSED_COLOR
                elif cell in app.open_set:
                    color = OPEN_COLOR
                if color:
                    pygame.draw.rect(screen, color, (c * CELL_SIZE + 1, r * CELL_SIZE + 1,
                                                     CELL_SIZE - 1, CELL_SIZE - 1))
                elif terrain in self.terrain_tints:
                    pos = (c * CELL_SIZE + 1, r * CELL_SIZE + 1)
                    screen.blit(self.terrain_tints[terrain], pos)
                    number = "3" if terrain == "sand" else "5"
                    ink = (75, 63, 39) if terrain == "sand" else (25, 55, 80)
                    glyph = self.small.render(number, True, ink)
                    screen.blit(glyph, glyph.get_rect(center=(pos[0] + 9, pos[1] + 9)))
        for c in range(COLS + 1):
            pygame.draw.line(screen, GRID_LINE, (c * CELL_SIZE, 0), (c * CELL_SIZE, GRID_HEIGHT))
        for r in range(ROWS + 1):
            pygame.draw.line(screen, GRID_LINE, (0, r * CELL_SIZE), (GRID_WIDTH, r * CELL_SIZE))
        self.draw_marker(app.grid.start, CYAN, "S")
        self.draw_marker(app.grid.goal, RED, "G")
        self.draw_robot(app)
        self.draw_sidebar(app)

    def draw_marker(self, cell: Cell | None, color: tuple[int, int, int], text: str) -> None:
        if cell is None:
            return
        r, c = cell
        rect = pygame.Rect(c * CELL_SIZE + 2, r * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        marker = self.small.render(text, True, (255, 255, 255))
        self.screen.blit(marker, marker.get_rect(center=rect.center))

    def draw_robot(self, app: "PathPlanningApp") -> None:
        state = app.robot_state()
        if state is None:
            return
        row, col, _accumulated, direction = state
        center = (round(col * CELL_SIZE + CELL_SIZE / 2), round(row * CELL_SIZE + CELL_SIZE / 2))
        self.screen.blit(self.robot_glow, (center[0] - 20, center[1] - 20))
        pygame.draw.circle(self.screen, (13, 62, 73), center, 9)
        pygame.draw.circle(self.screen, (63, 241, 214), center, 7, 2)
        if direction is not None:
            dr, dc = direction
            end = (round(center[0] + dc * 8), round(center[1] + dr * 8))
            pygame.draw.line(self.screen, (255, 255, 255), center, end, 3)
            pygame.draw.circle(self.screen, (255, 255, 255), end, 2)

    def draw_sidebar(self, app: "PathPlanningApp") -> None:
        screen = self.screen
        panel = pygame.Rect(GRID_WIDTH, 0, SIDEBAR_WIDTH, HEIGHT)
        pygame.draw.rect(screen, (19, 25, 36), panel)
        pygame.draw.line(screen, (53, 67, 85), (GRID_WIDTH, 0), (GRID_WIDTH, HEIGHT), 2)
        self.label("PATHFINDER", (GRID_WIDTH + 16, 14), TEXT, self.title)
        self.label("INT3120 AI  ·  GROUP 7", (GRID_WIDTH + 18, 40), MUTED, self.small)
        self.section("SEARCH ALGORITHM", 54)
        self.section("CELL BRUSH", 108)
        self.section("MAP GENERATORS", 228)
        self.section("ACTIONS", 282)
        self.section("MOVEMENT", 404)
        self.section("ANIMATION SPEED", 457)
        self.section("LIVE BENCHMARK ANALYTICS", 511)

        mouse = pygame.mouse.get_pos()
        for key, button in self.buttons.items():
            hovered = button.rect.collidepoint(mouse)
            active = app.button_active(key)
            fill = (42, 61, 76) if active else ((47, 59, 76) if hovered else (34, 44, 59))
            border = (49, 210, 190) if active else ((105, 133, 163) if hovered else (57, 70, 88))
            pygame.draw.rect(screen, fill, button.rect, border_radius=7)
            pygame.draw.rect(screen, border, button.rect, 1, border_radius=7)
            label = button.label
            if key == "movement":
                label = "8-Directional" if app.diagonal else "4-Directional"
            rendered = self.small.render(label, True, TEXT if active or hovered else (202, 212, 225))
            screen.blit(rendered, rendered.get_rect(center=button.rect.center))

        self.draw_analytics(app)

    def section(self, text: str, y: int) -> None:
        self.label(text, (GRID_WIDTH + 17, y), (132, 153, 177), self.section_font)

    def draw_analytics(self, app: "PathPlanningApp") -> None:
        x, y, width, height = GRID_WIDTH + 16, 532, SIDEBAR_WIDTH - 32, 254
        card = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (27, 37, 51), card, border_radius=10)
        pygame.draw.rect(self.screen, (56, 76, 98), card, 1, border_radius=10)
        a = app.benchmark_results.get("A*")
        u = app.benchmark_results.get("UCS")
        live_time = app.current_runtime_ms if app.searching else None

        def values(name: str) -> tuple[str, str, str, str]:
            result = a if name == "A*" else u
            is_live = app.searching and app.algorithm.name == name
            if result:
                return (f"{result.explored:,}", f"{result.runtime_ms:.3f}",
                        f"{result.peak_frontier:,}", "—" if result.cost is None else f"{result.cost:.3f}")
            if is_live:
                return (f"{app.explored:,}", f"{live_time:.3f}" if live_time is not None else "—",
                        f"{app.peak_frontier:,}", "running")
            return ("—", "—", "—", "—")

        an, at, af, ac = values("A*")
        un, ut, uf, uc = values("UCS")
        saved = "awaiting both runs"
        cost_text = f"A* {ac}  ·  UCS {uc}"
        if a and u:
            pct_saved = 0.0 if u.explored == 0 else (u.explored - a.explored) / u.explored * 100
            saved = f"{pct_saved:+.1f}% by A*"
            costs_match = a.cost is not None and u.cost is not None and math.isclose(
                a.cost, u.cost, rel_tol=1e-10, abs_tol=1e-10)
            cost_text = f"{a.cost:.3f}  {'MATCH' if costs_match else 'MISMATCH'}" if costs_match else f"A* {ac}  ·  UCS {uc}"
        rows = [
            ("NODES", f"A* {an}   ·   UCS {un}"),
            ("SPACE SAVED", saved),
            ("EXECUTION", f"A* {at} ms  ·  UCS {ut} ms"),
            ("PEAK FRONTIER", f"A* {af}   ·   UCS {uf}"),
            ("OPTIMAL COST", cost_text),
        ]
        for index, (label, value) in enumerate(rows):
            yy = y + 13 + index * 33
            self.label(label, (x + 11, yy), (132, 153, 177), self.section_font)
            self.label(value, (x + 115, yy - 1), TEXT, self.small)
        path_cells = str(app.path_length) if app.path_length is not None else "—"
        path_cost = f"{app.path_cost:.3f}" if app.path_cost is not None else "—"
        self.label("CURRENT PATH", (x + 11, y + 178), (132, 153, 177), self.section_font)
        self.label(f"{path_cells} cells  ·  cost {path_cost}", (x + 115, y + 177), TEXT, self.small)
        pygame.draw.line(self.screen, (56, 76, 98), (x + 10, y + 208), (x + width - 10, y + 208), 1)
        self.label(f"ROBOT COST  {app.robot_cost:.3f}", (x + 11, y + 216), (79, 231, 205), self.body)
        if app.searching:
            status = f"{app.algorithm.name} expanding · {app.explored:,} nodes"
        elif app.robot_path:
            status = "Robot traversing optimal path" if app.robot_is_moving else app.status
        else:
            status = app.status
        self.label(status[:39], (x + 11, y + 237), MUTED, self.small)


class PathPlanningApp:
    """Owns app state, sidebar actions, search control, and agent animation."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Robot Path Planning Simulator — INT3120 AI")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.grid = Grid()
        self.algorithms: dict[str, SearchAlgorithm] = {"A*": AStar(), "UCS": UCS()}
        self.algorithm: SearchAlgorithm = self.algorithms["A*"]
        self.diagonal = False
        self.speed = 2
        self.brush = "start_goal"
        self.marker_next = "start"
        self.benchmark_results: dict[str, SearchResult] = {}
        self.benchmark_queue: list[str] = []
        self.benchmark_active = False
        self.benchmark_grid: Grid | None = None
        self.search_grid: Grid | None = None
        self.reset_path()

    def reset_path(self) -> None:
        self.open_set: set[Cell] = set()
        self.closed: set[Cell] = set()
        self.path: set[Cell] = set()
        self.explored = 0
        self.path_cost: float | None = None
        self.path_length: int | None = None
        self.runtime_ms: float | None = None
        self.peak_frontier = 0
        self.current_runtime_ms = 0.0
        self.status = "Place start and goal"
        self.searching = False
        self.search_iterator = None
        self.last_advance = pygame.time.get_ticks()
        self.step_accumulator = 0.0
        self.batch_limit = 1
        self.robot_path: list[Cell] = []
        self.robot_grid: Grid | None = None
        self.robot_started: float | None = None
        self.robot_step_duration = 0.15
        self.robot_is_moving = False
        self.robot_cost = 0.0

    def reset_path_only(self) -> None:
        self.reset_path()
        self.benchmark_active = False
        self.benchmark_results = {}
        self.benchmark_queue = []
        self.benchmark_grid = None
        self.status = "Path reset"

    def board_cell(self, pos: tuple[int, int]) -> Cell | None:
        x, y = pos
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            return y // CELL_SIZE, x // CELL_SIZE
        return None

    def start_search(self) -> None:
        if self.grid.start is None or self.grid.goal is None:
            self.status = "Place start and goal before searching"
            return
        self.benchmark_active = False
        self.benchmark_results = {}
        self.benchmark_queue = []
        self.reset_path()
        self.search_grid = self.grid.clone()
        self.status = f"Searching with {self.algorithm.name}..."
        self.search_iterator = self.algorithm.steps(self.search_grid, self.grid.start,
                                                    self.grid.goal, self.diagonal)
        self.searching = True
        self.last_advance = pygame.time.get_ticks()

    def start_benchmark(self) -> None:
        if self.grid.start is None or self.grid.goal is None:
            self.status = "Place start and goal before benchmarking"
            return
        self.benchmark_results = {}
        self.benchmark_queue = ["A*", "UCS"]
        self.benchmark_active = True
        self.benchmark_grid = self.grid.clone()
        self.run_next_benchmark()

    def run_next_benchmark(self) -> None:
        if not self.benchmark_queue:
            self.searching = False
            self.search_iterator = None
            self.benchmark_active = False
            self.status = "Benchmark complete"
            return
        name = self.benchmark_queue.pop(0)
        self.algorithm = self.algorithms[name]
        self.reset_path()
        self.status = f"Benchmarking {name}..."
        assert self.benchmark_grid is not None
        self.search_iterator = self.algorithm.steps(self.benchmark_grid,
                                                    self.benchmark_grid.start,
                                                    self.benchmark_grid.goal,
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
            self.open_set, self.closed, self.explored = step.open_set, step.closed, step.explored
            self.peak_frontier = step.peak_frontier
            self.current_runtime_ms = step.runtime_ms

    def finish_search(self, result: SearchResult | None) -> None:
        self.searching = False
        if result is not None:
            self.explored = result.explored
            self.peak_frontier = result.peak_frontier
            self.current_runtime_ms = result.runtime_ms
            self.path = set(result.path)
            self.path_cost = result.cost
            self.path_length = len(result.path) if result.path else None
            self.runtime_ms = result.runtime_ms
            self.status = "Path found" if result.path else "No path found"
            if self.benchmark_active:
                self.benchmark_results[self.algorithm.name] = result
        self.search_iterator = None
        if self.benchmark_queue:
            self.run_next_benchmark()
            return
        if self.benchmark_active:
            self.benchmark_active = False
            a = self.benchmark_results.get("A*")
            u = self.benchmark_results.get("UCS")
            equal = bool(a and u and a.cost is not None and u.cost is not None
                         and math.isclose(a.cost, u.cost, rel_tol=1e-10, abs_tol=1e-10))
            self.status = "Benchmark complete · optimal costs match" if equal else "Benchmark complete · verify results"
            self.begin_robot(result.path if result else [], self.benchmark_grid)
        elif result is not None:
            self.begin_robot(result.path, self.search_grid)

    def begin_robot(self, route: list[Cell], source_grid: Grid | None) -> None:
        self.robot_path = route
        self.robot_grid = source_grid.clone() if source_grid is not None else None
        self.robot_started = time.perf_counter() if route else None
        self.robot_step_duration = {1: 0.34, 2: 0.16, 3: 0.07, 4: 0.035}[self.speed]
        self.robot_is_moving = len(route) > 1
        self.robot_cost = 0.0

    def robot_state(self) -> tuple[float, float, float, tuple[float, float] | None] | None:
        if not self.robot_path or self.robot_started is None:
            return None
        edges = max(0, len(self.robot_path) - 1)
        elapsed = max(0.0, time.perf_counter() - self.robot_started)
        segment_float = elapsed / self.robot_step_duration
        segment = min(edges, int(segment_float))
        fraction = min(1.0, max(0.0, segment_float - segment)) if segment < edges else 0.0
        accumulated = 0.0
        for i in range(segment):
            accumulated += self.robot_edge_cost(i)
        direction: tuple[float, float] | None = None
        if segment < edges:
            here, nxt = self.robot_path[segment], self.robot_path[segment + 1]
            dr, dc = nxt[0] - here[0], nxt[1] - here[1]
            move_scale = max(abs(dr), abs(dc))
            direction = (dr / move_scale, dc / move_scale)
            row = here[0] + dr * fraction
            col = here[1] + dc * fraction
            accumulated += self.robot_edge_cost(segment) * fraction
        else:
            row, col = self.robot_path[-1]
        self.robot_is_moving = segment < edges
        self.robot_cost = accumulated
        return row, col, accumulated, direction

    def robot_edge_cost(self, index: int) -> float:
        if self.robot_grid is None:
            return 0.0
        here, nxt = self.robot_path[index], self.robot_path[index + 1]
        diagonal = here[0] != nxt[0] and here[1] != nxt[1]
        return self.robot_grid.terrain_cost(nxt) * (DIAGONAL_COST if diagonal else 1.0)

    def button_active(self, key: str) -> bool:
        if key == "algo_astar":
            return self.algorithm.name == "A*"
        if key == "algo_ucs":
            return self.algorithm.name == "UCS"
        if key == "brush_start":
            return self.brush == "start_goal"
        if key.startswith("brush_"):
            return self.brush == key.removeprefix("brush_")
        if key.startswith("speed_"):
            return self.speed == int(key[-1])
        return False

    def activate_button(self, key: str) -> None:
        if key == "algo_astar":
            self.algorithm = self.algorithms["A*"]
            self.reset_path_only()
        elif key == "algo_ucs":
            self.algorithm = self.algorithms["UCS"]
            self.reset_path_only()
        elif key.startswith("brush_"):
            self.brush = "start_goal" if key == "brush_start" else key.removeprefix("brush_")
        elif key == "maze":
            MazeGenerator.generate(self.grid)
            self.marker_next = "goal"
            self.reset_path_only()
            self.status = "Perfect DFS maze generated"
        elif key == "terrain":
            MazeGenerator.scatter_terrain(self.grid)
            self.reset_path_only()
            self.status = "Weighted terrain scattered"
        elif key == "run":
            self.start_search()
        elif key == "benchmark":
            self.start_benchmark()
        elif key == "reset":
            self.reset_path_only()
        elif key == "clear":
            self.clear_board()
        elif key == "movement":
            self.diagonal = not self.diagonal
            self.reset_path_only()
            self.status = "8-directional movement" if self.diagonal else "4-directional movement"
        elif key.startswith("speed_"):
            self.speed = int(key[-1])

    def clear_board(self) -> None:
        self.grid.clear()
        self.marker_next = "start"
        self.reset_path_only()
        self.status = "Board cleared"

    def edit_cell(self, event: pygame.event.Event) -> None:
        cell = self.board_cell(event.pos)
        if cell is None:
            return
        if event.button == 3:
            if cell == self.grid.start:
                self.grid.start = None
                self.marker_next = "start"
            elif cell == self.grid.goal:
                self.grid.goal = None
                self.marker_next = "goal"
            else:
                self.grid.set_terrain(cell, "normal")
            self.reset_path_only()
            self.status = "Cell erased"
        elif event.button == 1:
            self.paint_cell(cell)

    def paint_cell(self, cell: Cell) -> None:
        if self.brush == "start_goal":
            if cell == self.grid.start or cell == self.grid.goal:
                return
            if self.marker_next == "start":
                self.grid.start = cell
                self.marker_next = "goal"
                self.status = "Start placed · place goal"
            else:
                self.grid.goal = cell
                self.marker_next = "start"
                self.status = "Goal placed · ready to search"
        else:
            terrain = {"wall": "wall", "sand": "sand", "water": "water", "eraser": "normal"}[self.brush]
            self.grid.set_terrain(cell, terrain)
        self.reset_search_state()

    def reset_search_state(self) -> None:
        self.reset_path_only()
        if self.grid.start and self.grid.goal:
            self.status = "Board edited · ready"

    def handle_events(self) -> bool:
        shortcuts = {
            pygame.K_a: "algo_astar", pygame.K_u: "algo_ucs",
            pygame.K_w: "brush_wall", pygame.K_s: "brush_sand",
            pygame.K_e: "brush_water", pygame.K_m: "maze", pygame.K_t: "terrain",
            pygame.K_SPACE: "run", pygame.K_x: "benchmark", pygame.K_r: "reset",
            pygame.K_c: "clear", pygame.K_d: "movement",
            pygame.K_1: "speed_1", pygame.K_2: "speed_2",
            pygame.K_3: "speed_3", pygame.K_4: "speed_4",
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in shortcuts:
                self.activate_button(shortcuts[event.key])
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[0] >= GRID_WIDTH:
                    for key, button in self.renderer.buttons.items():
                        if button.rect.collidepoint(event.pos):
                            self.activate_button(key)
                            break
                else:
                    self.edit_cell(event)
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                cell = self.board_cell(event.pos)
                if cell is not None and self.brush != "start_goal":
                    self.paint_cell(cell)
            elif event.type == pygame.MOUSEMOTION and event.buttons[2]:
                cell = self.board_cell(event.pos)
                if cell is not None:
                    self.grid.set_terrain(cell, "normal")
                    self.reset_search_state()
        return True

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.advance_search()
            self.robot_state()
            self.renderer.draw(self)
            self.clock.tick(FPS)
        pygame.quit()


def main() -> None:
    PathPlanningApp().run()


if __name__ == "__main__":
    main()
