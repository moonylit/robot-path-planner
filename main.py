"""Robot Path Planning Simulator — A* and Uniform Cost Search with Pygame."""

from __future__ import annotations

import heapq
import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
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
    g_scores: dict[Cell, float]


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
            # These sets/maps are stable while the generator is suspended; the
            # renderer reads them before the next expansion mutates them.
            yield SearchProgress(open_set, closed, len(closed),
                                 search_runtime * 1000, peak_frontier, costs)

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


VIVA_PRESET_NAMES = {
    "F1": "The U-Trap (Concave Obstacle)",
    "F2": "The Swamp Detour (Weighted Cost Proof)",
    "F3": "The Narrow Bottleneck",
    "F4": "Open Field Race",
}


def load_viva_preset(grid: Grid, preset: str) -> bool:
    """Load a deterministic 40x40 viva map; return its diagonal-mode setting."""
    if preset not in VIVA_PRESET_NAMES:
        raise ValueError(f"Unknown viva preset: {preset}")
    grid.clear()
    if preset == "F1":
        # The opening of the U faces north; a goal to the right is deceptively
        # close by h(n), but the optimal route must back out and go around.
        grid.start, grid.goal = (20, 18), (20, 34)
        for row in range(10, 26):
            grid.set_wall((row, 13), True)
            grid.set_wall((row, 25), True)
        for col in range(13, 26):
            grid.set_wall((25, col), True)
        return False
    if preset == "F2":
        grid.start, grid.goal = (20, 3), (20, 36)
        for col in range(4, 36):
            grid.set_terrain((20, col), "water" if col % 2 else "sand")
        return False
    if preset == "F3":
        grid.start, grid.goal = (10, 4), (10, 35)
        for row in range(ROWS):
            if row != 20:
                grid.set_wall((row, 20), True)
        return False
    grid.start, grid.goal = (0, 0), (39, 39)
    return True


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
        self.tiny = pygame.font.SysFont("segoeui", 9)
        self.body = pygame.font.SysFont("segoeui", 15)
        self.title = pygame.font.SysFont("segoeui", 21, bold=True)
        self.section_font = pygame.font.SysFont("segoeui", 12, bold=True)
        self.heat_labels: dict[str, pygame.Surface] = {}
        self.buttons: dict[str, Button] = {}
        self.help_close_rect = pygame.Rect(0, 0, 0, 0)
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
        self.modal_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.modal_overlay.fill((6, 11, 19, 205))

    def _build_buttons(self) -> None:
        x0, w, gap = GRID_WIDTH + 16, SIDEBAR_WIDTH - 32, 8

        def add(key: str, label: str, x: int, y: int, width: int, height: int = 28) -> None:
            self.buttons[key] = Button(key, label, pygame.Rect(x, y, width, height))

        half = (w - gap) // 2
        add("values", "Toggle Values", GRID_WIDTH + 156, 14, 102, 25)
        add("help", "Help [H/?]", GRID_WIDTH + 266, 14, 68, 25)
        add("algo_astar", "A* Search", x0, 69, half)
        add("algo_ucs", "Uniform Cost (UCS)", x0 + half + gap, 69, half)
        add("brush_start", "Start / Goal", x0, 123, half)
        add("brush_wall", "Wall", x0 + half + gap, 123, half)
        add("brush_sand", "Sand (3×)", x0, 157, half)
        add("brush_water", "Water (5×)", x0 + half + gap, 157, half)
        add("brush_eraser", "Eraser", x0, 191, w)
        add("maze", "DFS Maze (M)", x0, 244, half)
        add("terrain", "Terrain Map (T)", x0 + half + gap, 244, half)
        preset_gap = 4
        preset_w = (w - preset_gap * 3) // 4
        for index, (preset, label) in enumerate((
            ("F1", "F1 U-Trap"), ("F2", "F2 Swamp"),
            ("F3", "F3 Pinch"), ("F4", "F4 Open"),
        )):
            add(f"preset_{preset.lower()}", label,
                x0 + index * (preset_w + preset_gap), 298, preset_w)
        add("run", "Run Search (SPACE)", x0, 352, w)
        add("benchmark", "Benchmark Both (X)", x0, 384, w)
        add("reset", "Reset Path (R)", x0, 416, half)
        add("clear", "Clear All (C)", x0 + half + gap, 416, half)
        add("movement", "4-Directional", x0, 465, w)
        speed_gap = 5
        speed_w = (w - speed_gap * 3) // 4
        for i, label in enumerate(("1 Slow", "2 Normal", "3 Fast", "4 Instant"), 1):
            add(f"speed_{i}", label, x0 + (i - 1) * (speed_w + speed_gap), 516, speed_w)
        file_gap = 9
        file_w = (w - file_gap * 2) // 3
        for index, (key, label) in enumerate((
            ("save_map", "Save Map (K)"), ("load_map", "Load Map (L)"),
            ("screenshot", "Screenshot (P)"),
        )):
            add(key, label, x0 + index * (file_w + file_gap), 567, file_w)

    def label(self, text: str, pos: tuple[int, int], color: tuple[int, int, int],
              font: pygame.font.Font | None = None) -> None:
        self.screen.blit((font or self.body).render(text, True, color), pos)

    def draw(self, app: "PathPlanningApp") -> None:
        screen = self.screen
        screen.fill(BG)
        pygame.draw.rect(screen, EMPTY, (0, 0, GRID_WIDTH, GRID_HEIGHT))
        heat_scale = max(app.g_scores.values(), default=1.0)
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
                if app.show_values and not app.grid.is_wall(cell) and cell in app.g_scores:
                    value = app.g_scores[cell]
                    ratio = min(1.0, value / max(heat_scale, 1e-9))
                    heat = (round(35 + 200 * ratio), round(175 - 55 * ratio), round(220 - 145 * ratio))
                    pygame.draw.rect(screen, heat, (c * CELL_SIZE + 2, r * CELL_SIZE + 2,
                                                    CELL_SIZE - 3, CELL_SIZE - 3))
                    text = f"{value:.0f}"
                    if text not in self.heat_labels:
                        self.heat_labels[text] = self.tiny.render(text, True, (18, 27, 39))
                    glyph = self.heat_labels[text]
                    screen.blit(glyph, glyph.get_rect(center=(c * CELL_SIZE + 10, r * CELL_SIZE + 10)))
        for c in range(COLS + 1):
            pygame.draw.line(screen, GRID_LINE, (c * CELL_SIZE, 0), (c * CELL_SIZE, GRID_HEIGHT))
        for r in range(ROWS + 1):
            pygame.draw.line(screen, GRID_LINE, (0, r * CELL_SIZE), (GRID_WIDTH, r * CELL_SIZE))
        self.draw_marker(app.grid.start, CYAN, "S")
        self.draw_marker(app.grid.goal, RED, "G")
        self.draw_robot(app)
        self.draw_sidebar(app)
        if app.show_help:
            self.draw_help_modal()

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
        self.section("VIVA DEMO PRESETS  ·  F1–F4", 282)
        self.section("ACTIONS", 337)
        self.section("MOVEMENT", 450)
        self.section("ANIMATION SPEED", 500)
        self.section("MAP FILES & EXPORT", 551)
        self.section("LIVE BENCHMARK ANALYTICS", 603)

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
            elif key == "values":
                label = "Values: On" if app.show_values else "Toggle Values"
            elif key.startswith("preset_"):
                preset = key.removeprefix("preset_").upper()
                label = f"{preset} ✓" if app.active_preset == preset else button.label
            rendered = self.small.render(label, True, TEXT if active or hovered else (202, 212, 225))
            screen.blit(rendered, rendered.get_rect(center=button.rect.center))

        self.draw_analytics(app)

    def section(self, text: str, y: int) -> None:
        self.label(text, (GRID_WIDTH + 17, y), (132, 153, 177), self.section_font)

    def draw_analytics(self, app: "PathPlanningApp") -> None:
        x, y, width, height = GRID_WIDTH + 16, 620, SIDEBAR_WIDTH - 32, 174
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
            yy = y + 9 + index * 21
            self.label(label, (x + 11, yy), (132, 153, 177), self.section_font)
            self.label(value, (x + 115, yy - 1), TEXT, self.small)
        path_cells = str(app.path_length) if app.path_length is not None else "—"
        path_cost = f"{app.path_cost:.3f}" if app.path_cost is not None else "—"
        self.label("PATH / ROBOT", (x + 11, y + 118), (132, 153, 177), self.section_font)
        self.label(f"{path_cells} cells · {path_cost} / {app.robot_cost:.3f}",
                   (x + 103, y + 118), TEXT, self.tiny)
        hover = app.hover_info(pygame.mouse.get_pos()) if app.show_values else None
        if hover:
            cell, values = hover
            g = "—" if values["g"] is None else f"{values['g']:.9g}"
            h = f"{values['h']:.9g}"
            f = "—" if values["f"] is None else f"{values['f']:.9g}"
            hover_text = f"g {g} · h {h} · f {f}"
            self.label(f"CELL {cell[0]},{cell[1]}", (x + 11, y + 139), (132, 153, 177), self.section_font)
            self.label(hover_text, (x + 83, y + 139), TEXT, self.tiny)
        else:
            hover_text = "Press V · hover a grid cell"
            self.label("HOVER g/h/f", (x + 11, y + 139), (132, 153, 177), self.section_font)
            self.label(hover_text, (x + 115, y + 138), TEXT, self.small)
        pygame.draw.line(self.screen, (56, 76, 98), (x + 10, y + 158), (x + width - 10, y + 158), 1)
        if app.searching:
            status = f"{app.algorithm.name} expanding · {app.explored:,} nodes"
        elif app.robot_path:
            status = "Robot traversing optimal path" if app.robot_is_moving else app.status
        else:
            status = app.status
        self.label(status[:39], (x + 11, y + 159), MUTED, self.tiny)

    def draw_help_modal(self) -> None:
        self.screen.blit(self.modal_overlay, (0, 0))
        modal = pygame.Rect((WIDTH - 650) // 2, (HEIGHT - 580) // 2, 650, 580)
        pygame.draw.rect(self.screen, (30, 42, 58), modal, border_radius=16)
        pygame.draw.rect(self.screen, (84, 112, 141), modal, 1, border_radius=16)
        self.label("QUICK HELP · ROBOT PATH PLANNING", (modal.x + 28, modal.y + 23), TEXT, self.title)
        self.label("MOUSE", (modal.x + 28, modal.y + 70), (79, 231, 205), self.section_font)
        mouse_lines = [
            "Start / Goal brush: click to place endpoints; later clicks cycle them.",
            "Left-click or drag paints the selected Wall, Sand, Water, or Eraser brush.",
            "Right-click erases a cell; right-click a marker to remove it.",
            "Click a remaining route cell to add a dynamic wall and re-plan from the robot.",
            "Normal terrain costs 1 · Sand costs 3 · Water costs 5 · Walls are blocked.",
        ]
        self._modal_lines(mouse_lines, modal.x + 28, modal.y + 94)
        self.label("KEYBOARD SHORTCUTS", (modal.x + 28, modal.y + 228), (79, 231, 205), self.section_font)
        key_lines = [
            "A / U: choose A* / UCS      D: 4- or 8-directional movement",
            "W / S / E: Wall / Sand / Water brush      M / T: maze / terrain map",
            "Space: run search      X: benchmark both      R: reset path      C: clear all",
            "1–4: animation speed      V: g(n) heatmap and hover inspector",
            "F1–F4: Viva presets      K / L: save / load map      P: screenshot",
            "H or ?: toggle this help panel      Esc: close this panel",
        ]
        self._modal_lines(key_lines, modal.x + 28, modal.y + 252)
        self.label("SEARCH FORMULAS", (modal.x + 28, modal.y + 392), (79, 231, 205), self.section_font)
        formula_lines = [
            "UCS: f(n) = g(n)      A*: f(n) = g(n) + h(n)",
            "4-dir h: Manhattan distance      8-dir h: Octile distance",
            "Diagonal move cost = √2 × destination terrain cost.",
            "A* heuristics are scaled by min terrain cost = 1.0, preserving admissibility.",
        ]
        self._modal_lines(formula_lines, modal.x + 28, modal.y + 416)
        self.help_close_rect = pygame.Rect(modal.right - 118, modal.bottom - 48, 90, 30)
        pygame.draw.rect(self.screen, (41, 62, 76), self.help_close_rect, border_radius=7)
        pygame.draw.rect(self.screen, (49, 210, 190), self.help_close_rect, 1, border_radius=7)
        label = self.small.render("Close · Esc", True, TEXT)
        self.screen.blit(label, label.get_rect(center=self.help_close_rect.center))

    def _modal_lines(self, lines: list[str], x: int, y: int) -> None:
        for index, line in enumerate(lines):
            self.label(line, (x, y + index * 23), (218, 227, 238), self.small)


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
        self.active_preset: str | None = None
        self.show_values = False
        self.show_help = False
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
        self.g_scores: dict[Cell, float] = {}
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
            self.g_scores = step.g_scores

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

    def hover_info(self, mouse_pos: tuple[int, int]) -> tuple[Cell, dict[str, float | None]] | None:
        cell = self.board_cell(mouse_pos)
        if cell is None or self.grid.goal is None:
            return None
        g = self.g_scores.get(cell)
        h = self.algorithm.heuristic(cell, self.grid.goal, self.diagonal)
        f = None if g is None else g + (h if self.algorithm.name == "A*" else 0.0)
        return cell, {"g": g, "h": h, "f": f}

    def remaining_route_index(self) -> int:
        if not self.robot_path or self.robot_started is None:
            return 0
        edges = len(self.robot_path) - 1
        if edges <= 0:
            return 0
        elapsed = max(0.0, time.perf_counter() - self.robot_started)
        if elapsed >= edges * self.robot_step_duration:
            # A static displayed path can still be edited: replay its traversal
            # from the original start to demonstrate a fresh detour.
            return 0
        segment_float = elapsed / self.robot_step_duration
        segment = min(edges - 1, int(segment_float))
        if segment_float - segment >= 0.5:
            segment += 1
        return segment

    def inject_dynamic_obstacle(self, cell: Cell) -> bool:
        if not self.robot_path or cell not in self.robot_path or self.grid.goal is None:
            return False
        index = self.remaining_route_index()
        if cell not in self.robot_path[index:]:
            return False
        source = self.robot_path[index]
        goal = self.grid.goal
        self.grid.start = source
        # A robot may be standing on the injected cell; it must be able to
        # leave that coordinate, so clear its source after placing the wall.
        spot = self.grid.spots[cell[0]][cell[1]]
        spot.wall = True
        spot.terrain = "normal"
        self.grid.spots[source[0]][source[1]].wall = False
        self.grid.goal = goal
        self.marker_next = "goal"
        self.benchmark_results = {}
        self.benchmark_queue = []
        self.benchmark_active = False
        self.start_search()
        self.robot_path = [source]
        self.robot_grid = self.grid.clone()
        self.robot_started = time.perf_counter()
        self.robot_is_moving = False
        self.robot_cost = 0.0
        self.status = "Dynamic obstacle detected · re-planning"
        return True

    def button_active(self, key: str) -> bool:
        if key.startswith("preset_"):
            return self.active_preset == key.removeprefix("preset_").upper()
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
        if key == "values":
            return self.show_values
        if key == "help":
            return self.show_help
        return False

    def activate_button(self, key: str) -> None:
        if key.startswith("preset_"):
            self.load_preset(key.removeprefix("preset_").upper())
        elif key == "algo_astar":
            self.algorithm = self.algorithms["A*"]
            self.reset_path_only()
        elif key == "algo_ucs":
            self.algorithm = self.algorithms["UCS"]
            self.reset_path_only()
        elif key.startswith("brush_"):
            self.brush = "start_goal" if key == "brush_start" else key.removeprefix("brush_")
        elif key == "maze":
            MazeGenerator.generate(self.grid)
            self.active_preset = None
            self.marker_next = "goal"
            self.reset_path_only()
            self.status = "Perfect DFS maze generated"
        elif key == "terrain":
            MazeGenerator.scatter_terrain(self.grid)
            self.active_preset = None
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
        elif key == "values":
            self.show_values = not self.show_values
        elif key == "help":
            self.show_help = not self.show_help
        elif key == "save_map":
            self.save_map()
        elif key == "load_map":
            self.load_map()
        elif key == "screenshot":
            self.export_screenshot()

    def load_preset(self, preset: str) -> None:
        self.diagonal = load_viva_preset(self.grid, preset)
        self.active_preset = preset
        self.marker_next = "goal"
        self.reset_path_only()
        self.status = f"Loaded {preset}: {VIVA_PRESET_NAMES[preset]}"

    def save_map(self, path: str | None = None) -> str | None:
        map_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_map.json")
        data = {
            "version": 1,
            "rows": ROWS,
            "cols": COLS,
            "start": list(self.grid.start) if self.grid.start else None,
            "goal": list(self.grid.goal) if self.grid.goal else None,
            "walls": [list(cell) for cell in sorted(self.grid.walls())],
            "terrain": [
                {"cell": [r, c], "type": self.grid.spots[r][c].terrain}
                for r in range(ROWS) for c in range(COLS)
                if self.grid.spots[r][c].terrain in ("sand", "water")
            ],
            "diagonal": self.diagonal,
            "algorithm": self.algorithm.name,
            "speed": self.speed,
            "brush": self.brush,
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(map_path)), exist_ok=True)
            with open(map_path, "w", encoding="utf-8") as map_file:
                json.dump(data, map_file, indent=2)
                map_file.write("\n")
            self.status = f"Map saved: {os.path.basename(map_path)}"
            return map_path
        except OSError as error:
            self.status = f"Map save failed: {error}"
            return None

    def load_map(self, path: str | None = None) -> bool:
        map_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_map.json")
        try:
            with open(map_path, "r", encoding="utf-8") as map_file:
                data = json.load(map_file)
            if not isinstance(data, dict):
                raise ValueError("Map root must be an object")
            if data.get("version") != 1 or data.get("rows") != ROWS or data.get("cols") != COLS:
                raise ValueError("Unsupported map version or dimensions")

            loaded = Grid()

            def read_cell(value: object, field: str) -> Cell | None:
                if value is None:
                    return None
                if (not isinstance(value, list) or len(value) != 2
                        or any(type(axis) is not int for axis in value)):
                    raise ValueError(f"Invalid {field} coordinate")
                cell = (value[0], value[1])
                if not loaded.in_bounds(cell):
                    raise ValueError(f"{field} is outside the grid")
                return cell

            loaded.start = read_cell(data.get("start"), "start")
            loaded.goal = read_cell(data.get("goal"), "goal")
            if loaded.start is not None and loaded.start == loaded.goal:
                raise ValueError("Start and goal must be different cells")
            walls, terrain = data.get("walls", []), data.get("terrain", [])
            if not isinstance(walls, list) or not isinstance(terrain, list):
                raise ValueError("Walls and terrain must be lists")
            for item in walls:
                cell = read_cell(item, "wall")
                if cell in (loaded.start, loaded.goal):
                    raise ValueError("An endpoint cannot also be a wall")
                loaded.set_terrain(cell, "wall")  # type: ignore[arg-type]
            for item in terrain:
                if not isinstance(item, dict) or item.get("type") not in ("sand", "water"):
                    raise ValueError("Invalid terrain entry")
                cell = read_cell(item.get("cell"), "terrain")
                if cell in (loaded.start, loaded.goal) or loaded.is_wall(cell):  # type: ignore[arg-type]
                    raise ValueError("Terrain cannot overlap an endpoint or wall")
                loaded.set_terrain(cell, item["type"])  # type: ignore[arg-type]

            algorithm, speed = data.get("algorithm", "A*"), data.get("speed", 2)
            brush = data.get("brush", "start_goal")
            diagonal = data.get("diagonal", False)
            if algorithm not in self.algorithms or type(speed) is not int or speed not in (1, 2, 3, 4):
                raise ValueError("Invalid algorithm or speed setting")
            if brush not in ("start_goal", "wall", "sand", "water", "eraser"):
                raise ValueError("Invalid brush setting")
            if type(diagonal) is not bool:
                raise ValueError("Movement mode must be a boolean")

            self.grid = loaded
            self.algorithm = self.algorithms[algorithm]
            self.speed, self.brush, self.diagonal = speed, brush, diagonal
            self.marker_next = "start" if loaded.start is None else "goal"
            self.active_preset = None
            self.reset_path_only()
            self.status = f"Map loaded: {os.path.basename(map_path)}"
            return True
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
            self.status = f"Map load failed: {error}"
            return False

    def export_screenshot(self, directory: str | None = None) -> str | None:
        folder = directory or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "screenshots")
        filename = f"robot_path_planner_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        screenshot_path = os.path.join(folder, filename)
        try:
            os.makedirs(folder, exist_ok=True)
            self.renderer.draw(self)
            pygame.display.flip()
            pygame.image.save(self.screen, screenshot_path)
            self.status = f"Screenshot saved: {filename}"
            return screenshot_path
        except (OSError, pygame.error) as error:
            self.status = f"Screenshot failed: {error}"
            return None

    def clear_board(self) -> None:
        self.grid.clear()
        self.marker_next = "start"
        self.active_preset = None
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
            if self.inject_dynamic_obstacle(cell):
                return
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
        self.active_preset = None
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
            pygame.K_v: "values", pygame.K_h: "help",
            pygame.K_F1: "preset_f1", pygame.K_F2: "preset_f2",
            pygame.K_F3: "preset_f3", pygame.K_F4: "preset_f4",
            pygame.K_k: "save_map", pygame.K_l: "load_map", pygame.K_p: "screenshot",
            pygame.K_1: "speed_1", pygame.K_2: "speed_2",
            pygame.K_3: "speed_3", pygame.K_4: "speed_4",
        }
        for event in pygame.event.get():
            if self.show_help:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_QUESTION):
                        self.show_help = False
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if getattr(self.renderer, "help_close_rect", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        self.show_help = False
                    continue
                if event.type == pygame.QUIT:
                    return False
                continue
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in shortcuts:
                self.activate_button(shortcuts[event.key])
            elif event.type == pygame.KEYDOWN and (event.key == pygame.K_QUESTION or
                    (event.key == pygame.K_SLASH and event.mod & pygame.KMOD_SHIFT)):
                self.show_help = not self.show_help
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
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


def main() -> None:
    PathPlanningApp().run()


if __name__ == "__main__":
    main()
