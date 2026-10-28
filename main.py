"""Interactive grid-based robot path planning with A* and uniform-cost search."""

from __future__ import annotations

import heapq
import itertools
import sys
from dataclasses import dataclass

import pygame


# Display and board dimensions.
CELL_SIZE = 30
COLS, ROWS = 30, 19
GRID_WIDTH, GRID_HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE
HEADER_HEIGHT, FOOTER_HEIGHT = 58, 88
WIDTH, HEIGHT = GRID_WIDTH, HEADER_HEIGHT + GRID_HEIGHT + FOOTER_HEIGHT
FPS = 60
ANIMATION_INTERVAL_MS = 25

# Palette.
BACKGROUND = (19, 25, 35)
GRID_BACKGROUND = (239, 243, 248)
GRID_LINE = (205, 213, 224)
TEXT = (235, 241, 249)
MUTED = (157, 171, 190)
CYAN = (31, 190, 212)
RED = (231, 76, 82)
WALL = (22, 27, 35)
OPEN = (153, 218, 246)
CLOSED = (205, 190, 244)
PATH = (255, 205, 78)

Cell = tuple[int, int]


@dataclass
class SearchResult:
    path: list[Cell]
    explored: int
    cost: int | None


def manhattan(a: Cell, b: Cell) -> int:
    """Return Manhattan distance for four-directional grid movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(cell: Cell) -> list[Cell]:
    """Return valid orthogonal neighbors in a stable order."""
    x, y = cell
    return [(nx, ny) for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
            if 0 <= nx < COLS and 0 <= ny < ROWS]


def search_steps(start: Cell, goal: Cell, walls: set[Cell], algorithm: str):
    """Yield search snapshots, then a SearchResult; each move has cost one."""
    counter = itertools.count()
    costs = {start: 0}
    parent: dict[Cell, Cell] = {}
    closed: set[Cell] = set()
    open_cells = {start}
    priority = manhattan(start, goal) if algorithm == "A*" else 0
    frontier = [(priority, next(counter), start)]
    explored = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        open_cells.discard(current)
        closed.add(current)
        explored += 1
        yield (set(open_cells), set(closed), set())

        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            yield SearchResult(path, explored, costs[goal])
            return

        for neighbor in neighbors(current):
            if neighbor in walls or neighbor in closed:
                continue
            new_cost = costs[current] + 1
            if new_cost < costs.get(neighbor, sys.maxsize):
                costs[neighbor] = new_cost
                parent[neighbor] = current
                estimate = manhattan(neighbor, goal) if algorithm == "A*" else 0
                heapq.heappush(frontier, (new_cost + estimate, next(counter), neighbor))
                open_cells.add(neighbor)

    yield SearchResult([], explored, None)


class PathPlanningApp:
    """Owns the Pygame window, board interaction, rendering, and animation."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Robot Path Planning | A* & UCS")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 14)
        self.title_font = pygame.font.SysFont("segoeui", 21, bold=True)
        self.algorithm = "A*"
        self.reset()

    def reset(self) -> None:
        self.start: Cell | None = None
        self.goal: Cell | None = None
        self.walls: set[Cell] = set()
        self.open_cells: set[Cell] = set()
        self.closed: set[Cell] = set()
        self.path: set[Cell] = set()
        self.explored = 0
        self.path_cost: int | None = None
        self.status = "Place start and goal"
        self.searching = False
        self.steps = None
        self.last_step = 0

    def board_cell(self, pos: tuple[int, int]) -> Cell | None:
        x, y = pos[0] // CELL_SIZE, (pos[1] - HEADER_HEIGHT) // CELL_SIZE
        if 0 <= x < COLS and 0 <= y < ROWS and HEADER_HEIGHT <= pos[1] < HEADER_HEIGHT + GRID_HEIGHT:
            return x, y
        return None

    def paint_wall(self, cell: Cell) -> None:
        if cell != self.start and cell != self.goal:
            self.walls.add(cell)

    def start_search(self) -> None:
        if self.start is None or self.goal is None:
            self.status = "Set both start and goal first"
            return
        self.open_cells.clear()
        self.closed.clear()
        self.path.clear()
        self.explored = 0
        self.path_cost = None
        self.status = "Searching..."
        self.steps = iter(search_steps(self.start, self.goal, self.walls, self.algorithm))
        self.searching = True
        self.last_step = pygame.time.get_ticks()

    def handle_click(self, event: pygame.event.Event) -> None:
        cell = self.board_cell(event.pos)
        if cell is None:
            return
        if event.button == 3:
            self.walls.discard(cell)
            if cell == self.start:
                self.start = None
            if cell == self.goal:
                self.goal = None
            self.clear_search()
        elif event.button == 1:
            if self.start is None:
                self.start = cell
                self.status = "Place goal"
            elif self.goal is None and cell != self.start:
                self.goal = cell
                self.status = "Ready to search"
            else:
                self.paint_wall(cell)
            self.clear_search(keep_status=True)

    def clear_search(self, keep_status: bool = False) -> None:
        self.open_cells.clear()
        self.closed.clear()
        self.path.clear()
        self.explored = 0
        self.path_cost = None
        self.searching = False
        self.steps = None
        if not keep_status:
            self.status = "Ready to search" if self.start and self.goal else "Place start and goal"

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    self.reset()
                elif event.key == pygame.K_a:
                    self.algorithm = "A*"
                    self.clear_search()
                elif event.key == pygame.K_u:
                    self.algorithm = "UCS"
                    self.clear_search()
                elif event.key == pygame.K_SPACE:
                    self.start_search()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_click(event)
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                cell = self.board_cell(event.pos)
                if cell is not None and self.start is not None and self.goal is not None:
                    self.paint_wall(cell)
        return True

    def advance_animation(self) -> None:
        if not self.searching or self.steps is None:
            return
        now = pygame.time.get_ticks()
        if now - self.last_step < ANIMATION_INTERVAL_MS:
            return
        self.last_step = now
        try:
            state = next(self.steps)
            if isinstance(state, SearchResult):
                self.searching = False
                self.explored = state.explored
                self.path = set(state.path)
                self.path_cost = state.cost
                self.status = "Path found" if state.path else "No path found"
            else:
                self.open_cells, self.closed, _ = state
                self.explored = len(self.closed)
        except StopIteration:
            self.searching = False

    def draw_text(self, text: str, x: int, y: int, color: tuple[int, int, int] = TEXT,
                  font: pygame.font.Font | None = None) -> None:
        self.screen.blit((font or self.font).render(text, True, color), (x, y))

    def draw(self) -> None:
        self.screen.fill(BACKGROUND)
        self.draw_text("ROBOT PATH PLANNING", 18, 16, TEXT, self.title_font)
        self.draw_text("A* and Uniform Cost Search", 280, 20, MUTED, self.small_font)
        self.draw_text("A: A*   U: UCS   SPACE: Search   C: Clear", WIDTH - 365, 21, MUTED, self.small_font)

        board_top = HEADER_HEIGHT
        pygame.draw.rect(self.screen, GRID_BACKGROUND, (0, board_top, GRID_WIDTH, GRID_HEIGHT))
        for cell in self.walls:
            pygame.draw.rect(self.screen, WALL, (cell[0] * CELL_SIZE + 1, board_top + cell[1] * CELL_SIZE + 1,
                                                CELL_SIZE - 2, CELL_SIZE - 2), border_radius=3)
        for cell in self.closed:
            if cell != self.start and cell != self.goal:
                pygame.draw.rect(self.screen, CLOSED, (cell[0] * CELL_SIZE + 1, board_top + cell[1] * CELL_SIZE + 1,
                                                      CELL_SIZE - 2, CELL_SIZE - 2))
        for cell in self.open_cells:
            if cell != self.start and cell != self.goal:
                pygame.draw.rect(self.screen, OPEN, (cell[0] * CELL_SIZE + 1, board_top + cell[1] * CELL_SIZE + 1,
                                                     CELL_SIZE - 2, CELL_SIZE - 2))
        for cell in self.path:
            if cell != self.start and cell != self.goal:
                pygame.draw.rect(self.screen, PATH, (cell[0] * CELL_SIZE + 1, board_top + cell[1] * CELL_SIZE + 1,
                                                     CELL_SIZE - 2, CELL_SIZE - 2))

        for x in range(COLS + 1):
            pygame.draw.line(self.screen, GRID_LINE, (x * CELL_SIZE, board_top),
                             (x * CELL_SIZE, board_top + GRID_HEIGHT))
        for y in range(ROWS + 1):
            pygame.draw.line(self.screen, GRID_LINE, (0, board_top + y * CELL_SIZE),
                             (GRID_WIDTH, board_top + y * CELL_SIZE))
        self.draw_marker(self.start, CYAN, "S")
        self.draw_marker(self.goal, RED, "G")

        footer_y = HEADER_HEIGHT + GRID_HEIGHT
        pygame.draw.rect(self.screen, (29, 37, 50), (0, footer_y, WIDTH, FOOTER_HEIGHT))
        pygame.draw.line(self.screen, (56, 69, 88), (0, footer_y), (WIDTH, footer_y), 2)
        self.draw_text(f"ALGORITHM  {self.algorithm}", 20, footer_y + 15, TEXT, self.small_font)
        self.draw_text(f"NODES EXPLORED  {self.explored}", 220, footer_y + 15, TEXT, self.small_font)
        cost = str(self.path_cost) if self.path_cost is not None else "—"
        self.draw_text(f"PATH COST  {cost}", 465, footer_y + 15, TEXT, self.small_font)
        status_color = (117, 226, 167) if self.status == "Path found" else MUTED
        self.draw_text(f"STATUS  {self.status}", 635, footer_y + 15, status_color, self.small_font)
        self.draw_legend(footer_y + 48)
        pygame.display.flip()

    def draw_marker(self, cell: Cell | None, color: tuple[int, int, int], label: str) -> None:
        if cell is None:
            return
        center = (cell[0] * CELL_SIZE + CELL_SIZE // 2,
                  HEADER_HEIGHT + cell[1] * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(self.screen, color, center, CELL_SIZE // 2 - 3)
        text = self.small_font.render(label, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=center))

    def draw_legend(self, y: int) -> None:
        items = [(CYAN, "Start"), (RED, "Goal"), (WALL, "Wall"), (OPEN, "Open set"),
                 (CLOSED, "Closed set"), (PATH, "Path")]
        x = 20
        for color, label in items:
            pygame.draw.rect(self.screen, color, (x, y + 2, 13, 13), border_radius=3)
            self.draw_text(label, x + 19, y, MUTED, self.small_font)
            x += 112

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.advance_animation()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


def main() -> None:
    PathPlanningApp().run()


if __name__ == "__main__":
    main()
