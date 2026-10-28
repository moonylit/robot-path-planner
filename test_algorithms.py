"""Headless unit tests for path planning optimality and movement costs."""

from __future__ import annotations

import math
import unittest

from main import (AStar, DIAGONAL_COST, Grid, MazeGenerator, SearchResult, UCS,
                  VIVA_PRESET_NAMES, load_viva_preset)


def solve(algorithm: AStar | UCS, grid: Grid, diagonal: bool = False) -> SearchResult:
    """Consume the animated search generator without opening a Pygame window."""
    assert grid.start is not None and grid.goal is not None
    for update in algorithm.steps(grid, grid.start, grid.goal, diagonal):
        if isinstance(update, SearchResult):
            return update
    raise AssertionError("Search generator ended without returning a result")


class SearchAlgorithmTests(unittest.TestCase):
    def test_optimality_unweighted(self) -> None:
        open_grid = Grid()
        open_grid.start, open_grid.goal = (0, 0), (39, 39)
        for diagonal in (False, True):
            with self.subTest(topology="open", diagonal=diagonal):
                astar = solve(AStar(), open_grid, diagonal)
                ucs = solve(UCS(), open_grid, diagonal)
                self.assertIsNotNone(astar.cost)
                self.assertEqual(astar.cost, ucs.cost)

        maze = Grid()
        MazeGenerator.generate(maze, seed=1402)
        for diagonal in (False, True):
            with self.subTest(topology="maze", diagonal=diagonal):
                astar = solve(AStar(), maze, diagonal)
                ucs = solve(UCS(), maze, diagonal)
                self.assertIsNotNone(astar.cost)
                self.assertEqual(astar.cost, ucs.cost)

    def test_optimality_weighted_terrains(self) -> None:
        grid = Grid()
        grid.start, grid.goal = (10, 5), (10, 9)
        grid.set_terrain((10, 6), "sand")
        grid.set_terrain((10, 7), "water")
        grid.set_terrain((10, 8), "sand")

        astar = solve(AStar(), grid)
        ucs = solve(UCS(), grid)
        self.assertEqual(astar.cost, ucs.cost)
        self.assertEqual(astar.cost, 6.0)
        self.assertTrue(all(row != 10 for row, _ in astar.path[1:-1]))

    def test_node_efficiency(self) -> None:
        grid = Grid()
        grid.start, grid.goal = (0, 0), (39, 39)
        for diagonal in (False, True):
            with self.subTest(diagonal=diagonal):
                astar = solve(AStar(), grid, diagonal)
                ucs = solve(UCS(), grid, diagonal)
                self.assertLessEqual(astar.explored, ucs.explored)

    def test_unreachable_goal(self) -> None:
        for enclosed in ("start", "goal"):
            grid = Grid()
            grid.start, grid.goal = (10, 10), (20, 20)
            center = grid.start if enclosed == "start" else grid.goal
            assert center is not None
            row, col = center
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                grid.set_wall(neighbor, True)
            for algorithm in (AStar(), UCS()):
                with self.subTest(enclosed=enclosed, algorithm=algorithm.name):
                    result = solve(algorithm, grid)
                    self.assertEqual(result.path, [])
                    self.assertIsNone(result.cost)

    def test_diagonal_kinematics(self) -> None:
        grid = Grid()
        grid.start, grid.goal = (0, 0), (10, 10)
        four_way = solve(AStar(), grid, diagonal=False)
        eight_way = solve(AStar(), grid, diagonal=True)
        self.assertEqual(four_way.cost, 20.0)
        self.assertAlmostEqual(eight_way.cost or 0.0, 10 * DIAGONAL_COST)
        self.assertLess(eight_way.cost or math.inf, four_way.cost or math.inf)

        grid.set_terrain((1, 1), "water")
        diagonal_edges = dict(grid.neighbors((0, 0), diagonal=True))
        self.assertAlmostEqual(diagonal_edges[(1, 1)], 5.0 * DIAGONAL_COST)


class VivaPresetTests(unittest.TestCase):
    def test_all_presets_load_expected_endpoints_and_are_solvable(self) -> None:
        self.assertEqual(set(VIVA_PRESET_NAMES), {"F1", "F2", "F3", "F4"})
        for preset in VIVA_PRESET_NAMES:
            with self.subTest(preset=preset):
                grid = Grid()
                diagonal = load_viva_preset(grid, preset)
                self.assertIsNotNone(grid.start)
                self.assertIsNotNone(grid.goal)
                self.assertNotEqual(grid.start, grid.goal)
                result = solve(AStar(), grid, diagonal)
                self.assertTrue(result.path)
                self.assertEqual(result.path[0], grid.start)
                self.assertEqual(result.path[-1], grid.goal)
                self.assertIsNotNone(result.cost)

    def test_preset_layouts_demonstrate_their_concepts(self) -> None:
        grid = Grid()
        self.assertFalse(load_viva_preset(grid, "F1"))
        self.assertTrue(grid.is_wall((25, 18)))  # U base closes the greedy route.
        self.assertFalse(grid.is_wall((9, 18)))  # Escape is northward.

        self.assertFalse(load_viva_preset(grid, "F2"))
        direct_cost = sum(
            grid.terrain_cost((20, col))
            for col in range(4, 37)
        )
        result = solve(AStar(), grid)
        self.assertIsNotNone(result.cost)
        self.assertLess(result.cost or math.inf, direct_cost)
        self.assertGreater(len(result.path), 34)  # The cheapest route takes a detour.

        self.assertFalse(load_viva_preset(grid, "F3"))
        self.assertTrue(grid.is_wall((19, 20)))
        self.assertFalse(grid.is_wall((20, 20)))
        bottleneck = solve(AStar(), grid)
        self.assertEqual(bottleneck.cost, 51.0)

        self.assertTrue(load_viva_preset(grid, "F4"))
        open_field = solve(AStar(), grid, diagonal=True)
        self.assertAlmostEqual(open_field.cost or 0.0, 39 * DIAGONAL_COST)

    def test_preset_loading_is_deterministic_and_clears_previous_map(self) -> None:
        grid = Grid()
        load_viva_preset(grid, "F2")
        first_walls = grid.walls()
        first_terrain = [(spot.position, spot.terrain)
                         for row in grid.spots for spot in row
                         if spot.terrain != "normal"]
        grid.set_wall((0, 0), True)
        load_viva_preset(grid, "F2")
        self.assertEqual(grid.walls(), first_walls)
        self.assertEqual([(spot.position, spot.terrain)
                          for row in grid.spots for spot in row
                          if spot.terrain != "normal"], first_terrain)


if __name__ == "__main__":
    unittest.main()
