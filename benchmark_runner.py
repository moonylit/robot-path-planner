"""Reproducible A* vs UCS benchmarks for four 40x40 grid topologies."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from pathlib import Path

from main import AStar, Grid, MazeGenerator, SearchAlgorithm, SearchResult, UCS


REPEATS = 3
OUTPUT_FILE = Path(__file__).with_name("benchmark_results.md")


def open_grid() -> tuple[Grid, bool]:
    grid = Grid()
    grid.start, grid.goal = (0, 0), (39, 39)
    return grid, True


def dfs_maze() -> tuple[Grid, bool]:
    grid = Grid()
    MazeGenerator.generate(grid, seed=1402)
    return grid, False


def weighted_terrain() -> tuple[Grid, bool]:
    grid = Grid()
    grid.start, grid.goal = (0, 0), (39, 39)
    MazeGenerator.scatter_terrain(grid, seed=4205)
    return grid, False


def u_trap() -> tuple[Grid, bool]:
    grid = Grid()
    grid.start, grid.goal = (20, 18), (20, 34)
    for row in range(10, 26):
        grid.set_wall((row, 12), True)
        grid.set_wall((row, 26), True)
    for col in range(12, 27):
        grid.set_wall((25, col), True)
    return grid, False


SCENARIOS = [
    ("Open unobstructed grid · diagonal corner-to-corner", open_grid),
    ("Procedural DFS maze · seed 1402", dfs_maze),
    ("Weighted swamp and sand · seed 4205", weighted_terrain),
    ("U-shaped concave trap", u_trap),
]


def consume(algorithm: SearchAlgorithm, grid: Grid, diagonal: bool) -> SearchResult:
    assert grid.start is not None and grid.goal is not None
    result = None
    for update in algorithm.steps(grid, grid.start, grid.goal, diagonal):
        if isinstance(update, SearchResult):
            result = update
    if result is None:
        raise RuntimeError(f"{algorithm.name} ended without a SearchResult")
    return result


def measure(name: str, grid: Grid, diagonal: bool) -> tuple[SearchResult, float, float]:
    algorithm_type = AStar if name == "A*" else UCS
    durations: list[float] = []
    result: SearchResult | None = None
    for _ in range(REPEATS):
        started = time.perf_counter()
        result = consume(algorithm_type(), grid.clone(), diagonal)
        durations.append((time.perf_counter() - started) * 1000)

    # Profile memory in a separate run so tracemalloc overhead does not inflate
    # the reported search execution time.
    tracemalloc.start()
    try:
        consume(algorithm_type(), grid.clone(), diagonal)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result is not None
    return result, statistics.mean(durations), peak_bytes / 1024


def markdown_table() -> str:
    header = (
        "| Scenario | Algorithm | Nodes Explored | Execution (ms, mean of 3) | "
        "Peak Frontier | Path Cost | Nodes Saved by A* | Peak Python Memory (KiB) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for scenario, factory in SCENARIOS:
        grid, diagonal = factory()
        outcomes = {name: measure(name, grid, diagonal) for name in ("A*", "UCS")}
        a_result, _, _ = outcomes["A*"]
        u_result, _, _ = outcomes["UCS"]
        if a_result.cost is None or u_result.cost is None:
            raise RuntimeError(f"Benchmark scenario is unreachable: {scenario}")
        if abs(a_result.cost - u_result.cost) > 1e-9:
            raise RuntimeError(f"A* and UCS disagree on optimal cost: {scenario}")
        saved = ((u_result.explored - a_result.explored) / u_result.explored * 100
                 if u_result.explored else 0.0)
        for name in ("A*", "UCS"):
            result, duration, peak_kib = outcomes[name]
            saved_cell = f"{saved:.2f}%" if name == "A*" else "—"
            lines.append(
                f"| {scenario} | {name} | {result.explored:,} | {duration:.4f} | "
                f"{result.peak_frontier:,} | {result.cost:.6f} | {saved_cell} | {peak_kib:.2f} |"
            )
    return "\n".join(lines)


def main() -> None:
    table = markdown_table()
    print(table)
    OUTPUT_FILE.write_text(
        "# Empirical Benchmark Results\n\n"
        "Python `time.perf_counter()` execution time averaged over three runs; "
        "peak Python memory measured separately with `tracemalloc`.\n\n"
        + table + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved benchmark table to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
