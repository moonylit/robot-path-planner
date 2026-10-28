# Empirical Benchmark Results

Python `time.perf_counter()` execution time averaged over three runs; peak Python memory measured separately with `tracemalloc`.

| Scenario | Algorithm | Nodes Explored | Execution (ms, mean of 3) | Peak Frontier | Path Cost | Nodes Saved by A* | Peak Python Memory (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Open unobstructed grid · diagonal corner-to-corner | A* | 40 | 2.4238 | 155 | 55.154329 | 97.50% | 223.77 |
| Open unobstructed grid · diagonal corner-to-corner | UCS | 1,600 | 18.3790 | 79 | 55.154329 | — | 526.65 |
| Procedural DFS maze · seed 1402 | A* | 355 | 3.4205 | 6 | 248.000000 | 4.57% | 260.12 |
| Procedural DFS maze · seed 1402 | UCS | 372 | 3.4700 | 5 | 248.000000 | — | 259.70 |
| Weighted swamp and sand · seed 4205 | A* | 788 | 8.0762 | 147 | 90.000000 | 49.10% | 330.77 |
| Weighted swamp and sand · seed 4205 | UCS | 1,548 | 11.9445 | 69 | 90.000000 | — | 525.16 |
| U-shaped concave trap | A* | 290 | 2.5899 | 41 | 38.000000 | 67.88% | 214.61 |
| U-shaped concave trap | UCS | 903 | 6.4993 | 42 | 38.000000 | — | 318.96 |
