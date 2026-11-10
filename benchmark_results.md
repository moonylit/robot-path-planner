# Empirical Benchmark Results

Python `time.perf_counter()` execution time averaged over three runs; peak Python memory measured separately with `tracemalloc`.

| Scenario | Algorithm | Nodes Explored | Execution (ms, mean of 3) | Peak Frontier | Path Cost | Nodes Saved by A* | Peak Python Memory (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Open unobstructed grid · diagonal corner-to-corner | A* | 40 | 1.4951 | 155 | 55.154329 | 97.50% | 223.77 |
| Open unobstructed grid · diagonal corner-to-corner | UCS | 1,600 | 13.9493 | 79 | 55.154329 | — | 526.65 |
| Procedural DFS maze · seed 1402 | A* | 355 | 3.0828 | 6 | 248.000000 | 4.57% | 260.12 |
| Procedural DFS maze · seed 1402 | UCS | 372 | 2.7011 | 5 | 248.000000 | — | 259.70 |
| Weighted swamp and sand · seed 4205 | A* | 788 | 3.3418 | 147 | 90.000000 | 49.10% | 330.77 |
| Weighted swamp and sand · seed 4205 | UCS | 1,548 | 5.3898 | 69 | 90.000000 | — | 525.16 |
| U-shaped concave trap | A* | 290 | 1.3036 | 41 | 38.000000 | 67.88% | 214.61 |
| U-shaped concave trap | UCS | 903 | 3.0056 | 42 | 38.000000 | — | 318.96 |
