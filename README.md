# Optimised Smith–Waterman

Accelerating DNA local alignment with **AVX2 SIMD, OpenMP multithreading, and cache-aware wavefront tiling**.

Achieved **224.9× speedup over the Python baseline**, scoring two 20,000-base sequences in **0.359 seconds using 8 threads** in recorded benchmarks.

- Python and C baselines alongside an optimised C implementation.
- Benchmark automation, scalability plots, and performance analysis.
- Alternative single-threaded implementation with O(N) memory.

### Build & Run

Requires an AVX2-capable CPU and GCC with OpenMP support.

```bash
gcc -O3 -mavx2 -fopenmp optimized/sw_opt.c -o optimized/sw_opt
./optimized/sw_opt 20000 8
```

See [the report](report.md) for optimisation details and experimental results.
