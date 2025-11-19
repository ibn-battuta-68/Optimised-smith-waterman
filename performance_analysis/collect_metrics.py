#!/usr/bin/env python3
"""
Performance Metrics Collection for Smith-Waterman Optimization
Collects runtime, GFLOPS, speedup, scaling, and hardware counters
Generates formatted tables for analysis
"""

import subprocess
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Container for performance measurements"""
    sequence_length: int
    num_threads: int
    runtime: float
    gflops: float = 0.0
    speedup: float = 0.0
    l1_misses: int = 0
    l2_misses: int = 0
    l3_misses: int = 0
    instructions: int = 0
    cycles: int = 0
    ipc: float = 0.0
    bandwidth_gbps: float = 0.0


class SmithWatermanBenchmark:
    """Benchmark suite for Smith-Waterman implementations"""
    
    def __init__(self, baseline_py: str, optimized_exe: str, output_file: str = "performance_metrics.txt"):
        self.baseline_py = baseline_py
        self.optimized_exe = optimized_exe
        self.output_file = output_file
        self.baseline_times: Dict[int, float] = {}
        self.metrics: List[PerformanceMetrics] = []
    
    def run_python_baseline(self, py_path: str, seq_length: int) -> Optional[float]:
        """Run Python baseline script and extract runtime"""
        cmd = ['python3', py_path, str(seq_length)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            output = result.stdout
            
            # Extract execution time
            match = re.search(r'Execution time:\s+([\d.]+)\s+seconds', output)
            if match:
                return float(match.group(1))
            else:
                print(f"Warning: Could not parse output from {' '.join(cmd)}")
                return None
        except subprocess.TimeoutExpired:
            print(f"Timeout for {' '.join(cmd)}")
            return None
        except Exception as e:
            print(f"Error running {' '.join(cmd)}: {e}")
            return None
    
    def run_executable(self, exe_path: str, seq_length: int, num_threads: Optional[int] = None) -> Optional[float]:
        """Run executable and extract runtime"""
        cmd = [exe_path, str(seq_length)]
        if num_threads is not None:
            cmd.append(str(num_threads))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            output = result.stdout
            
            # Extract execution time
            match = re.search(r'Execution time:\s+([\d.]+)\s+seconds', output)
            if match:
                return float(match.group(1))
            else:
                print(f"Warning: Could not parse output from {' '.join(cmd)}")
                return None
        except subprocess.TimeoutExpired:
            print(f"Timeout for {' '.join(cmd)}")
            return None
        except Exception as e:
            print(f"Error running {' '.join(cmd)}: {e}")
            return None
    
    def run_with_perf(self, exe_path: str, seq_length: int, num_threads: Optional[int] = None) -> Tuple[Optional[float], Dict]:
        """Run with perf stat to collect hardware counters"""
        cmd = [exe_path, str(seq_length)]
        if num_threads is not None:
            cmd.append(str(num_threads))
        
        perf_cmd = [
            'perf', 'stat', 
            '-e', 'cycles,instructions,L1-dcache-load-misses,L1-icache-load-misses,LLC-load-misses,cache-misses',
            '--'
        ] + cmd
        
        counters = {}
        runtime = None
        
        try:
            result = subprocess.run(perf_cmd, capture_output=True, text=True, timeout=600)
            
            # Extract runtime from stdout
            match = re.search(r'Execution time:\s+([\d.]+)\s+seconds', result.stdout)
            if match:
                runtime = float(match.group(1))
            
            # Parse perf output from stderr
            perf_output = result.stderr
            
            # Extract cycles
            cycles_match = re.search(r'([\d,]+)\s+cycles', perf_output)
            if cycles_match:
                counters['cycles'] = int(cycles_match.group(1).replace(',', ''))
            
            # Extract instructions
            inst_match = re.search(r'([\d,]+)\s+instructions', perf_output)
            if inst_match:
                counters['instructions'] = int(inst_match.group(1).replace(',', ''))
            
            # Extract L1 dcache misses
            l1d_match = re.search(r'([\d,]+)\s+L1-dcache-load-misses', perf_output)
            if l1d_match:
                counters['l1_misses'] = int(l1d_match.group(1).replace(',', ''))
            
            # Extract LLC (L3) misses
            l3_match = re.search(r'([\d,]+)\s+(?:LLC-load-misses|cache-misses)', perf_output)
            if l3_match:
                counters['l3_misses'] = int(l3_match.group(1).replace(',', ''))
            
            # Calculate IPC
            if 'cycles' in counters and 'instructions' in counters and counters['cycles'] > 0:
                counters['ipc'] = counters['instructions'] / counters['cycles']
            
        except subprocess.TimeoutExpired:
            print(f"Timeout for perf command")
        except FileNotFoundError:
            print("Warning: 'perf' command not found. Hardware counters will not be collected.")
            print("Install perf with: sudo apt-get install linux-tools-common linux-tools-generic")
            # Fallback to regular execution
            runtime = self.run_executable(exe_path, seq_length, num_threads)
        except Exception as e:
            print(f"Error running perf: {e}")
            runtime = self.run_executable(exe_path, seq_length, num_threads)
        
        return runtime, counters
    
    def calculate_gflops(self, seq_length: int, runtime: float) -> float:
        """
        Calculate GFLOPS for Smith-Waterman
        Approximate operations: N^2 * ~10 ops per cell (comparisons, additions, max operations)
        """
        if runtime <= 0:
            return 0.0
        ops_per_cell = 10  # Approximate: match/mismatch calc, 3 additions, 3 comparisons, 1 max
        total_ops = seq_length * seq_length * ops_per_cell
        gflops = (total_ops / runtime) / 1e9
        return gflops
    
    def collect_baseline_times(self, n_values: List[int]):
        """Collect baseline times from Python implementation"""
        print("=" * 80)
        print("COLLECTING BASELINE TIMES (Python implementation)")
        print("=" * 80)
        
        for n in n_values:
            print(f"Running Python baseline with N={n}...")
            runtime = self.run_python_baseline(self.baseline_py, n)
            if runtime:
                self.baseline_times[n] = runtime
                print(f"  Runtime: {runtime:.4f} seconds")
            else:
                print(f"  Failed to get runtime")
        print()
    
    def collect_scaling_metrics(self, n_values: List[int], thread_values: List[int], use_perf: bool = True):
        """Collect metrics for different N and thread counts"""
        print("=" * 80)
        print("COLLECTING SCALING METRICS")
        print("=" * 80)
        
        for n in n_values:
            for threads in thread_values:
                print(f"Running optimized: N={n}, threads={threads}...")
                
                if use_perf:
                    runtime, counters = self.run_with_perf(self.optimized_exe, n, threads)
                else:
                    runtime = self.run_executable(self.optimized_exe, n, threads)
                    counters = {}
                
                if runtime:
                    metrics = PerformanceMetrics(
                        sequence_length=n,
                        num_threads=threads,
                        runtime=runtime
                    )
                    
                    # Calculate GFLOPS
                    metrics.gflops = self.calculate_gflops(n, runtime)
                    
                    # Calculate speedup vs baseline
                    if n in self.baseline_times:
                        metrics.speedup = self.baseline_times[n] / runtime
                    
                    # Add hardware counters if available
                    metrics.l1_misses = counters.get('l1_misses', 0)
                    metrics.l3_misses = counters.get('l3_misses', 0)
                    metrics.instructions = counters.get('instructions', 0)
                    metrics.cycles = counters.get('cycles', 0)
                    metrics.ipc = counters.get('ipc', 0.0)
                    
                    self.metrics.append(metrics)
                    
                    print(f"  Runtime: {runtime:.4f}s, GFLOPS: {metrics.gflops:.2f}, Speedup: {metrics.speedup:.2f}x")
                else:
                    print(f"  Failed to get runtime")
        print()
    
    def generate_report(self):
        """Generate formatted performance report"""
        with open(self.output_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("SMITH-WATERMAN PERFORMANCE ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            
            # Baseline times table
            f.write("1. BASELINE PERFORMANCE (Python Implementation)\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Sequence Length (N)':<20} {'Runtime (s)':<20} {'GFLOPS':<20}\n")
            f.write("-" * 100 + "\n")
            for n in sorted(self.baseline_times.keys()):
                runtime = self.baseline_times[n]
                gflops = self.calculate_gflops(n, runtime)
                f.write(f"{n:<20} {runtime:<20.6f} {gflops:<20.2f}\n")
            f.write("\n\n")
            
            # Runtime and GFLOPS table
            f.write("2. OPTIMIZED PERFORMANCE - RUNTIME AND THROUGHPUT\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'N':<10} {'Threads':<10} {'Runtime (s)':<15} {'GFLOPS':<15} {'Speedup vs Python':<20}\n")
            f.write("-" * 100 + "\n")
            for m in sorted(self.metrics, key=lambda x: (x.sequence_length, x.num_threads)):
                f.write(f"{m.sequence_length:<10} {m.num_threads:<10} {m.runtime:<15.6f} "
                       f"{m.gflops:<15.2f} {m.speedup:<20.2f}x\n")
            f.write("\n\n")
            
            # Thread scaling table (fixed N)
            f.write("3. THREAD SCALING ANALYSIS\n")
            f.write("-" * 100 + "\n")
            
            # Group by N
            n_groups = {}
            for m in self.metrics:
                if m.sequence_length not in n_groups:
                    n_groups[m.sequence_length] = []
                n_groups[m.sequence_length].append(m)
            
            for n in sorted(n_groups.keys()):
                f.write(f"\nSequence Length N = {n}\n")
                f.write(f"{'Threads':<12} {'Runtime (s)':<15} {'Speedup vs Python':<20} {'Efficiency (%)':<15}\n")
                f.write("-" * 70 + "\n")
                
                sorted_metrics = sorted(n_groups[n], key=lambda x: x.num_threads)
                for m in sorted_metrics:
                    efficiency = (m.speedup / m.num_threads * 100) if m.num_threads > 0 else 0
                    f.write(f"{m.num_threads:<12} {m.runtime:<15.6f} {m.speedup:<12.2f}x {efficiency:<15.1f}\n")
            f.write("\n\n")
            
            # Hardware counters table
            if any(m.cycles > 0 or m.l1_misses > 0 for m in self.metrics):
                f.write("4. HARDWARE PERFORMANCE COUNTERS\n")
                f.write("-" * 120 + "\n")
                f.write(f"{'N':<10} {'Threads':<10} {'L1 Misses':<15} {'L3 Misses':<15} "
                       f"{'Instructions':<18} {'Cycles':<18} {'IPC':<10}\n")
                f.write("-" * 120 + "\n")
                
                for m in sorted(self.metrics, key=lambda x: (x.sequence_length, x.num_threads)):
                    if m.cycles > 0 or m.l1_misses > 0:
                        f.write(f"{m.sequence_length:<10} {m.num_threads:<10} "
                               f"{m.l1_misses:<15,} {m.l3_misses:<15,} "
                               f"{m.instructions:<18,} {m.cycles:<18,} {m.ipc:<10.3f}\n")
                f.write("\n\n")
            else:
                f.write("4. HARDWARE PERFORMANCE COUNTERS\n")
                f.write("-" * 100 + "\n")
                f.write("Hardware counters not available (perf not installed or insufficient permissions)\n")
                f.write("To enable: sudo apt-get install linux-tools-common linux-tools-generic\n")
                f.write("And run with: sudo -E python3 collect_metrics.py\n")
                f.write("\n\n")
            
            # Summary statistics
            f.write("5. SUMMARY STATISTICS\n")
            f.write("-" * 100 + "\n")
            
            if self.metrics:
                max_speedup_metric = max(self.metrics, key=lambda x: x.speedup if x.speedup > 0 else 0)
                max_speedup = max_speedup_metric.speedup
                max_gflops = max(m.gflops for m in self.metrics)
                best_gflops_metric = max(self.metrics, key=lambda x: x.gflops)
                
                f.write(f"Maximum Speedup vs Python Baseline: {max_speedup:.2f}x\n")
                f.write(f"  Configuration: N={max_speedup_metric.sequence_length}, threads={max_speedup_metric.num_threads}\n")
                f.write(f"  Runtime: {max_speedup_metric.runtime:.6f} seconds\n")
                if max_speedup_metric.sequence_length in self.baseline_times:
                    f.write(f"  Python Baseline Runtime: {self.baseline_times[max_speedup_metric.sequence_length]:.6f} seconds\n")
                f.write(f"\n")
                f.write(f"Maximum GFLOPS: {max_gflops:.2f} (N={best_gflops_metric.sequence_length}, "
                       f"threads={best_gflops_metric.num_threads})\n")
                
                if any(m.ipc > 0 for m in self.metrics):
                    avg_ipc = sum(m.ipc for m in self.metrics if m.ipc > 0) / len([m for m in self.metrics if m.ipc > 0])
                    f.write(f"Average IPC: {avg_ipc:.3f}\n")
            
            f.write("\n" + "=" * 100 + "\n")
        
        print(f"\nPerformance report saved to: {self.output_file}")
        print(f"You can view it with: cat {self.output_file}")


def main():
    """Main execution"""
    # Configuration
    BASELINE_PY = "../baseline/sw_baseline.py"  # Python baseline
    OPTIMIZED_EXE = "../optimized/sw_opt"
    OUTPUT_FILE = "performance_metrics.txt"
    
    # Test configurations
    N_VALUES = [1000, 5000, 10000, 20000]
    THREAD_VALUES = [1, 2, 4, 8, 16]
    
    # Check if files exist
    if not Path(BASELINE_PY).exists():
        print(f"Error: Python baseline not found at {BASELINE_PY}")
        print("Please ensure the baseline script exists or adjust the path.")
        sys.exit(1)
    
    if not Path(OPTIMIZED_EXE).exists():
        print(f"Error: Optimized executable not found at {OPTIMIZED_EXE}")
        print("Please compile the code first or adjust the path.")
        sys.exit(1)
    
    # Check if perf is available
    try:
        subprocess.run(['perf', '--version'], capture_output=True, check=True)
        use_perf = True
        print("perf found - will collect hardware counters\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_perf = False
        print("Warning: perf not found - hardware counters will not be collected")
        print("To enable perf: sudo apt-get install linux-tools-common linux-tools-generic\n")
    
    # Run benchmarks
    benchmark = SmithWatermanBenchmark(BASELINE_PY, OPTIMIZED_EXE, OUTPUT_FILE)
    benchmark.collect_baseline_times(N_VALUES)
    benchmark.collect_scaling_metrics(N_VALUES, THREAD_VALUES, use_perf=use_perf)
    benchmark.generate_report()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_FILE}")
    print("\nTo view the report:")
    print(f"  cat {OUTPUT_FILE}")
    print(f"  less {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
