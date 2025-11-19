import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

def run_sw_opt(sequence_length, num_threads=None):
    if num_threads is None:
        cmd = ['../optimized/sw_opt', str(sequence_length)]
    else:
        cmd = ['../optimized/sw_opt', str(sequence_length), str(num_threads)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
        match = re.search(r'Execution time:\s+([\d.]+)\s+seconds', output)
        if match:
            return float(match.group(1))
        else:
            print(f"Could not parse output: {output}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Timeout for N={sequence_length}, threads={num_threads}")
        return None
    except Exception as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        return None

def plot_runtime_vs_n(n_values, output_file='runtime_vs_n.png'):
    print("Generating runtime vs N plot...")
    runtimes = []
    
    for n in n_values:
        print(f"  Running with N={n}...")
        runtime = run_sw_opt(n)
        if runtime is not None:
            runtimes.append(runtime)
        else:
            runtimes.append(np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, runtimes, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length (N)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Smith-Waterman Runtime vs Sequence Length', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    plt.close()

def plot_runtime_vs_threads(n, thread_values, output_file='runtime_vs_threads.png'):
    print(f"Generating runtime vs threads plot (N={n})...")
    runtimes = []
    
    for threads in thread_values:
        print(f"  Running with {threads} threads...")
        runtime = run_sw_opt(n, threads)
        if runtime is not None:
            runtimes.append(runtime)
        else:
            runtimes.append(np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thread_values, runtimes, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title(f'Smith-Waterman Runtime vs Number of Threads (N={n})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_values)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    plt.close()

def plot_scalability_different_threads(n_values, thread_values, output_file='scalability.png'):
    print("Generating scalability plot (runtime vs N for different thread counts)...")
    
    results = {}
    for threads in thread_values:
        results[threads] = []
        for n in n_values:
            print(f"  Running with N={n}, threads={threads}...")
            runtime = run_sw_opt(n, threads)
            if runtime is not None:
                results[threads].append(runtime)
            else:
                results[threads].append(np.nan)
    
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(thread_values)))
    
    for i, threads in enumerate(thread_values):
        plt.plot(n_values, results[threads], 'o-', linewidth=2, markersize=6, 
                 label=f'{threads} thread{"s" if threads > 1 else ""}', color=colors[i])
    
    plt.xlabel('Sequence Length (N)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Smith-Waterman Scalability: Runtime vs Problem Size for Different Thread Counts', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    plt.close()

def plot_parallel_overhead(n, thread_values, output_file='parallel_overhead.png'):
    print(f"Generating parallel overhead plot (N={n})...")
    runtimes = []
    
    for threads in thread_values:
        print(f"  Running with {threads} threads...")
        runtime = run_sw_opt(n, threads)
        if runtime is not None:
            runtimes.append(runtime)
        else:
            runtimes.append(np.nan)
    
    baseline_time = runtimes[0]
    overheads = [(rt * threads / baseline_time - 1) * 100 if not np.isnan(rt) else np.nan 
                 for rt, threads in zip(runtimes, thread_values)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thread_values, overheads, 'o-', linewidth=2, markersize=8, color='red')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='No overhead (ideal)')
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Parallel Overhead (%)', fontsize=12)
    plt.title(f'Parallel Overhead vs Number of Threads (N={n})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_values)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    plt.close()

if __name__ == '__main__':
    n_values = [1000, 2000, 5000, 10000, 20000, 30000, 40000]
    
    fixed_n = 40000
    thread_values = [1, 2, 4, 6, 8, 12, 16]
    
    n_values_scalability = [1000, 5000, 10000, 20000, 30000, 40000]
    thread_values_scalability = [1, 2, 4, 8, 16]
    
    plot_runtime_vs_n(n_values, 'runtime_vs_n.png')
    plot_runtime_vs_threads(fixed_n, thread_values, 'runtime_vs_threads.png')
    plot_scalability_different_threads(n_values_scalability, thread_values_scalability, 'scalability.png')
    plot_parallel_overhead(fixed_n, thread_values, 'parallel_overhead.png')
    
    print("\nAll plots generated successfully!")
