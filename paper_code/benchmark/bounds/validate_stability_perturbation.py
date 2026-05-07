import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
import json

# Adjust path to locate the core model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from core.models.regression import ChoquisticRegression
except ImportError:
    pass

RESULTS_FILE = 'stability_validation_results.json'

def save_all_results_json(all_results, filename=RESULTS_FILE):
    """Save all stability results to a single JSON file."""
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"  -> All results saved to {filename}")

def run_stability_for_k(k, n, N, C_values, repeats, seed):
    """
    Run stability test for a specific k value.
    
    Returns a dictionary with results for this k value.
    """
    print(f"\n--- Stability Test for k={k} (seed={seed}) ---")
    
    json_data = {
        'k': k,
        'seed': seed,
        'N': N,
        'n': n,
        'C_values': C_values,
        'results': []
    }
    plot_C, plot_diff, plot_err = [], [], []

    for C in C_values:
        print(f"  Testing C={C}...")
        diffs = []
        for rep in range(repeats):
            np.random.seed(seed + int(C * 1000) + rep)  # Deterministic seed for each iteration
            X = np.random.rand(N, n)
            y = np.random.randint(0, 2, N) 
            
            # Train Model 1
            m1 = ChoquisticRegression(k_add=k, penalty='l2', C=C, tol=1e-6)
            m1.fit(X, y)
            w1 = m1.model_.coef_.flatten()

            # Train Model 2 (1 flipped label)
            y_flip = y.copy()
            y_flip[0] = 1 - y_flip[0]
            m2 = ChoquisticRegression(k_add=k, penalty='l2', C=C, tol=1e-6)
            m2.fit(X, y_flip)
            w2 = m2.model_.coef_.flatten()
            
            diffs.append(np.linalg.norm(w1 - w2))
        
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        
        plot_C.append(C)
        plot_diff.append(mean_diff)
        plot_err.append(std_diff)
        
        json_data['results'].append({'C': C, 'mean': mean_diff, 'std': std_diff})

    return json_data, plot_C, plot_diff, plot_err

def plot_individual_stability(k, plot_C, plot_diff, plot_err):
    """Plot stability results for a single k value."""
    plt.figure(figsize=(8, 6))
    
    # 1. Empirical Data
    plt.errorbar(plot_C, plot_diff, yerr=plot_err, fmt='o', 
                 color='#6a0dad', ecolor='gray', capsize=4, 
                 label='Empirical Sensitivity')
    
    # 2. Linear Regime Fit (The "Actual Value" of the Linear Bound)
    # Fit line to the first few points (C <= 0.75) where behavior is strictly linear
    linear_points = min(5, len(plot_C))
    slope_emp = np.mean([plot_diff[i]/plot_C[i] for i in range(linear_points) if plot_C[i] > 0])
    
    x_range = np.linspace(0, max(plot_C)*1.05, 100)
    linear_fit = slope_emp * x_range
    
    plt.plot(x_range, linear_fit, 'k--', linewidth=1.5, 
             label=r'Theoretical Linear Trend ($\beta \propto C$)')

    plt.xlabel(r'Inverse Regularisation ($C$)', fontsize=12)
    plt.ylabel(r'Parameter Sensitivity $\|\Delta \theta\|_2$', fontsize=12)
    plt.title(rf'Stability: Linear Regime vs Saturation ($k={k}$)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    filename = f'stability_proof_k{k}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  Saved {filename}")

def plot_combined_stability(all_k_results):
    """Plot all k values in a single combined plot."""
    plt.figure(figsize=(10, 7))
    
    # Color palette for different k values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_k_results)))
    
    for idx, (k, data) in enumerate(sorted(all_k_results.items())):
        plot_C = [r['C'] for r in data['results']]
        plot_diff = [r['mean'] for r in data['results']]
        plot_err = [r['std'] for r in data['results']]
        
        plt.errorbar(plot_C, plot_diff, yerr=plot_err, fmt='o-', 
                     color=colors[idx], ecolor=colors[idx], capsize=3, 
                     alpha=0.8, linewidth=1.5, markersize=5,
                     label=f'$k={k}$')
    
    plt.xlabel(r'Inverse Regularisation ($C$)', fontsize=12)
    plt.ylabel(r'Parameter Sensitivity $\|\Delta \theta\|_2$', fontsize=12)
    plt.title(r'Stability Analysis: All k-Additivity Levels', fontsize=14)
    plt.legend(loc='upper left', title='k-additivity', ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    filename = 'stability_proof_all_k.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\nSaved combined plot: {filename}")

def run_stability(seed=42):
    np.random.seed(seed)
    n = 10  # number of features
    N = 100
    # Range covering linear and saturation
    C_values = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5] 
    repeats = 40
    
    # All available k values (from 1 to n)
    k_values = list(range(1, n + 1))
    
    print(f"=== Stability Test for All k Values (k=1 to {n}) ===")
    print(f"Parameters: N={N}, n={n}, repeats={repeats}, seed={seed}")
    print(f"C values: {C_values}")
    print(f"k values: {k_values}")
    
    # Master dictionary to store all results
    all_results = {
        'description': "Stability test measuring parameter sensitivity to single label perturbation for all k-additivity levels.",
        'seed': seed,
        'N': N,
        'n': n,
        'C_values': C_values,
        'repeats': repeats,
        'k_values': k_values,
        'results_by_k': {}
    }
    
    # Dictionary to store plotting data for combined plot
    all_k_plot_data = {}
    
    for k in k_values:
        json_data, plot_C, plot_diff, plot_err = run_stability_for_k(
            k=k, n=n, N=N, C_values=C_values, repeats=repeats, seed=seed
        )
        
        # Store results
        all_results['results_by_k'][str(k)] = json_data
        all_k_plot_data[k] = json_data
        
        # Create individual plot for this k
        plot_individual_stability(k, plot_C, plot_diff, plot_err)
    
    # Create combined plot with all k values
    plot_combined_stability(all_k_plot_data)
    
    # Save all results to a single JSON file
    save_all_results_json(all_results)
    
    print("\n=== Stability Analysis Complete ===")
    print(f"Individual plots saved: stability_proof_k1.png to stability_proof_k{n}.png")
    print(f"Combined plot saved: stability_proof_all_k.png")
    print(f"Results saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    run_stability()