import os
os.environ["SCIPY_ARRAY_API"] = "1"

RANDOM_STATE = 42

import numpy as np
import random

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import warnings
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from core.models.choquet import (nParam_kAdd,
                                choquet_k_additive_game, 
                                choquet_k_additive_mobius,
                                choquet_k_additive_shapley
) 
from utils.data_loader import func_read_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore")

def get_project_root():
    """Returns the project root directory."""
    # Assuming this script is in paper_code/benchmark/noise_robustness
    # Root is ../../../
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def run_analysis_for_dataset(
    dataset, 
    representation="shapley",
    regularization='l2',
    random_state=42
):    
    """
    Analyzes a single dataset for noise robustness and bootstrap stability.
    """
    # 1. Setup paths
    root_dir = get_project_root()
    
    # Determine dataset name and data
    if isinstance(dataset, (list, tuple)) and len(dataset) == 3:
        dataset_name, X, y = dataset
    else:
        dataset_name = dataset
        X, y = func_read_data(dataset_name)

    print(f"\nProcessing: {dataset_name} ({representation}, {regularization})")
    
    # Create dataset-specific subdirectories with regularization subfolders
    reg_folder = regularization if regularization != 'none' else 'None'
    dataset_results_dir = os.path.join(root_dir, 'results', 'benchmark', dataset_name, reg_folder)
    
    os.makedirs(dataset_results_dir, exist_ok=True)

    # 2. Prepare Data
    nSamp, nAttr = X.shape
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    # Convert to numpy arrays (matching original script)
    X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_values = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_values)
    X_test_scaled = scaler.transform(X_test_values)

    # Select transform
    if representation == "game": transform_func = choquet_k_additive_game
    elif representation == "mobius": transform_func = choquet_k_additive_mobius
    elif representation == "shapley": transform_func = choquet_k_additive_shapley
    else: raise ValueError(f"Unknown representation: {representation}")

    # 3. Initialize Results
    columns = [
        'k_value', 'n_params', 'train_time', 'baseline_accuracy',
        'noise_0.1', 'noise_0.2', 'noise_0.3',
        'noise_0.1_std', 'noise_0.2_std', 'noise_0.3_std',
        'bootstrap_mean', 'bootstrap_std'
    ]
    results_df = pd.DataFrame(index=range(1, nAttr + 1), columns=columns)

    # 4. Main Loop over k
    for k in range(1, nAttr + 1):
        print(f"  > k={k}")
        results_df.loc[k, 'k_value'] = k
        results_df.loc[k, 'n_params'] = nParam_kAdd(k, nAttr)
        
        try:
            # Transform & Train
            start = time.time()
            X_tr_ch = transform_func(X_train_scaled, k_add=k)
            X_te_ch = transform_func(X_test_scaled, k_add=k)
            
            model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                solver="newton-cg" if regularization in ['l2', 'none'] else "saga",
                penalty=None if regularization == 'none' else regularization
            )
            model.fit(X_tr_ch, y_train_values)
            model.coef_ = model.coef_.astype(np.float64)
            model.intercept_ = model.intercept_.astype(np.float64)
            results_df.loc[k, 'train_time'] = time.time() - start
            
            # Baseline
            y_pred = model.predict(X_te_ch)
            baseline_acc = accuracy_score(y_test_values, y_pred)
            results_df.loc[k, 'baseline_accuracy'] = baseline_acc
            results_df.loc[k, 'noise_0.1_std'] = 0
            results_df.loc[k, 'noise_0.2_std'] = 0
            results_df.loc[k, 'noise_0.3_std'] = 0
            print(f"    Baseline accuracy: {baseline_acc:.4f}")
            
            # Noise robustness testing
            print("    Testing noise robustness...")
            noise_levels = [0.1, 0.2, 0.3]
            
            for noise_level in noise_levels:
                noise_accuracies = []
                
                # Run multiple noise tests and average results
                for _ in range(5):  # 5 repetitions for each noise level
                    # Apply Gaussian noise to test set (scale by data std)
                    feature_stds = np.std(X_test_scaled, axis=0, keepdims=True)
                    noise = np.random.normal(0, 1, X_test_scaled.shape) * noise_level * feature_stds
                    X_test_noisy = X_test_scaled.copy() + noise
                    
                    # Transform noisy data and evaluate
                    X_test_noisy_choquet = transform_func(X_test_noisy, k_add=k)
                    y_pred_noisy = model.predict(X_test_noisy_choquet)
                    noise_acc = accuracy_score(y_test_values, y_pred_noisy)
                    noise_accuracies.append(noise_acc)
                
                # Record average noise performance
                avg_noise_acc = np.mean(noise_accuracies)
                std_noise_acc = np.std(noise_accuracies)
                results_df.loc[k, f'noise_{noise_level}'] = avg_noise_acc
                results_df.loc[k, f'noise_{noise_level}_std'] = std_noise_acc
                print(f"      Noise level {noise_level}: accuracy = {avg_noise_acc:.4f}, std = {std_noise_acc:.4f}")

            # Bootstrap stability testing
            print("    Testing bootstrap stability...")
            bootstrap_accuracies = []
            n_bootstrap = 50  # Number of bootstrap samples
            
            for _ in range(n_bootstrap):
                # Create bootstrap sample
                indices = np.random.choice(len(X_test_scaled), size=int(0.8*len(X_test_scaled)), replace=True)
                X_boot = X_test_scaled[indices]
                y_boot = y_test_values[indices]
                
                # Transform bootstrap data and evaluate
                X_boot_choquet = transform_func(X_boot, k_add=k)
                y_pred_boot = model.predict(X_boot_choquet)
                boot_acc = accuracy_score(y_boot, y_pred_boot)
                bootstrap_accuracies.append(boot_acc)
            
            # Record bootstrap results
            bootstrap_mean = np.mean(bootstrap_accuracies)
            bootstrap_std = np.std(bootstrap_accuracies)
            results_df.loc[k, 'bootstrap_mean'] = bootstrap_mean
            results_df.loc[k, 'bootstrap_std'] = bootstrap_std
            print(f"      Bootstrap: mean = {bootstrap_mean:.4f}, std = {bootstrap_std:.4f}")

        except Exception as e:
            print(f"Error processing k={k}: {str(e)}")
            # Fill with NaN to indicate missing data
            for col in results_df.columns[2:]:  # Skip k_value and n_params
                results_df.loc[k, col] = np.nan

    # 5. Save Results & Plots
    
    # Save single CSV with both noise and bootstrap results
    csv_path = os.path.join(dataset_results_dir, "results.csv")
    results_df.to_csv(csv_path)
    
    # Generate Summary text
    summary_txt = f"FULL RESULTS TABLE:\n{results_df.to_string()}"
    with open(os.path.join(dataset_results_dir, "summary.txt"), 'w') as f: f.write(summary_txt)
    
    # Plot Noise Robustness
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, 3))
    for i, nl in enumerate([0.1, 0.2, 0.3]):
        plt.plot(results_df['k_value'], results_df[f'noise_{nl}'], 'o-', color=colors[i], linewidth=2, label=f'Noise level: {nl}')
    plt.plot(results_df['k_value'], results_df['baseline_accuracy'], 'k--', linewidth=2, label='Baseline (no noise)')
    plt.xlabel('k-additivity'); plt.ylabel('Accuracy'); plt.title(f'Noise Robustness vs k-additivity ({dataset_name})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xticks(range(1, nAttr + 1))  # Ensure integer k values on x-axis
    plt.savefig(os.path.join(dataset_results_dir, "noise_robustness.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Bootstrap Stability
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df['k_value'], results_df['bootstrap_mean'], yerr=results_df['bootstrap_std'], fmt='o-', capsize=5)
    plt.xlabel('k-additivity'); plt.ylabel('Bootstrap Accuracy (mean ± std)'); plt.title(f'Bootstrap Stability vs k-additivity ({dataset_name})')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, nAttr + 1))  # Ensure integer k values on x-axis
    plt.savefig(os.path.join(dataset_results_dir, "bootstrap_stability.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis completed. Results saved to: {dataset_results_dir}")
    return results_df

def generate_scaled_plots():
    """
    Generate scaled plots with consistent y-axis across all datasets for publication.
    Reads all existing results and regenerates plots with shared y-limits.
    """
    root_dir = get_project_root()
    benchmark_root = os.path.join(root_dir, 'results', 'benchmark')
    
    if not os.path.exists(benchmark_root):
        print(f"No results found in {benchmark_root}")
        return
    
    # Find all datasets (top-level folders in benchmark)
    datasets = [d for d in os.listdir(benchmark_root) if os.path.isdir(os.path.join(benchmark_root, d))]
    if not datasets:
        print("No dataset folders found.")
        return
    
    print(f"\nGenerating scaled plots for {len(datasets)} datasets...")
    
    # Collect all data to find global y-limits across all regularizations
    all_dfs = {}
    for dataset in datasets:
        dataset_path = os.path.join(benchmark_root, dataset)
        
        # Check all regularization subfolders
        for reg_folder in ['None', 'l1', 'l2']:
            reg_path = os.path.join(dataset_path, reg_folder)
            csv_path = os.path.join(reg_path, 'results.csv')
            if os.path.exists(csv_path):
                key = f"{dataset}_{reg_folder}"
                all_dfs[key] = pd.read_csv(csv_path, index_col=0)
    
    if not all_dfs:
        print("No results CSV files found.")
        return
    
    # Calculate global y-limits
    min_acc, max_acc = 1.0, 0.0
    for df in all_dfs.values():
        # Check all accuracy-related columns
        cols = ['baseline_accuracy', 'noise_0.1', 'noise_0.2', 'noise_0.3', 'bootstrap_mean']
        for col in cols:
            if col in df.columns:
                if col == 'bootstrap_mean':
                    # Include std margin for bootstrap
                    std_col = 'bootstrap_std'
                    if std_col in df.columns:
                        min_val = (df[col] - df[std_col]).min()
                        max_val = (df[col] + df[std_col]).max()
                    else:
                        min_val = df[col].min()
                        max_val = df[col].max()
                else:
                    min_val = df[col].min()
                    max_val = df[col].max()
                
                if not np.isnan(min_val): min_acc = min(min_acc, min_val)
                if not np.isnan(max_val): max_acc = max(max_acc, max_val)
    
    y_margin = (max_acc - min_acc) * 0.05
    y_lims = (max(0, min_acc - y_margin), min(1, max_acc + y_margin))
    
    print(f"  Using shared y-axis: [{y_lims[0]:.3f}, {y_lims[1]:.3f}]")
    
    # Regenerate plots for each dataset and regularization with shared y-limits
    for key, df in all_dfs.items():
        parts = key.rsplit('_', 1)  # Split from right: dataset_reg
        dataset = parts[0]
        reg_folder = parts[1]
        
        results_path = os.path.join(benchmark_root, dataset, reg_folder)
        
        # Noise plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, 3))
        for i, nl in enumerate([0.1, 0.2, 0.3]):
            plt.plot(df['k_value'], df[f'noise_{nl}'], 'o-', color=colors[i], linewidth=2, label=f'Noise level: {nl}')
        plt.plot(df['k_value'], df['baseline_accuracy'], 'k--', linewidth=2, label='Baseline (no noise)')
        plt.xlabel('k-additivity'); plt.ylabel('Accuracy')
        plt.title(f'Noise Robustness vs k-additivity ({dataset}, {reg_folder})')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.ylim(y_lims)
        k_values = df['k_value'].dropna().astype(int)
        plt.xticks(range(int(k_values.min()), int(k_values.max()) + 1))  # Ensure integer k values on x-axis
        out_path = os.path.join(results_path, 'noise_robustness_scaled.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bootstrap plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['k_value'], df['bootstrap_mean'], yerr=df['bootstrap_std'], fmt='o-', capsize=5)
        plt.xlabel('k-additivity'); plt.ylabel('Bootstrap Accuracy (mean ± std)')
        plt.title(f'Bootstrap Stability vs k-additivity ({dataset}, {reg_folder})')
        plt.grid(True, alpha=0.3)
        plt.ylim(y_lims)
        k_values = df['k_value'].dropna().astype(int)
        plt.xticks(range(int(k_values.min()), int(k_values.max()) + 1))  # Ensure integer k values on x-axis
        out_path = os.path.join(results_path, 'bootstrap_stability_scaled.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Scaled plots saved for all {len(all_dfs)} result sets")

if __name__ == "__main__":
    # Load custom dataset if needed
    try:
        base_X, base_y = func_read_data("pure_pairwise_interaction")
        ppi_dataset = ("pure_pairwise_interaction", base_X, base_y)
    except: ppi_dataset = None

    # All available datasets
    datasets = ['dados_covid_sbpo_atual', 'banknotes', 'mammographic', 'diabetes', 'skin']
    if ppi_dataset: datasets.append(ppi_dataset)

    # All regularization types
    regularizations = ['none', 'l1', 'l2']
    
    # Run analysis for all combinations
    for ds in datasets:
        for reg in regularizations:
            run_analysis_for_dataset(ds, representation="shapley", regularization=reg)
    
    # After all datasets are processed, generate scaled plots for publication
    print("\n" + "="*80)
    generate_scaled_plots()
    print("="*80)
    print("\nAll analyses complete!")
