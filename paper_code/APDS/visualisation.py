"""
visualization.py

Post-hoc interpretability and visualization for Choquistic Regression
(Shapley representation, k-additive = 2).

This script extracts:
- Main effect coefficients
- Pairwise interaction coefficients
- Fold-averaged coefficients
- Stability-masked interaction heatmaps
from trained Choquistic Regression models.
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


# ==========================================================
# --- MODEL LOADING ---
# ==========================================================
def load_choquistic_models(model_dir):
    """Load all Choquistic Regression models from a directory."""
    return glob.glob(os.path.join(model_dir, "Choquistic_Shapley_fold*.joblib"))


# ==========================================================
# --- COEFFICIENT EXTRACTION ---
# ==========================================================
def extract_coefficients(model_files, feature_names):
    """
    Extract main effects and pairwise interactions from Choquistic models.

    Returns:
        all_coef_df : pd.DataFrame : all coefficients (main + interactions) across folds
        main_effects_df : pd.DataFrame : main-effect coefficients across folds
        all_interactions : list[pd.DataFrame] : interaction matrices per fold
    """
    all_coef_dfs = []
    all_main_effects = []
    all_interactions = []

    n_features = len(feature_names)

    # Expanded feature names (main + pairwise interactions)
    expanded_feature_names = feature_names.copy()
    for i, j in combinations(range(n_features), 2):
        expanded_feature_names.append(f"{feature_names[i]} × {feature_names[j]}")

    for file in model_files:
        pipeline = joblib.load(file)
        clf = pipeline.named_steps["clf"]

        coefs = clf.model_.coef_[0]
        fold_id = file.split("_fold")[-1].replace(".joblib", "")

        # All coefficients
        coef_df = pd.DataFrame({
            "feature": expanded_feature_names,
            "coefficient": coefs,
            "fold": fold_id,
        })
        all_coef_dfs.append(coef_df)

        # Main effects
        main_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs[:n_features],
            "fold": fold_id,
        })
        all_main_effects.append(main_df)

        # Pairwise interactions
        interaction_matrix = np.zeros((n_features, n_features))
        idx = n_features
        for i, j in combinations(range(n_features), 2):
            interaction_matrix[i, j] = coefs[idx]
            interaction_matrix[j, i] = coefs[idx]
            idx += 1

        interaction_df = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names)
        interaction_df["fold"] = fold_id
        all_interactions.append(interaction_df)

    return pd.concat(all_coef_dfs, ignore_index=True), pd.concat(all_main_effects, ignore_index=True), all_interactions


# ==========================================================
# --- AGGREGATION ---
# ==========================================================
def aggregate_across_folds(all_coef_df, main_effects_df, all_interactions):
    """Compute mean coefficients and mean interaction matrix across folds."""
    mean_coef_df = all_coef_df.groupby("feature")["coefficient"].mean().reset_index()
    mean_main_df = main_effects_df.groupby("feature")["coefficient"].mean().reset_index()

    interaction_stack = np.stack([df.drop(columns="fold").values for df in all_interactions])
    mean_interaction_matrix = interaction_stack.mean(axis=0)
    mean_interaction_df = pd.DataFrame(mean_interaction_matrix, index=all_interactions[0].index, columns=all_interactions[0].columns)

    return mean_coef_df, mean_main_df, mean_interaction_df


# ==========================================================
# --- VISUALIZATION ---
# ==========================================================
def plot_interaction_heatmap(mean_interaction_df, all_interactions, top_n=30, stability_threshold=0.8, figsize=(14,12)):
    """
    Plot a stability-masked heatmap for top phenotype interactions.

    Parameters
    ----------
    mean_interaction_df : pd.DataFrame
        Mean interaction matrix across folds.
    all_interactions : list[pd.DataFrame]
        List of fold-wise interaction matrices.
    top_n : int
        Number of phenotypes to include.
    stability_threshold : float
        Fraction of folds in which interaction must appear.
    """
    # Select top phenotypes by total interaction strength
    phenotype_strength = mean_interaction_df.abs().sum(axis=1)
    top_features = phenotype_strength.sort_values(ascending=False).head(top_n).index

    # Subset interaction matrices
    subset_interactions = [df.drop(columns="fold").loc[top_features, top_features] for df in all_interactions]
    mean_subset = np.stack([df.values for df in subset_interactions]).mean(axis=0)
    mean_subset_df = pd.DataFrame(mean_subset, index=top_features, columns=top_features)

    # Compute stability mask
    stack = np.stack([(df.values != 0).astype(int) for df in subset_interactions])
    stability = stack.mean(axis=0)
    mask = stability < stability_threshold
    np.fill_diagonal(mask, True)

    # Reorder by overall strength
    order = mean_subset_df.abs().sum(axis=1).sort_values(ascending=False).index
    mean_subset_df = mean_subset_df.loc[order, order]
    mask = mask.loc[order, order]

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(mean_subset_df, mask=mask, cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"label": "Mean Interaction Strength"})
    plt.title(f"Choquistic Regression Interactions\nTop {top_n} Phenotypes (stability ≥ {int(stability_threshold*100)}%)")
    plt.xlabel("Phenotypes")
    plt.ylabel("Phenotypes")
    plt.tight_layout()
    plt.show()


# ==========================================================
# --- SAVE OUTPUTS ---
# ==========================================================
def save_visualization_outputs(output_dir, all_coef_df, mean_coef_df, mean_main_df, mean_interaction_df):
    """Save coefficients and interaction matrices to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    all_coef_df.to_csv(os.path.join(output_dir, "shapley_coefficients_all_folds.csv"), index=False)
    mean_coef_df.to_csv(os.path.join(output_dir, "shapley_coefficients_mean.csv"), index=False)
    mean_main_df.to_csv(os.path.join(output_dir, "mean_main_effects.csv"), index=False)
    mean_interaction_df.to_csv(os.path.join(output_dir, "mean_interaction_matrix.csv"))


# ==========================================================
# --- EXAMPLE USAGE ---
# ==========================================================
if __name__ == "__main__":
    # Example: load features and models
    # Replace these paths with your project-specific paths
    DATA_PATH = "data/data_apds.csv" #path to the data
    MODEL_DIR = "results/..." #path to joblib file containing the model 
    OUTPUT_DIR = "results/.." #path to save the coeffients and visualisation plots

    import pandas as pd

    # Example: load feature names from preprocessed CSV
    df = pd.read_csv(DATA_PATH)
    feature_names = df["concept_str_found"].unique().tolist()

    # Load trained Choquistic models
    model_files = load_choquistic_models(MODEL_DIR)

    # Extract coefficients and interactions
    all_coef_df, main_df, all_interactions = extract_coefficients(model_files, feature_names)

    # Aggregate fold-wise results
    mean_coef_df, mean_main_df, mean_interaction_df = aggregate_across_folds(all_coef_df, main_df, all_interactions)

    # Save CSV outputs
    save_visualization_outputs(OUTPUT_DIR, all_coef_df, mean_coef_df, mean_main_df, mean_interaction_df)

    # Plot top interactions
    plot_interaction_heatmap(mean_interaction_df, all_interactions, top_n=30, stability_threshold=0.8)

    print(f"\n✅ Visualization completed. Outputs saved in '{OUTPUT_DIR}'")
