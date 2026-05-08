# Shapley Regression: Game-theoretic Extensions of Logistic Regression

This repository provides a general-purpose implementation of **Shapley Regression**, a game-theoretic extension of logistic regression based on the Choquet integral. The framework is designed to model non-linear feature interactions for various machine learning tasks while preserving the interpretability, efficiency, and statistical grounding of classical logistic regression.

Additionally, this repository contains the code, experiments, and implementation details for the IJCAI-2026 paper **"Shapley Regression for Rare Disease Diagnosis Support: a case study on APDS"**.

## Project Overview

Traditional logistic regression assumes additive feature effects, limiting its ability to capture complex relationships between variables. This project explicitly models these feature interactions through the Choquet integral and cooperative game theory. 

By replacing the standard linear predictor with a k-additive cooperative game formulation, Shapley Regression provides a powerful general tool that enables:
- Explicit modeling of pairwise or higher-order feature interactions
- Decomposition of predictions into individual and interaction contributions
- Preservation of convex optimization properties
- Controllable model complexity through the choice of k and regularization strategies

### IJCAI-2026 Case Study: Rare Disease Diagnosis (APDS)

While broadly applicable, the framework is notably effective for biomedical and rare disease applications. In rare diseases such as Activated PI3K Delta Syndrome (**APDS**), diagnosis often depends not on isolated phenotypes, but on combinations of interacting clinical manifestations. 

In such settings, datasets are often small, imbalanced, heterogeneous, and partially incomplete. Conventional linear models may overlook important interaction patterns, while deep learning methods can suffer from limited interpretability. Shapley Regression addresses this gap by providing an interaction-aware yet interpretable modeling framework capable of supporting rare disease screening, phenotype analysis, and hypothesis generation from electronic health records.

This repository includes both theoretical and practical components for studying interaction-aware learning across different settings, including:
- Implementation of Shapley Regression models
- Optimization procedures for learning capacities
- Interaction analysis and visualization tools
- Evaluation pipelines for different classification tasks
- Experiments on phenotype-based representations for APDS diagnosis support

### Key Features

- **Multiple Representation Bases**: Implementation of different mathematical bases (Game, Mobius, and Shapley) that are linearly related but have distinct interpretability properties
- **K-additivity Analysis**: Tools for analyzing the impact of k-additivity on model performance, complexity, and interpretability
- **Robustness Testing**: Comprehensive framework for testing model robustness under various perturbations
- **Visualization Tools**: Specialized visualization functions for each representation basis

## Repository Structure

```
project/
├── core/                # Core implementation
│   ├── models/          # Model implementations
│   │   ├── choquet.py   # Choquet integral implementations
│   │   └── regression.py # ChoquisticRegression model
│   └── __init__.py
├── datasets/            # Data directory containing all datasets
├── examples/            # Example scripts demonstrating usage
├── paper_code/          # Code and scripts used for IJCAI-2026 paper results
├── paper_results/       # Results used for IJCAI-2026 paper
├── results/             # Generated simulation and benchmark results
├── tests/               # Testing suite
│   ├── complexity/      # Complexity testing
│   ├── k_additivity/    # K-additivity testing
│   ├── robustness/      # Robustness testing
│   ├── theory_bounds/   # Theory bounds testing
│   └── __init__.py
├── utils/               # Utility functions
│   ├── visualization/   # Visualization utilities
│   ├── data_loader.py   # Data loading utilities
│   └── __init__.py
├── LICENSE              # License file
├── README.md            # This file
├── requirements.txt     # Project dependencies
└── setup.py             # Package installation script
```

## Mathematical Background

The project is based on three different mathematical bases that are linearly related:

1. **Game Representation**: The traditional representation of fuzzy measures with the caveat of having fewer restrictions such as monotonicity.
2. **Möbius Representation**: An alternative representation that directly captures the interaction between features.
3. **Shapley Representation**: A representation that uses the Shapley value and the pairwise interaction indices between features.

Each representation has its own interpretability properties and is suitable for different types of analysis.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tomasbrogueira/shapley_regression.git
cd shapley_regression
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Install additional dependencies if needed:
```bash
pip install -r requirements.txt
```

### Option 2: Use Without Installation

If you prefer not to install the package, you can still run the scripts by adding the project root to your Python path:

1. Clone the repository:
```bash
git clone https://github.com/tomasbrogueira/shapley_regression.git
cd shapley_regression
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run scripts from the project root:
```bash
python -m examples.comparison_example
```

## Dataset Preparation

Before running the examples, you need to place your datasets in the `datasets/` directory. The data loader expects the following files:

- `data_banknotes.csv`: Banknote authentication dataset
- `transfusion.csv`: Blood Transfusion Service Center Data Set
- `data_mammographic.data`: Mammographic mass dataset
- `data_raisin.xlsx`: Raisin dataset
- `data_rice.xlsx`: Rice (Commeo and Osmancik) dataset
- `diabetes.csv`: Diabetes (PIMA) dataset
- `data_skin.csv`: Skin segmentation dataset
- `pure_pairwise_interaction_dataset.csv`: Pure pairwise interaction dataset

Copy these files from your original project to the `datasets/` directory.

## Usage

### Basic Example

```python
from core.models.regression import ChoquisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_loader import func_read_data

# Load data
X, y = func_read_data("banknotes")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model with game representation
model_game = ChoquisticRegression(
    representation="game",
    k_add=2,
    scale_data=True
)
model_game.fit(X_train, y_train)

# Create and train model with Mobius representation
model_mobius = ChoquisticRegression(
    representation="mobius",
    k_add=2,
    scale_data=True
)
model_mobius.fit(X_train, y_train)

# Create and train model with Shapley representation (k=2)
model_shapley = ChoquisticRegression(
    representation="shapley",
    k_add=2,
    scale_data=True
)
model_shapley.fit(X_train, y_train)

# Evaluate models
y_pred_game = model_game.predict(X_test)
y_pred_mobius = model_mobius.predict(X_test)
y_pred_shapley = model_shapley.predict(X_test)

print(f"Game representation accuracy: {accuracy_score(y_test, y_pred_game):.4f}")
print(f"Mobius representation accuracy: {accuracy_score(y_test, y_pred_mobius):.4f}")
print(f"Shapley representation accuracy: {accuracy_score(y_test, y_pred_shapley):.4f}")
```

### Running the Example Script

The repository includes an example script that demonstrates the use of different representations:

```bash
# If installed as a package (Option 1):
python examples/comparison_example.py

# If using without installation (Option 2):
python -m examples.comparison_example
```

### K-additivity Analysis

```python
from tests.k_additivity.k_additivity import direct_k_additivity_analysis, find_optimal_k
from utils.visualization.plotting import plot_k_additivity_results

# Analyze k-additivity impact with game representation
results_game = direct_k_additivity_analysis(
    dataset_name="banknotes",
    representation="game",
    output_dir="results/k_additivity/game"
)

# Analyze k-additivity impact with Mobius representation
results_mobius = direct_k_additivity_analysis(
    dataset_name="banknotes",
    representation="mobius",
    output_dir="results/k_additivity/mobius"
)

# Find optimal k value
optimal_k_game, scores_game = find_optimal_k(results_game)
optimal_k_mobius, scores_mobius = find_optimal_k(results_mobius)

print(f"Optimal k for game representation: {optimal_k_game}")
print(f"Optimal k for Mobius representation: {optimal_k_mobius}")

# Plot results
plot_k_additivity_results(
    results_game,
    plot_folder="results/k_additivity/plots",
    dataset_name="banknotes",
    representation="game"
)
```

### Robustness Testing

```python
from tests.robustness.robustness import test_model_robustness
from utils.visualization.plotting import plot_model_performance_comparison, plot_noise_robustness

# Create models dictionary
models = {
    "Game (k=2)": model_game,
    "Mobius (k=2)": model_mobius,
    "Shapley (k=2)": model_shapley
}

# Test robustness
robustness_results = test_model_robustness(
    models=models,
    X_test=X_test,
    y_test=y_test,
    feature_names=X.columns,
    output_folder="results/robustness"
)

# Plot results
plot_model_performance_comparison(
    robustness_results,
    plot_folder="results/robustness",
    title="Model Performance Comparison"
)

plot_noise_robustness(
    robustness_results,
    plot_folder="results/robustness"
)
```

## Visualization for Interpretability

```python
from utils.visualization.plotting import plot_coefficients, plot_interaction_matrix_2add

# Plot model coefficients
plot_coefficients(
    feature_names=X.columns.tolist(),
    all_coefficients=[model_shapley.coef_[0]],
    plot_folder="results/visualization",
    k_add=2
)

# Plot interaction matrix for Shapley representation (2-additive model)
plot_interaction_matrix_2add(
    feature_names=X.columns.tolist(),
    coefs=model_shapley.coef_[0],  # Coefficients from the fitted model
    plot_folder="results/visualization"
)
```

## Troubleshooting

### Import Errors

If you encounter import errors like `ModuleNotFoundError: No module named 'core'`, try one of these solutions:

1. **Install the package** (recommended):
   ```bash
   pip install -e .
   ```

2. **Run from the project root**:
   ```bash
   python -m examples.comparison_example
   ```

3. **Add the project root to your Python path** in your script:
   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```
