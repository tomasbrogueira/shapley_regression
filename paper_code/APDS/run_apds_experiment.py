import os
import time
import json
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone

from preprocess import load_and_filter_data, build_feature_matrix
from models import get_models
from utils.metrics import compute_metrics, estimate_flops, model_size_mb


DATA_PATH = "data/data_apds.csv" #path to data
OUTPUT_DIR = "results"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    df = load_and_filter_data(DATA_PATH)
    X, y = build_feature_matrix(df)

    models = get_models()

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    summary, logs = [], []

    for model_name, (pipeline, grid) in models.items():
        print(f"\n▶ Running {model_name}")
        fold_metrics = []

        for fold, (tr, te) in enumerate(outer_cv.split(X, y), 1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]

            gs = GridSearchCV(
                pipeline,
                grid,
                scoring="balanced_accuracy",
                cv=inner_cv,
                n_jobs=-1,
            )

            gs.fit(X_tr, y_tr)
            best_model = clone(gs.best_estimator_)
            best_model.fit(X_tr, y_tr)

            y_pred = best_model.predict(X_te)
            y_proba = best_model.predict_proba(X_te)[:, 1]

            metrics = compute_metrics(y_te, y_pred, y_proba)
            metrics["model_size_mb"] = model_size_mb(best_model)
            metrics.update(estimate_flops(best_model, X_te))

            fold_metrics.append(metrics)

            joblib.dump(
                best_model,
                f"{OUTPUT_DIR}/{model_name}_fold{fold}.joblib",
            )

            del gs, best_model
            gc.collect()

        agg = {"model": model_name}
        for k in fold_metrics[0]:
            vals = [m[k] for m in fold_metrics]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))

        summary.append(agg)

    pd.DataFrame(summary).to_csv(
        f"{OUTPUT_DIR}/results.csv", index=False
    )

    with open(f"{OUTPUT_DIR}/logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("✔ Experiment completed successfully")


if __name__ == "__main__":
    main()
