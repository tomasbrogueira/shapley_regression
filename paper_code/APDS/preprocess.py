import pandas as pd


def load_and_filter_data(csv_path):
    """
    Load raw phenotype CSV and apply domain-specific filters.
    """
    df = pd.read_csv(csv_path)

    # Domain-specific exclusions
    df = df[df["concept_str_found"] != "deficit immunitaire primitif"]

    return df


def build_feature_matrix(df, label_col="label"):
    """
    Convert long-format phenotype table into:
      X: patient × phenotype binary matrix
      y: patient-level label vector
    """
    df = df[["patient_num", "concept_str_found", label_col]].copy()
    df["presence"] = 1

    X = df.pivot_table(
        index="patient_num",
        columns="concept_str_found",
        values="presence",
        fill_value=0,
    )

    y = (
        df[["patient_num", label_col]]
        .drop_duplicates(subset="patient_num")
        .set_index("patient_num")[label_col]
    )

    X = X.reindex(index=y.index)

    return X, y
