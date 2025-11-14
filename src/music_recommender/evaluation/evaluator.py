import pandas as pd

def get_best_models(df):
    """
    Find the best model for each task group based on primary metrics.
    For regression: use r2 (higher is better)
    For classification: use f1_weighted (higher is better)
    """
    best_models = []

    # Group by task_type and group
    for (task_type, group), group_df in df.groupby(["task_type", "group"]):
        if task_type == "regression":
            # For regression, maximize R²
            best_idx = group_df["r2"].idxmax()
            metric_col = "r2"
            metric_value = group_df.loc[best_idx, "r2"]
        else:  # classification
            # For classification, maximize F1-weighted
            best_idx = group_df["f1_weighted"].idxmax()
            metric_col = "f1_weighted"
            metric_value = group_df.loc[best_idx, "f1_weighted"]

        best_model = group_df.loc[best_idx]
        best_models.append(
            {
                "task_type": task_type,
                "group": group,
                "best_model": best_model["model"],
                "targets": best_model["targets"],
                "primary_metric": metric_col,
                "primary_score": metric_value,
                "best_params": best_model["best_params"],
            }
        )

    return pd.DataFrame(best_models)