import pandas as pd


def get_best_models(results_df):
    """
    Find the best model for each task group based on primary metrics.
    For regression: use r2 (higher is better)
    For classification: use f1_weighted (higher is better)
    """
    best_models = []

    # Group by task_type and group
    for (task_type, group), group_df in results_df.groupby(["task_type", "group"]):
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


def get_top_3_models(results_df, task_metrics=None):
    if task_metrics is None:
        task_metrics = {"regression": "r2", "classification": "f1_weighted"}

    required_cols = ["task_type", "group", "model", "targets", "best_params"]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    results = []

    for (task_type, group_name), group_df in results_df.groupby(["task_type", "group"]):
        metric = task_metrics.get(task_type)
        if metric not in group_df.columns:
            available_metrics = [
                col for col in group_df.columns if col in task_metrics.values()
            ]
            print(
                f"Warning: Metric '{metric}' not found for {task_type}. Available: {available_metrics}"
            )
            continue

        top_3_df = group_df.nlargest(3, metric)

        for idx, (_, row) in enumerate(top_3_df.iterrows(), 1):
            results.append(
                {
                    "task_type": task_type,
                    "group": group_name,
                    "rank": idx,
                    "model": row["model"],
                    "targets": row["targets"],
                    "primary_metric": metric,
                    "primary_score": row[metric],
                    "best_params": row["best_params"],
                }
            )

    return pd.DataFrame(results)
