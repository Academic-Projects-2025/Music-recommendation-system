import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

from src.music_recommender.models.model_registry import (
    MODEL_TEST,
    N_ITER_CONFIG,
    SCORING_METRICS,
    TARGET_GROUPS,
)


def train_models(X_train, X_test, y_train_df, y_test_df, target_groups, cfg):
    """Train all models and return results dataframe"""

    results = []
    for task_type, models in MODEL_TEST.items():
        scoring_metrics = SCORING_METRICS[task_type]
        groups = TARGET_GROUPS[task_type]

        for group_name, target_cols in groups.items():
            y_train_group = y_train_df[target_cols].values
            y_test_group = y_test_df[target_cols].values

            is_multi_output = len(target_cols) > 1

            for model_name, model_info in models.items():
                print(f"Training {model_name} on {task_type} - {group_name}...")

                base_model = model_info["base_model"]

                if (
                    task_type == "classification"
                    and model_name == "XGBoost"
                    and not is_multi_output
                ):
                    from sklearn.utils.class_weight import compute_class_weight

                    classes = np.unique(y_train_group.ravel())
                    if len(classes) == 2:
                        class_weights = compute_class_weight(
                            "balanced", classes=classes, y=y_train_group.ravel()
                        )
                        base_model.set_params(
                            scale_pos_weight=class_weights[1] / class_weights[0]
                        )

                if is_multi_output:
                    if task_type == "regression":
                        model = MultiOutputRegressor(base_model)
                        param_grid = {
                            k.replace("estimator__", "estimator__estimator__"): v
                            for k, v in model_info["param_grid"].items()
                        }
                    else:
                        model = MultiOutputClassifier(base_model)
                        param_grid = {
                            k.replace("estimator__", "estimator__estimator__"): v
                            for k, v in model_info["param_grid"].items()
                        }
                else:
                    model = base_model
                    param_grid = model_info["param_grid"]
                    y_train_group = y_train_group.ravel()
                    y_test_group = y_test_group.ravel()

                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=50)),
                        ("estimator", model),
                    ]
                )

                refit_metric = "r2" if task_type == "regression" else "f1_weighted"

                if task_type == "classification":
                    from sklearn.model_selection import StratifiedKFold

                    cv_strategy = StratifiedKFold(
                        n_splits=4, shuffle=True, random_state=42
                    )
                else:
                    cv_strategy = 4

                search = BayesSearchCV(
                    estimator=pipe,
                    search_spaces=param_grid,
                    scoring=scoring_metrics,
                    refit=refit_metric,
                    cv=cv_strategy,
                    n_iter=N_ITER_CONFIG[model_name],
                    n_jobs=-1,
                    random_state=42,
                )

                search.fit(X_train, y_train_group)
                y_pred = search.predict(X_test)

                result = {
                    "task_type": task_type,
                    "group": group_name,
                    "model": model_name,
                    "targets": ", ".join(target_cols),
                }

                if task_type == "regression":
                    if not is_multi_output:
                        y_test_group = y_test_group.reshape(-1, 1)
                        y_pred = y_pred.reshape(-1, 1)

                    result["r2"] = r2_score(y_test_group, y_pred)
                    result["neg_mae"] = -mean_absolute_error(y_test_group, y_pred)
                    result["neg_rmse"] = -np.sqrt(
                        mean_squared_error(y_test_group, y_pred)
                    )
                    try:
                        result["neg_mape"] = -mean_absolute_percentage_error(
                            y_test_group, y_pred
                        )
                    except:
                        result["neg_mape"] = None
                    result["explained_variance"] = explained_variance_score(
                        y_test_group, y_pred
                    )
                    for metric in SCORING_METRICS["classification"].values():
                        result[metric] = None
                else:
                    result["accuracy"] = accuracy_score(y_test_group, y_pred)
                    result["f1_macro"] = f1_score(
                        y_test_group, y_pred, average="macro", zero_division=0
                    )
                    result["f1_weighted"] = f1_score(
                        y_test_group, y_pred, average="weighted", zero_division=0
                    )
                    result["precision_macro"] = precision_score(
                        y_test_group, y_pred, average="macro", zero_division=0
                    )
                    result["recall_macro"] = recall_score(
                        y_test_group, y_pred, average="macro", zero_division=0
                    )
                    result["balanced_accuracy"] = balanced_accuracy_score(
                        y_test_group, y_pred
                    )

                    try:
                        if hasattr(
                            search.best_estimator_.named_steps["estimator"],
                            "predict_proba",
                        ):
                            y_proba = search.predict_proba(X_test)
                            if len(np.unique(y_test_group)) == 2:
                                result["roc_auc_ovr"] = roc_auc_score(
                                    y_test_group,
                                    y_proba[:, 1] if y_proba.ndim == 2 else y_proba,
                                )
                            else:
                                result["roc_auc_ovr"] = roc_auc_score(
                                    y_test_group,
                                    y_proba,
                                    multi_class="ovr",
                                    average="weighted",
                                )
                        else:
                            result["roc_auc_ovr"] = None
                    except Exception as e:
                        print(f"ROC AUC calculation failed: {e}")
                        result["roc_auc_ovr"] = None

                    for metric in SCORING_METRICS["regression"].values():
                        result[metric] = None

                result["best_params"] = str(search.best_params_)
                results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.paths.models / "model_comparison_results.csv", index=False)

    return results_df
