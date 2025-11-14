import ast
import inspect
from collections import defaultdict
from typing import Any, Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    r2_score,
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.music_recommender.utils.logger import get_logger

from .model_registry import MODEL_CLASS_LOOKUP, TARGET_GROUPS

logger = get_logger(context="hybrid")


class HybridModel(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        top_models: Dict[str, Any],
        best_models: Dict[str, Any] | None = None,
        skip_stacking: Dict[str, list] | None = None,
        target_groups: dict = TARGET_GROUPS,
        lookup_table: Dict[str, Any] = MODEL_CLASS_LOOKUP,
        random_state: int = 42,
        cv: int = 5,
        final_estimator_reg: BaseEstimator = Ridge(alpha=1.0),
        final_estimator_class: BaseEstimator = LogisticRegression(
            max_iter=5000, random_state=42
        ),
    ) -> None:
        super().__init__()
        self.top_models = top_models
        self.best_models = best_models
        self.skip_stacking = skip_stacking or {}
        self.lookup_table = lookup_table
        self.random_state = random_state
        self.final_estimator_reg = final_estimator_reg
        self.final_estimator_class = final_estimator_class
        self.cv = cv
        self.target_groups = target_groups
        self.estimator_stack: Dict | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def get_clean_params(params: str):
        params = params.replace("OrderedDict", "")
        params = params.replace("estimator__estimator__", "")
        params = params.replace("estimator__", "")

        try:
            params_list = ast.literal_eval(params)
            param_dict = dict(params_list)
            return param_dict
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing params: {params}")
            raise e

    def _make_regression_base(self):
        estimator_stack_reg = defaultdict(list)
        for task_type, task_info in self.top_models.items():
            if task_type == "regression":
                for target, models_stack in task_info.items():
                    for model_info in models_stack.values():
                        model_class = self.lookup_table[task_type][
                            model_info["model_name"]
                        ]
                        clean_params = self.get_clean_params(model_info["best_params"])
                        if (
                            "random_state"
                            in inspect.signature(model_class.__init__).parameters
                            and "random_state" not in clean_params.keys()
                        ):
                            clean_params["random_state"] = self.random_state
                        estimator_stack_reg[target].append(
                            {model_info["model_name"]: model_class(**clean_params)}
                        )
        return dict(estimator_stack_reg)

    def _make_classification_base(self):
        estimator_stack_class = defaultdict(list)
        for task_type, task_info in self.top_models.items():
            if task_type == "classification":
                for target, models_stack in task_info.items():
                    for model_info in models_stack.values():
                        model_class = self.lookup_table[task_type][
                            model_info["model_name"]
                        ]
                        clean_params = self.get_clean_params(model_info["best_params"])
                        if (
                            "random_state"
                            in inspect.signature(model_class.__init__).parameters
                        ):
                            clean_params["random_state"] = self.random_state
                        estimator_stack_class[target].append(
                            {model_info["model_name"]: model_class(**clean_params)}
                        )
        return dict(estimator_stack_class)

    def _make_stack_reg(self):
        stacking_reg = {}
        skip_list = (
            self.skip_stacking.get("regression", []) if self.skip_stacking else []
        )

        for target, models_stack in self._make_regression_base().items():
            target_cols = self.target_groups["regression"][target]

            if target in skip_list:
                best_model_info = self.best_models["regression"][target]
                model_class = self.lookup_table["regression"][
                    best_model_info["model_name"]
                ]
                clean_params = self.get_clean_params(best_model_info["best_params"])

                if (
                    "random_state" in inspect.signature(model_class.__init__).parameters
                    and "random_state" not in clean_params.keys()
                ):
                    clean_params["random_state"] = self.random_state

                stacking_reg[target] = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=50)),
                        ("estimator", model_class(**clean_params)),
                    ]
                )
                continue

            estimators = []
            for model_info in models_stack:
                for name, model in model_info.items():
                    preprocessed_model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=50)),
                            ("estimator", model),
                        ]
                    )
                    estimators.append((name, preprocessed_model))

            base_stacker = StackingRegressor(
                estimators=estimators,
                final_estimator=self.final_estimator_reg,
                cv=self.cv,
                n_jobs=-1,
            )

            if len(target_cols) > 1:
                stacking_reg[target] = MultiOutputRegressor(base_stacker)
            else:
                stacking_reg[target] = base_stacker

        return stacking_reg

    def _make_stack_class(self) -> dict:
        stacking_class = {}
        skip_list = (
            self.skip_stacking.get("classification", []) if self.skip_stacking else []
        )

        for target, models_stack in self._make_classification_base().items():
            target_cols = self.target_groups["classification"][target]

            if target in skip_list:
                best_model_info = self.best_models["classification"][target]
                model_class = self.lookup_table["classification"][
                    best_model_info["model_name"]
                ]
                clean_params = self.get_clean_params(best_model_info["best_params"])

                if (
                    "random_state" in inspect.signature(model_class.__init__).parameters
                    and "random_state" not in clean_params.keys()
                ):
                    clean_params["random_state"] = self.random_state

                stacking_class[target] = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=50)),
                        ("estimator", model_class(**clean_params)),
                    ]
                )
                continue

            estimators = []
            for model_info in models_stack:
                for name, model in model_info.items():
                    preprocessed_model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=50)),
                            ("estimator", model),
                        ]
                    )
                    estimators.append((name, preprocessed_model))

            base_stacker = StackingClassifier(
                estimators=estimators,
                final_estimator=self.final_estimator_class,
                cv=self.cv,
                n_jobs=-1,
            )

            if len(target_cols) > 1:
                stacking_class[target] = MultiOutputClassifier(base_stacker)
            else:
                stacking_class[target] = base_stacker

        return stacking_class

    def _make_stack(self):
        return {
            "regression": {**self._make_stack_reg()},
            "classification": {**self._make_stack_class()},
        }

    def fit(self, X, y):
        logger.info("Building hybrid stacking ensemble...")
        self.estimator_stack = self._make_stack()

        for task_type, groups in self.estimator_stack.items():
            skip_list = self.skip_stacking.get(task_type, [])

            for group_name, stacker in groups.items():
                target_cols = self.target_groups[task_type][group_name]
                y_group = y[target_cols].values

                if len(target_cols) == 1:
                    y_group = y_group.ravel()

                stacking_status = (
                    "(best model only)" if group_name in skip_list else "(stacking)"
                )
                logger.info(
                    f"Fitting {task_type} - {group_name} {stacking_status} ({', '.join(target_cols)})..."
                )

                stacker.fit(X, y_group)

        self.is_fitted_ = True
        logger.info("✓ Hybrid model training complete!")
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction!")

        predictions = {}

        for task_type, groups in self.estimator_stack.items():
            for group_name, stacker in groups.items():
                target_cols = self.target_groups[task_type][group_name]
                y_pred = stacker.predict(X)

                if len(target_cols) == 1:
                    predictions[target_cols[0]] = y_pred
                else:
                    for i, col in enumerate(target_cols):
                        predictions[col] = y_pred[:, i]

        return pd.DataFrame(predictions)

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction!")

        probabilities = {}

        if "classification" in self.estimator_stack:
            for group_name, stacker in self.estimator_stack["classification"].items():
                target_cols = self.target_groups["classification"][group_name]

                try:
                    y_proba = stacker.predict_proba(X)

                    if len(target_cols) == 1:
                        probabilities[target_cols[0]] = y_proba
                    else:
                        for i, col in enumerate(target_cols):
                            probabilities[col] = y_proba[i]
                except AttributeError:
                    print(f"Warning: {group_name} doesn't support predict_proba")

        return probabilities

    def score(self, X, y):
        predictions = self.predict(X)
        scores = {}

        for task_type, groups in self.estimator_stack.items():
            for group_name, _ in groups.items():
                target_cols = self.target_groups[task_type][group_name]

                y_true = y[target_cols].values
                y_pred = predictions[target_cols].values

                if task_type == "regression":
                    score_val = r2_score(y_true, y_pred)
                    metric_name = "R²"
                else:
                    if len(target_cols) == 1:
                        score_val = accuracy_score(y_true.ravel(), y_pred.ravel())
                    else:
                        score_val = accuracy_score(y_true, y_pred)
                    metric_name = "Accuracy"

                scores[f"{group_name}"] = {
                    "metric": metric_name,
                    "score": score_val,
                    "targets": target_cols,
                }

        return scores
