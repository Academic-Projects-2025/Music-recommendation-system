from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVC, SVR
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier, XGBRegressor

MODEL_CLASS_LOOKUP = {
    "regression": {
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "Random Forest": RandomForestRegressor,
        "XGBoost": XGBRegressor,
        "SVM": SVR,
    },
    "classification": {
        "Random Forest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "SVM": SVC,
    },
}

MODEL_TEST = {
    "regression": {
        "Ridge": {
            "base_model": Ridge(),
            "param_grid": {
                "estimator__alpha": Real(0.01, 100.0, prior="log-uniform"),
                "estimator__solver": Categorical(
                    ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
                ),
                "estimator__random_state": Categorical([42]),
                "estimator__max_iter": Integer(1000, 5000),
                "estimator__tol": Real(1e-5, 1e-2, prior="log-uniform"),
            },
        },
        "Lasso": {
            "base_model": Lasso(),
            "param_grid": {
                "estimator__alpha": Real(0.00001, 10.0, prior="log-uniform"),
                "estimator__selection": Categorical(["cyclic", "random"]),
                "estimator__random_state": Categorical([42]),
                "estimator__max_iter": Integer(1000, 5000),
                "estimator__tol": Real(1e-5, 1e-2, prior="log-uniform"),
            },
        },
        "ElasticNet": {
            "base_model": ElasticNet(),
            "param_grid": {
                "estimator__alpha": Real(0.00001, 10.0, prior="log-uniform"),
                "estimator__l1_ratio": Real(0.0, 1.0),
                "estimator__selection": Categorical(["cyclic", "random"]),
                "estimator__random_state": Categorical([42]),
                "estimator__max_iter": Integer(1000, 5000),
                "estimator__tol": Real(1e-5, 1e-2, prior="log-uniform"),
            },
        },
        "Random Forest": {
            "base_model": RandomForestRegressor(),
            "param_grid": {
                "estimator__n_estimators": Integer(100, 500),
                "estimator__max_depth": Integer(5, 30),
                "estimator__min_samples_split": Integer(2, 20),
                "estimator__min_samples_leaf": Integer(1, 10),
                "estimator__max_features": Categorical(["sqrt", "log2", None]),
                "estimator__random_state": Categorical([42]),
            },
        },
        "XGBoost": {
            "base_model": XGBRegressor(),
            "param_grid": {
                "estimator__n_estimators": Integer(200, 700),
                "estimator__max_depth": Integer(8, 15),
                "estimator__learning_rate": Real(0.01, 0.05, prior="log-uniform"),
                "estimator__subsample": Real(0.6, 1.0),
                "estimator__colsample_bytree": Real(0.6, 1.0),
                "estimator__gamma": Real(0, 1.0),
                "estimator__random_state": Categorical([42]),
            },
        },
        "SVM": {
            "base_model": SVR(kernel="rbf"),
            "param_grid": {
                "estimator__C": Real(0.1, 100.0, prior="log-uniform"),
                "estimator__gamma": Real(0.00001, 0.01, prior="log-uniform"),
                "estimator__epsilon": Real(0.01, 0.2),
            },
        },
    },
    "classification": {
        "Random Forest": {
            "base_model": RandomForestClassifier(class_weight="balanced"),
            "param_grid": {
                "estimator__n_estimators": Integer(150, 350),
                "estimator__max_depth": Integer(8, 20),
                "estimator__min_samples_split": Integer(2, 10),
                "estimator__min_samples_leaf": Integer(3, 6),
                "estimator__max_features": Categorical(["sqrt", "log2", None]),
                "estimator__bootstrap": Categorical([True, False]),
                "estimator__random_state": Categorical([42]),
            },
        },
        "XGBoost": {
            "base_model": XGBClassifier(),
            "param_grid": {
                "estimator__n_estimators": Integer(300, 700),
                "estimator__max_depth": Integer(8, 15),
                "estimator__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "estimator__subsample": Real(0.6, 1.0),
                "estimator__colsample_bytree": Real(0.8, 1.0),
                "estimator__gamma": Real(0, 2.0),
                "estimator__reg_alpha": Real(0, 10.0),
                "estimator__reg_lambda": Real(0, 30.0),
                "estimator__random_state": Categorical([42]),
            },
        },
        "SVM": {
            "base_model": SVC(kernel="rbf", probability=True, class_weight="balanced"),
            "param_grid": {
                "estimator__C": Real(1.0, 20.0, prior="log-uniform"),
                "estimator__gamma": Real(0.001, 0.02, prior="log-uniform"),
                "estimator__random_state": Categorical([42]),
            },
        },
    },
}

N_ITER_CONFIG = {
    "Ridge": 30,
    "Lasso": 30,
    "ElasticNet": 30,
    "Random Forest": 50,
    "XGBoost": 50,
    "SVM": 40,
}

SCORING_METRICS = {
    "regression": {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mape": "neg_mean_absolute_percentage_error",
        "explained_variance": "explained_variance",
    },
    "classification": {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "balanced_accuracy": "balanced_accuracy",
    },
}
TARGET_GROUPS = {
    "regression": {
        "energy_mood": ["energy", "valence", "danceability"],
        "production": ["loudness", "acousticness", "instrumentalness", "liveness"],
        "structure": ["speechiness"],
    },
    "classification": {"key": ["key"], "mode": ["mode"], "tempo_bins": ["tempo_bins"]},
}
