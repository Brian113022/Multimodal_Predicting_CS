"""Standard estimator configurations and Optuna search spaces."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC

from .optional_dependencies import LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE, lgb, xgb


def get_rf_config():
    pipe = SklearnPipeline([("rf", RandomForestClassifier(n_jobs=1, random_state=42))])

    def opt(trial):
        return {
            "rf__n_estimators": trial.suggest_int("rf__n_estimators", 200, 800),
            "rf__max_depth": trial.suggest_int("rf__max_depth", 2, 10),
            "rf__min_samples_leaf": trial.suggest_int("rf__min_samples_leaf", 2, 20),
            "rf__max_features": trial.suggest_categorical(
                "rf__max_features", ["sqrt", "log2", 0.5, 1.0]
            ),
        }

    return pipe, opt


def get_knn_config():
    pipe = SklearnPipeline([("knn", KNeighborsClassifier())])

    def opt(trial):
        return {
            "knn__n_neighbors": trial.suggest_int("knn__n_neighbors", 3, 15),
            "knn__weights": trial.suggest_categorical("knn__weights", ["uniform", "distance"]),
            "knn__p": trial.suggest_int("knn__p", 1, 2),
        }

    return pipe, opt


def get_svm_config():
    pipe = SklearnPipeline([("svm", SVC(probability=True, random_state=42))])

    def opt(trial):
        kernel = trial.suggest_categorical("svm__kernel", ["linear", "rbf"])
        params = {
            "svm__kernel": kernel,
            "svm__C": trial.suggest_float("svm__C", 0.1, 50, log=True),
        }
        if kernel == "rbf":
            params["svm__gamma"] = trial.suggest_float("svm__gamma", 1e-3, 1, log=True)
        return params

    return pipe, opt


def get_lr_config():
    pipe = SklearnPipeline([
        ("lr", LogisticRegression(solver="liblinear", max_iter=5000, random_state=42))
    ])

    def opt(trial):
        return {
            "lr__C": trial.suggest_float("lr__C", 0.01, 10, log=True),
            "lr__penalty": trial.suggest_categorical("lr__penalty", ["l1", "l2"]),
        }

    return pipe, opt


def get_lgbm_config():
    if not LIGHTGBM_AVAILABLE:
        return None
    pipe = SklearnPipeline([
        ("lgbm", lgb.LGBMClassifier(n_jobs=1, random_state=42, verbose=-1, objective="binary"))
    ])

    def opt(trial):
        return {
            "lgbm__n_estimators": trial.suggest_int("lgbm__n_estimators", 100, 500),
            "lgbm__learning_rate": trial.suggest_float("lgbm__learning_rate", 1e-3, 0.3, log=True),
            "lgbm__num_leaves": trial.suggest_int("lgbm__num_leaves", 2, 31),
            "lgbm__max_depth": trial.suggest_int("lgbm__max_depth", 2, 7),
            "lgbm__min_child_samples": trial.suggest_int(
                "lgbm__min_child_samples", 10, 80
            ),
            "lgbm__subsample": trial.suggest_float("lgbm__subsample", 0.6, 1.0),
            "lgbm__colsample_bytree": trial.suggest_float(
                "lgbm__colsample_bytree", 0.5, 1.0
            ),
            "lgbm__reg_alpha": trial.suggest_float("lgbm__reg_alpha", 1e-3, 10.0, log=True),
            "lgbm__reg_lambda": trial.suggest_float("lgbm__reg_lambda", 1e-3, 10.0, log=True),
        }

    return pipe, opt


def get_xgb_config():
    if not XGBOOST_AVAILABLE:
        return None
    pipe = SklearnPipeline([
        ("xgb", xgb.XGBClassifier(
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        ))
    ])

    def opt(trial):
        return {
            "xgb__n_estimators": trial.suggest_int("xgb__n_estimators", 100, 500),
            "xgb__learning_rate": trial.suggest_float("xgb__learning_rate", 1e-3, 0.3, log=True),
            "xgb__max_depth": trial.suggest_int("xgb__max_depth", 1, 5),
            "xgb__min_child_weight": trial.suggest_float(
                "xgb__min_child_weight", 1.0, 20.0, log=True
            ),
            "xgb__subsample": trial.suggest_float("xgb__subsample", 0.6, 1.0),
            "xgb__colsample_bytree": trial.suggest_float(
                "xgb__colsample_bytree", 0.5, 1.0
            ),
            "xgb__gamma": trial.suggest_float("xgb__gamma", 1e-4, 5.0, log=True),
            "xgb__reg_alpha": trial.suggest_float(
                "xgb__reg_alpha", 1e-4, 10.0, log=True
            ),
            "xgb__reg_lambda": trial.suggest_float(
                "xgb__reg_lambda", 1e-3, 20.0, log=True
            ),
        }

    return pipe, opt


def get_models_map():
    models = {
        "RF": get_rf_config(),
        "KNN": get_knn_config(),
        "SVM": get_svm_config(),
        "LR": get_lr_config(),
    }
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = get_lgbm_config()
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = get_xgb_config()
    return {name: config for name, config in models.items() if config is not None}


def get_meta_models_map():
    """Use the same standard model zoo for base learning and late fusion."""
    return get_models_map()
