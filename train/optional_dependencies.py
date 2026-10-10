"""Availability of optional ML, sampling and SHAP dependencies."""

try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = SMOTENC = ImbPipeline = None
    IMBLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. XGBoost will be skipped.")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not installed. LightGBM will be skipped.")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Winners will use default parameters.")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
