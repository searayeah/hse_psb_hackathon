import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 0
CAT_FEATURES = ["Способ оплаты", "Источник", "Категория номера"]
EVAL_METRIC = "AUC"


def fit_catboost(trial, train, val):

    X_train, y_train = train
    X_val, y_val = val

    params = {
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10),
        # "border_count": trial.suggest_int("border_count", 32, 512),
        # "random_strength": trial.suggest_float("random_strength", 1e-3, 10),
        # "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
        # "rsm": trial.suggest_float("rsm", 0.1, 1.0),
        # "boosting_type": trial.suggest_categorical(
        #     "boosting_type", ["Ordered", "Plain"]
        # ),
        # "bootstrap_type": trial.suggest_categorical(
        #     "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        # ),
        # "leaf_estimation_method": trial.suggest_categorical(
        #     "leaf_estimation_method", ["Newton", "Gradient"]
        # ),
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 20.0
        )
    # elif params["bootstrap_type"] == "Bernoulli":
    #     params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    model = CatBoostClassifier(
        **params,
        verbose=0,
        thread_count=-1,
        random_seed=RANDOM_SEED,
        cat_features=CAT_FEATURES,
        eval_metric=EVAL_METRIC,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0,
    )

    preds = model.predict_proba(X_val)

    return model, preds


def fit_lgbm(trial, train, val):

    X_train, y_train = train
    X_val, y_val = val

    params = {
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["gbdt", "dart", "goss"]
        ),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    if params["boosting_type"] != "goss":
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)
        params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)

    if params["boosting_type"] != "dart":
        params["early_stopping_round"] = 50

    model = LGBMClassifier(
        **params,
        n_estimators=1000,
        verbose=-1,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=EVAL_METRIC,
        categorical_feature=CAT_FEATURES,
    )

    preds = model.predict_proba(X_val)

    return model, preds


def fit_rf(trial, train, val):

    X_train, y_train = train
    X_val, y_val = val

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": trial.suggest_float("max_features", 0.0, 1.0),
    }

    model = RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)

    return model, preds


def fit_et(trial, train, val):

    X_train, y_train = train
    X_val, y_val = val

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": trial.suggest_float("max_features", 0.0, 1.0),
    }

    model = ExtraTreesClassifier(**params, n_jobs=-1, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)

    return model, preds


def objective(trial, X_train, y_train, model_type, return_models=False, **kwargs):

    model_types_dict = {
        "CatBoost": fit_catboost,
        "LightGBM": fit_lgbm,
        "RandomForest": fit_rf,
        "ExtraTrees": fit_et,
    }

    if model_type not in model_types_dict:
        raise ValueError("Передан неподходящий тип модели (параметр model_type)")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    scores, models = [], []

    for train_index, val_index in kf.split(X_train, y_train):
        train = X_train.iloc[train_index], y_train.iloc[train_index]
        val = X_train.iloc[val_index], y_train.iloc[val_index]

        model, y_pred = model_types_dict[model_type](trial, train, val, **kwargs)
        scores.append(roc_auc_score(val[1], y_pred[:, 1]))
        models.append(model)

    result = np.mean(scores)

    if return_models:
        return result, models
    return result
