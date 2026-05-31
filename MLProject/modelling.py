import os
import json

import mlflow
import mlflow.sklearn
import optuna

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("bitcoin.csv")

# Fitur dan target
features = [
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_rolling_mean_7",
    "price_rolling_mean_30",
    "total_volume",
    "market_cap",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "quarter"
]

target = "price"

X = df[features]
y = df[target]

# Split data time series
split_point = int(len(df) * 0.8)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]

y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]


# Objective Optuna
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 25, step=5),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestRegressor(**params)

    # Train model
    model.fit(X_train, y_train)

    # Predict sekali aja
    y_pred = model.predict(X_test)

    # Hitung metric
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Simpan metric
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("mae", mae)

    return r2

# Training
os.environ.pop("MLFLOW_RUN_ID", None)
mlflow.set_experiment("Bitcoin Optuna RandomForest")
with mlflow.start_run(run_name="Optuna_RF"):

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=3)

    # Ambil best trial
    best_trial = study.best_trial

    print("\nBest Trial")
    print(f"R2   : {best_trial.value:.4f}")
    print(f"RMSE : {best_trial.user_attrs['rmse']:.4f}")
    print(f"MAE  : {best_trial.user_attrs['mae']:.4f}")

    print("\nBest Params")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")

    # Train final model
    best_model = RandomForestRegressor(
        **best_trial.params,
        random_state=42,
        n_jobs=-1
    )

    best_model.fit(X_train, y_train)

    # Log params
    mlflow.log_params(best_trial.params)

    # Log metrics
    mlflow.log_metric("r2_score", best_trial.value)
    mlflow.log_metric("rmse", best_trial.user_attrs["rmse"])
    mlflow.log_metric("mae", best_trial.user_attrs["mae"])

    # Save model
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    # Save feature importance
    feature_importance = dict(
        zip(features, best_model.feature_importances_)
    )

    mlflow.sklearn.save_model(
        sk_model=best_model,
        path="saved_model"
    )

    with open("feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=4)

    mlflow.log_artifact("feature_importance.json")

    print("\nTraining selesai.")