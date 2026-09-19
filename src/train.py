
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

RANDOM_STATE = 42


def load_features(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["rate"])
    y = df["rate"]
    return X, y


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name:20s}  RMSE: {rmse:.4f}   R2: {r2:.4f}")
    return {"model": name, "rmse": rmse, "r2": r2}


def train_all(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []
    trained_models = {}

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    results.append(evaluate("Linear Regression", lin_reg, X_test, y_test))
    trained_models["Linear Regression"] = lin_reg

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))
    trained_models["Random Forest"] = rf

    xgb = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgb, X_test, y_test))
    trained_models["XGBoost"] = xgb

    results_df = pd.DataFrame(results).sort_values("rmse")
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]

    print(f"\nBest model: {best_name}")
    return results_df, best_model, best_name, (X_train, X_test, y_train, y_test)


def save_comparison_table(results_df: pd.DataFrame, path: str):
    with open(path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write(results_df.to_markdown(index=False))
    print(f"Saved comparison table to {path}")


if __name__ == "__main__":
    X, y = load_features("data/processed/zomato_features.csv")
    results_df, best_model, best_name, splits = train_all(X, y)

    save_comparison_table(results_df, "reports/model_comparison.md")
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(splits, "models/train_test_splits.pkl")  # for use in evaluate.py / SHAP
    print("Saved best model to models/best_model.pkl")
