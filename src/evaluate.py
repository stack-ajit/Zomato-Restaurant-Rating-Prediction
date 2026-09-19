

import joblib
import matplotlib.pyplot as plt
import shap


def load_model_and_splits():
    model = joblib.load("models/best_model.pkl")
    X_train, X_test, y_train, y_test = joblib.load("models/train_test_splits.pkl")
    return model, X_train, X_test, y_train, y_test


def generate_shap_summary(model, X_test, output_path="reports/figures/shap_summary.png"):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {output_path}")


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = load_model_and_splits()
    generate_shap_summary(model, X_test)
