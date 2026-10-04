import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


mlflow.set_experiment("AI4I-Predictive-Maintenance")


def load_data():

    X_train = pd.read_csv(
        DATA_DIR / "X_train.csv"
    )

    X_test = pd.read_csv(
        DATA_DIR / "X_test.csv"
    )

    y_train = pd.read_csv(
        DATA_DIR / "y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze()

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    return metrics


def train_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    models = {

        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
    }

    results = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        with mlflow.start_run(
            run_name=name
        ):

            # Train
            model.fit(
                X_train,
                y_train
            )

            # Evaluate
            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            # Log parameters
            if name == "logistic_regression":

                mlflow.log_params({
                    "model": name,
                    "max_iter": 1000
                })

            elif name == "random_forest":

                mlflow.log_params({
                    "model": name,
                    "n_estimators": 200,
                    "class_weight": "balanced"
                })

            # Log metrics
            mlflow.log_metrics({
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"]
            })

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model"
            )

            print(
                f"Accuracy : "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"Recall   : "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"F1 Score : "
                f"{metrics['f1_score']:.4f}"
            )

            results[name] = {
                "model": model,
                "metrics": metrics
            }

    return results


def save_best_model(results):

    best_model_name = max(
        results,
        key=lambda name:
        results[name]["metrics"]["f1_score"]
    )

    best_model = results[
        best_model_name
    ]["model"]

    best_metrics = results[
        best_model_name
    ]["metrics"]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR /
        "best_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print("\n==============================")
    print("BEST MODEL")
    print("==============================")

    print(
        f"Model: {best_model_name}"
    )

    print(
        f"F1 Score: "
        f"{best_metrics['f1_score']:.4f}"
    )

    print(
        f"Saved to: {model_path}"
    )


def main():

    print(
        "Loading processed data..."
    )

    X_train, X_test, y_train, y_test = (
        load_data()
    )

    results = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_best_model(results)


if __name__ == "__main__":
    main()