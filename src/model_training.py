
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import dagshub

from pathlib import Path
from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

REGISTERED_MODEL_NAME = "AI4I-Failure-Model"

EXPERIMENT_NAME = "AI4I-Predictive-Maintenance"

DAGSHUB_REPO_OWNER = "osamashabih6960"
DAGSHUB_REPO_NAME = "Production-Ready-MLOps-Pipeline"


# ============================================================
# INITIALIZE DAGSHUB + MLFLOW
# ============================================================

dagshub.init(
    repo_owner=DAGSHUB_REPO_OWNER,
    repo_name=DAGSHUB_REPO_NAME,
    mlflow=True
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# LOAD DATA
# ============================================================

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

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

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


# ============================================================
# TRAIN MODELS
# ============================================================

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

        print(
            f"\nTraining {name}..."
        )

        with mlflow.start_run(
            run_name=name
        ):

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # EVALUATE
            # ------------------------------------------------

            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            # ------------------------------------------------
            # LOG PARAMETERS
            # ------------------------------------------------

            if name == "logistic_regression":

                mlflow.log_params({

                    "model": name,

                    "max_iter": 1000,

                    "random_state": 42
                })

            elif name == "random_forest":

                mlflow.log_params({

                    "model": name,

                    "n_estimators": 200,

                    "random_state": 42,

                    "class_weight": "balanced"
                })

            # ------------------------------------------------
            # LOG METRICS
            # ------------------------------------------------

            mlflow.log_metrics({

                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1_score":
                    metrics["f1_score"]
            })

            # ------------------------------------------------
            # MLFLOW MODEL SIGNATURE
            # ------------------------------------------------

            input_example = (
                X_train.head(5)
            )

            predictions = model.predict(
                input_example
            )

            signature = infer_signature(
                input_example,
                predictions
            )

            # ------------------------------------------------
            # LOG MODEL
            # ------------------------------------------------

            mlflow.sklearn.log_model(

                model,

                "model",

                signature=signature,

                input_example=input_example
            )

            # ------------------------------------------------
            # PRINT METRICS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # SAVE RESULTS
            # ------------------------------------------------

            results[name] = {

                "model": model,

                "metrics": metrics,

                "run_id":
                    mlflow.active_run().info.run_id
            }

    return results


# ============================================================
# SAVE + REGISTER BEST MODEL
# ============================================================

def save_best_model(
    results
):

    # --------------------------------------------------------
    # FIND BEST MODEL
    # --------------------------------------------------------

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

    best_run_id = results[
        best_model_name
    ]["run_id"]


    # --------------------------------------------------------
    # SAVE MODEL LOCALLY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MLFLOW MODEL URI
    # --------------------------------------------------------

    model_uri = (
        f"runs:/{best_run_id}/model"
    )


    # --------------------------------------------------------
    # REGISTER MODEL
    # --------------------------------------------------------

    print(
        "\nRegistering best model "
        "with MLflow..."
    )

    registered_model = (
        mlflow.register_model(

            model_uri=model_uri,

            name=REGISTERED_MODEL_NAME
        )
    )


    # --------------------------------------------------------
    # SET CHAMPION ALIAS
    # --------------------------------------------------------

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(

        REGISTERED_MODEL_NAME,

        "champion",

        registered_model.version
    )


    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "BEST MODEL"
    )

    print(
        "=============================="
    )

    print(
        f"Model       : "
        f"{best_model_name}"
    )

    print(
        f"F1 Score    : "
        f"{best_metrics['f1_score']:.4f}"
    )

    print(
        f"Accuracy    : "
        f"{best_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : "
        f"{best_metrics['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{best_metrics['recall']:.4f}"
    )

    print(
        f"Run ID      : "
        f"{best_run_id}"
    )

    print(
        f"Registry    : "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version     : "
        f"{registered_model.version}"
    )

    print(
        "Alias       : champion"
    )

    print(
        f"Saved to    : "
        f"{model_path}"
    )

    print(
        "==============================\n"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading processed data..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()


    results = train_models(

        X_train,

        X_test,

        y_train,

        y_test
    )


    save_best_model(
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

