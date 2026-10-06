from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/processed")


def test_processed_data_exists():
    required_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
    ]

    for file_name in required_files:
        file_path = DATA_DIR / file_name
        assert file_path.exists(), f"Missing file: {file_path}"


def test_train_test_row_counts_match():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_processed_data_has_no_missing_values():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()