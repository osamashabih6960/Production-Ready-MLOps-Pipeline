import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/ai4i2020.csv")
PROCESSED_DATA_PATH = Path("data/processed/ai4i2020.csv")


def load_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    return df


def validate_data(df):
    if df.empty:
        raise ValueError("Dataset is empty.")

    required_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("Data validation successful.")


def save_processed_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        f"Processed data saved to: "
        f"{PROCESSED_DATA_PATH}"
    )


def run():
    print("Starting data ingestion...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    validate_data(df)

    print("\nDataset columns:")
    for column in df.columns:
        print(f"- {column}")

    save_processed_data(df)

    print("\nData ingestion completed successfully.")


if __name__ == "__main__":
    run()