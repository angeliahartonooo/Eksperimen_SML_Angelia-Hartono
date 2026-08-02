"""
Automasi preprocessing dataset Telco Customer Churn.

Konversi dari tahapan manual pada Eksperimen_Angelia-Hartono.ipynb (data loading,
cleaning TotalCharges, encoding kategorikal, scaling numerik, train-test split)
menjadi fungsi yang dapat dijalankan otomatis di luar notebook.
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_COL = "Churn"


def preprocess_data(input_path: str, output_dir: str, test_size: float = 0.2):
    """Load raw Telco Churn CSV, clean, encode, scale, split, and save train/test CSVs.

    Returns (X_train, X_test, y_train, y_test) for programmatic use (e.g. by modelling.py).
    """
    df = pd.read_csv(input_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.drop(columns=["customerID"])

    y = df[TARGET_COL].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[TARGET_COL])

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])

    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Preprocessing selesai. train={train_df.shape}, test={test_df.shape}")
    print(f"Output disimpan di: {output_dir}")

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset Telco Customer Churn")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "telco_churn_raw", "Telco-Customer-Churn.csv"),
        help="Path ke file CSV raw",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "telco_churn_preprocessing"),
        help="Folder output hasil preprocessing",
    )
    args = parser.parse_args()

    preprocess_data(args.input, args.output)


if __name__ == "__main__":
    main()
