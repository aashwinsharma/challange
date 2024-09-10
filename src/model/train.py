# Import libraries
import argparse
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


import mlflow
import mlflow.sklearn


def main(args):  # Line 15
    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model and log metrics
    with mlflow.start_run():
        train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):  # Line 29
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")

    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")

    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):  # Line 37
    # Assume 'Diabetic' is the target column
    X = df.drop('Diabetic', axis=1)
    y = df['Diabetic']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):  # Line 47
    # Train logistic regression model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(
        f"Accuracy: {accuracy}"
    )
    print(
        f"Classification Report:\n{report}"
    )

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)


def parse_args():  # Line 63
    # Setup arg parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--training_data", dest='training_data', type=str, required=True
    )
    parser.add_argument(
        "--reg_rate", dest='reg_rate', type=float, default=0.01
    )

    # Parse args
    args = parser.parse_args()

    return args


# Run script
if __name__ == "__main__":  # Line 78
    print("\n\n")
    print("*" * 60)

    # Parse args
    args = parse_args()

    # Run main function
    main(args)

    print("*" * 60)
    print("\n\n")
