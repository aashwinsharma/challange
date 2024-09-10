# Add necessary blank lines and fix long lines
import argparse
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn


def main(args):
    mlflow.sklearn.autolog()

    df = get_csvs_df(args.training_data)

    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run():
        train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    X = df.drop('Diabetic', axis=1)
    y = df['Diabetic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    model = LogisticRegression(
        C=1/reg_rate, solver="liblinear"
    )
    model.fit(
        X_train, y_train
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    mlflow.log_metric("accuracy", accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str, required=True)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    return parser.parse_args()

if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    args = parse_args()

    main(args)

    print("*" * 60)
    print("\n\n")
