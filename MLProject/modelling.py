import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Aktifkan autolog
    mlflow.sklearn.autolog()
    
    # Buat dummy dataset jika file CSV tidak ditemukan (untuk simulasi CI test)
    data_dir = 'breast_cancer_preprocessing'
    try:
        X_train = pd.read_csv(f'{data_dir}/X_train.csv')
        X_test = pd.read_csv(f'{data_dir}/X_test.csv')
        y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
        y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    except FileNotFoundError:
        print("Data preprocessed tidak ditemukan, generate data sintesis untuk keperluan testing pipeline CI...")
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
    with mlflow.start_run(run_name="CI_Automated_Run"):
        print("Training model in CI Pipeline...")
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Model accuracy: {acc:.4f}")
        mlflow.sklearn.log_model(model, "model")
        print("Training finished.")

if __name__ == "__main__":
    main()
