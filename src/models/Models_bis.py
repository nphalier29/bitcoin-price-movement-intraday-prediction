import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


class CryptoModel:
    def __init__(self, df):
        """
        df : DataFrame contenant toutes les features et la target 'target'
        """
        self.df = df.copy().dropna()
        self.model = None
        self.scaler = None

        self.features = [col for col in self.df.columns if col != 'target']

    def xgboost_classification(self, verbose=True,  test_size=0.2,  random_state=42,):
        """
        Classification binaire avec XGBoost
        """
        X = self.df[self.features].values
        y = self.df['target'].values

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        X_test_scaled = scaler.transform(X_test)

        y_pred = self.model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print("=== XGBoost Classification Metrics ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        return 
    

# Scaler les données pour le logitic regression