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
        self.df = df.copy()
        self.model = None
        self.scaler = None


        # Exclure colonnes temporelles + garder uniquement numériques
        self.feature_names = [
            col for col in df.columns
            if col not in ['target', 'open_time', 'close_time']
            and pd.api.types.is_numeric_dtype(df[col])
        ]


    def logistic_regression(self, test_size=0.2, random_state=42, verbose=True):
        """
        Régression logistique binaire
        """
        X = self.df[self.feature_names].values
        y = self.df['target'].values


        split_idx = int(len(X)*(1-test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]


        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)


        self.model = LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)


        y_pred = self.model.predict(X_test_scaled)


        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)


        if verbose:
            print("=== Logistic Regression Metrics ===")
            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall   : {rec:.4f}")
            print(f"F1-score : {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)


        return self.model


    def xgboost_classification(self, test_size=0.2, random_state=42, verbose=True, plot_importance=True):
        """
        Classification avec XGBoost
        """
        X = self.df[self.feature_names].values
        y = self.df['target'].values


        split_idx = int(len(X)*(1-test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]


        self.model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            use_label_encoder=False,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8
        )


        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)


        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)


        if verbose:
            print("=== XGBoost Metrics ===")
            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall   : {rec:.4f}")
            print(f"F1-score : {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)


            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)


            print("\n=== Feature Importances ===")
            print(importance_df)


            if plot_importance:
                plt.figure(figsize=(10, 6))
                plt.barh(importance_df['feature'], importance_df['importance'])
                plt.xlabel("Importance")
                plt.ylabel("Feature")
                plt.title("XGBoost Feature Importances")
                plt.gca().invert_yaxis()
                plt.show()


        return self.model
