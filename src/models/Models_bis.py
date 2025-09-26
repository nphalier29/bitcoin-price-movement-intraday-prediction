import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import product

class CryptoModel:

    """Classe pour entraîner et évaluer un modèle de classification binaire (XGBoost) avec une approche de fenêtre glissante."""

    def __init__(self, df: pd.DataFrame, target_col: str = "target", lag: int = 1):
        """
        df : DataFrame contenant les features et la cible.
        target_col : colonne cible (0/1).
        lag : décalage de la cible.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.lag = lag

        self.df[target_col] = self.df[target_col].shift(-lag)

        self.df.dropna(inplace=True)

        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]

        # Normalisation
        self.scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )

    def rolling_xgboost(self, months_train: int = 7, weeks_test: int = 1, param_grid: dict = None):
        """
        Entraîne et évalue un modèle XGBoost avec une approche de fenêtre glissante.

        months_train : durée d'entraînement (en mois).
        weeks_test : durée de test (en semaines).
        param_grid : dict d'hyperparamètres à tester.
        """
        if param_grid is None:
            param_grid = {
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.01],
                "n_estimators": [100]
            }

        results = []

        # Génération de toutes les combinaisons d'hyperparamètres
        param_combinations = list(product(*param_grid.values()))
        param_keys = list(param_grid.keys())

        start_date = self.df.index.min()
        end_date = self.df.index.max()

        # Découpage par rolling window
        current_start = start_date
        while True:
            train_end = current_start + pd.DateOffset(months=months_train)
            test_end = train_end + pd.DateOffset(weeks=weeks_test)

            if test_end > end_date:
                break

            # Séparation train/test
            X_train = self.X_scaled.loc[current_start:train_end]
            y_train = self.y.loc[current_start:train_end]
            X_test = self.X_scaled.loc[train_end:test_end]
            y_test = self.y.loc[train_end:test_end]

            for params in param_combinations:
                param_dict = dict(zip(param_keys, params))
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **param_dict)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                results.append({
                    "train_start": current_start,
                    "train_end": train_end,
                    "test_end": test_end,
                    **param_dict,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                    "y_proba": y_proba
                })

            # Décalage de la fenêtre
            current_start = current_start + pd.DateOffset(weeks=weeks_test)

        return pd.DataFrame(results)
