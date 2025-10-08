import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler

class CryptoModel:

    """Classe pour entraîner et évaluer un modèle de classification binaire (XGBoost) avec une approche de fenêtre glissante."""

    def __init__(self, df: pd.DataFrame, target_col: str = "target", lag: int = 1):
        """
        df : DataFrame contenant les features et la cible.
        target_col : colonne cible (0/1).
        lag : décalage de la cible. Nous pouvons ajuster ce lag en fonction de l'horizon de prédiction souhaité. Par exemple, un lag de 1 signifie que nous essayons de prédire la valeur de la cible pour le prochain intervalle de temps (par exemple, la prochaine valeur à 5 minutes dans notre cas) en utilisant les données actuelles.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.lag = lag

        self.df[target_col] = self.df[target_col].shift(-lag)

        self.df.dropna(inplace=True)

        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]

        # Normalisation des features : permet de mettre toutes les variables sur la même échelle pour améliorer la performance et la stabilité du modèle.
        self.scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )


    def rolling_xgboost(self, months_train: int = 4, weeks_test: int = 4, n_trials: int = 100):
        """
        Entraîne et évalue un modèle XGBoost avec une approche de fenêtre glissante et optimisation des hyperparamètres avec Optuna.
        months_train : durée d'entraînement (en mois).
        weeks_test : durée de test (en semaines).
        n_trials : nombre d'essais pour l'optimisation des hyperparamètres.
        """
        results = []
        start_date = self.df.index.min()
        end_date = self.df.index.max()
        # Découpage par rolling window
        current_start = start_date
        window_count = 0
        while True:
            window_count += 1
            train_end = current_start + pd.DateOffset(months=months_train)
            test_end = train_end + pd.DateOffset(weeks=weeks_test)
            if test_end > end_date:
                break
            print(f"Train: {current_start.date()} to {train_end.date()}, Test: {train_end.date()} to {test_end.date()}")
            print("je suis à la fenêtre n°", window_count)
            # Séparation train/test
            X_train = self.X_scaled.loc[current_start:train_end]
            y_train = self.y.loc[current_start:train_end]
            X_test = self.X_scaled.loc[train_end:test_end]
            y_test = self.y.loc[train_end:test_end]

            def objective(trial):
                params = {
                    "max_depth": trial.suggest_int("max_depth", 4, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_int("gamma", 0, 5),
                    "min_child_weight": trial.suggest_int("min_child_weight", 10, 50)
                }
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                f1 = f1_score(y_test, y_pred)
                return f1

            study = optuna.create_study(direction="maximize", sampler=TPESampler())
            study.optimize(objective, n_trials=n_trials)

            # Get the best parameters
            best_params = study.best_params
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **best_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            results.append({
                "train_start": current_start,
                "train_end": train_end,
                "test_end": test_end,
                **best_params,
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

