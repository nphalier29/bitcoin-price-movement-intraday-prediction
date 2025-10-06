import pandas as pd
import numpy as np
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
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_int("gamma", 0, 5),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
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
    

    
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

    
class LSTMForecaster:
    def __init__(self, data, seq_length=5, forecast_horizon=1, epochs=500, batch_size=32):
        self.data = data
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None

    def build_sequences(self):
        X, y = [], []
        for t in range(len(self.data) - self.seq_length - self.forecast_horizon + 1):
            X.append(self.data[t:t+self.seq_length])
            y.append(self.data[t+self.seq_length:t+self.seq_length+self.forecast_horizon])
        X = np.array(X).reshape(-1, self.seq_length, 1)
        y = np.array(y)
        return X, y

    def train_test_split(self, X, y, test_size=0.2):
        split_idx = int(len(X)*(1-test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test

    def build_model(self):
        model = Sequential([
            LSTM(64, input_shape=(self.seq_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(self.forecast_horizon)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def fit(self, validation_split=0.2, verbose=1):
        X, y = self.build_sequences()
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        self.X_test = X_test
        self.y_test = y_test

        self.build_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        return self.history

    def evaluate(self):
        yhat = self.model.predict(self.X_test)
        yhat = yhat[:,0] if self.forecast_horizon == 1 else yhat
        mse = mean_squared_error(self.y_test, yhat)
        print("MSE:", mse)
        plt.figure(figsize=(12,5))
        plt.plot(self.y_test, label="True")
        plt.plot(yhat, label="Predicted", color="green")
        plt.title("LSTM Forecast")
        plt.legend()
        plt.show()
        return mse

    def plot_loss(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.history.history["loss"], label="loss")
        plt.plot(self.history.history["val_loss"], label="val_loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()

def predict_next_return(self, last_sequence):
    """
Prédit la prochaine valeur à partir de la dernière séquence connue
et calcule le rendement relatif par rapport à la dernière valeur connue.

Args:
    last_sequence : liste ou array de longueur seq_length
                    contenant les dernières valeurs de la série

Returns:
    predicted_return : rendement relatif (float)
                        >0 si prix prédit > dernière bougie
                        <0 si prix prédit < dernière bougie
"""

# Vérification de la longueur

if len(last_sequence) != self.seq_length:
    raise ValueError(f"last_sequence doit avoir une longueur de {self.seq_length}")

# Conversion en array et reshape pour LSTM
seq = np.array(last_sequence).reshape(1, self.seq_length, 1)

# Prédiction
yhat = self.model.predict(seq, verbose=0)

# Récupère la première valeur prédite si forecast_horizon >1
predicted_price = yhat[0,0] if self.forecast_horizon == 1 else yhat[0,0]

# Dernière valeur connue
last_price = last_sequence[-1]

# Calcul du rendement relatif
predicted_return = (predicted_price - last_price) / last_price

    return predicted_return

