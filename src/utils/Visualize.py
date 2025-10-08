# visualisation des résultats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def visualize_xgb_results_advanced(results, X_test, y_test, feature_names):
    """
    Visualisation avancée des résultats d'un modèle XGBoost
    - Metrics
    - Matrice de confusion (valeurs et pourcentages)
    - Importance des features (colorée selon importance)
    
    Args:
        model : XGBClassifier entraîné
        X_test : features de test
        y_test : target de test
        feature_names : liste des noms des features
    """
    # Prédictions
    y_pred = results.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("=== XGBoost Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    
    # Matrice de confusion avec pourcentages
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot=cm, fmt='d', cmap='Blues', cbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.5, f"{cm_percent[i,j]*100:.1f}%", 
                     color='red', ha='center', va='center', fontsize=10)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (counts + %)")    
    plt.show()
    
    # Importance des features
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # Couleur : plus clair = faible importance, plus foncé = haute importance
    colors = plt.cm.viridis(importance_df['Importance'] / importance_df['Importance'].max())
    
    plt.figure(figsize=(10,6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importances (colorées par magnitude)")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(self):
    """Affiche la courbe de loss et val_loss"""
    plt.figure(figsize=(10,5))
    plt.plot(self.history.history["loss"], label="train loss")
    plt.plot(self.history.history["val_loss"], label="val loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def plot_predictions(self):
    """Affiche les prédictions vs valeurs réelles"""
    yhat = self.model.predict(self.X_test)
    yhat = yhat[:,0] if self.forecast_horizon==1 else yhat[:,0]
    
    plt.figure(figsize=(12,5))
    plt.plot(self.y_test, label="True")
    plt.plot(yhat, label="Predicted", color="green")
    plt.title("LSTM Predictions")
    plt.xlabel("Samples")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_predicted_returns(self):
    """Affiche la distribution des rendements réels et prédit"""
    yhat = self.model.predict(self.X_test)
    yhat = yhat[:,0] if self.forecast_horizon==1 else yhat[:,0]
    predicted_return = (yhat - self.X_test[:,-1,0]) / self.X_test[:,-1,0]
    actual_return = (self.y_test - self.X_test[:,-1,0]) / self.X_test[:,-1,0]
    
    plt.figure(figsize=(10,5))
    plt.hist(actual_return, bins=50, alpha=0.5, label="Actual Return")
    plt.hist(predicted_return, bins=50, alpha=0.5, label="Predicted Return")
    plt.title("Distribution of Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_signal_comparison(self):
    """Compare le signal binaire réel et prédit (rendement >0 ou <=0)"""
    yhat = self.model.predict(self.X_test)
    yhat = yhat[:,0] if self.forecast_horizon==1 else yhat[:,0]
    predicted_return = (yhat - self.X_test[:,-1,0]) / self.X_test[:,-1,0]
    predicted_signal = np.where(predicted_return > 0, 1, 0)
    actual_return = (self.y_test - self.X_test[:,-1,0]) / self.X_test[:,-1,0]
    actual_signal = np.where(actual_return > 0, 1, 0)
    
    plt.figure(figsize=(12,5))
    plt.plot(actual_signal, label="Actual Signal", alpha=0.7)
    plt.plot(predicted_signal, label="Predicted Signal", alpha=0.7)
    plt.title("Binary Signal Comparison (1=positive return)")
    plt.xlabel("Samples")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()

