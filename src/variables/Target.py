import pandas as pd

class Target:
    
    @staticmethod
    def compute(log_returns: pd.Series, threshold: float = 0.001) -> pd.Series:
        """
        Renvoie une série binaire : 1 si le log return est supérieur au threshold, sinon 0.
        threshold : float, seuil pour classer les rendements (exemple : 0.0005 ≈ 0.05%)
        """
        return (log_returns > threshold).astype(int)
