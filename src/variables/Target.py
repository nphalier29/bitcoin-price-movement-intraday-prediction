import pandas as pd

class Target:

    @staticmethod
    def compute(log_returns: pd.Series) -> pd.Series:
        """
        Renvoie une série binaire : 1 si le log return est positif, sinon 0.
        """
        return (log_returns > 0).astype(int)
