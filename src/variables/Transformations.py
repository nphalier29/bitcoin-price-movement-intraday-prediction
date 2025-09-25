import numpy as np
import pandas as pd

class Transformations:
    
    @staticmethod
    def return_(prices):
        """
        Rendements logarithmiques : ln(Pt / Pt-1)
        prices: iterable (list, Series, np.array)
        return: list de rendements log
        """
        prices = pd.Series(prices).astype(float)
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.tolist()
    
    @staticmethod
    def return_10(prices):
        """
        Rendements cumulés des 10 derniers prix (fenêtre glissante).
        Cumulative log-return = ln(Pt / Pt-10)
        
        prices: iterable (list, Series, np.array)
        return: list de rendements cumulatifs
        """
        prices = pd.Series(prices).astype(float)
        cum_log_returns = np.log(prices / prices.shift(10))
        return cum_log_returns.tolist()
    
