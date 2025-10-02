import numpy as np
import pandas as pd

class TradingIndicators:
    """Classe pour ajouter des indicateurs techniques aux données de trading."""

    @staticmethod
    def add_sma(df: pd.DataFrame, price_col: str = "close", window: int = 20, new_col: str = "SMA_20") -> pd.DataFrame:
        """Ajoute une colonne SMA au DataFrame."""
        df[new_col] = df[price_col].rolling(window=window).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, price_col: str = "close", window: int = 12, new_col: str = "EMA_12") -> pd.DataFrame:
        """Ajoute une colonne EMA (Exponential Moving Average) au DataFrame. C'est une moyenne mobile qui donne plus de poids aux prix récents."""
        df[new_col] = df[price_col].ewm(span=window, adjust=False).mean()
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        price_col: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        macd_col: str = "MACD",
        signal_col: str = "MACD_Signal"
    ) -> pd.DataFrame:
        """Ajoute les colonnes MACD (Moving Average Convergence Divergence) et MACD_Signal au DataFrame. Permet d'identifier les changements de tendance."""
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
        df[macd_col] = ema_fast - ema_slow
        df[signal_col] = df[macd_col].ewm(span=signal, adjust=False).mean()
        return df

    @staticmethod
    def add_bollinger_bands_width(
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 20,
        num_std: float = 2.0,
        upper_col: str = "Bollinger_Upper",
        bb_width: str = "BB_width",
        #sma_col: str = "Bollinger_SMA",
        lower_col: str = "Bollinger_Lower"
    ) -> pd.DataFrame:
        """Ajoute les bandes de Bollinger au DataFrame. Permet d'identifier les conditions de surachat et de survente."""
        sma = df[price_col].rolling(window=window).mean()
        rolling_std = df[price_col].rolling(window=window).std()
        #df[sma_col] = sma
        df[bb_width] = sma + num_std * rolling_std - sma - num_std * rolling_std
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, price_col: str = "close", window: int = 14, new_col: str = "RSI_14") -> pd.DataFrame:
        """Ajoute une colonne RSI (Relative Strength Index) au DataFrame. Permet d'identifier les conditions de surachat et de survente."""
        delta = df[price_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        df[new_col] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_atr(
        df: pd.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        window: int = 14,
        new_col: str = "ATR_14"
    ) -> pd.DataFrame:
        """Ajoute une colonne ATR (Average True Range) au DataFrame. Cela mesure la volatilité récente."""
        tr = pd.DataFrame({
            "h-l": df[high_col] - df[low_col],
            "h-pc": abs(df[high_col] - df[close_col].shift(1)),
            "l-pc": abs(df[low_col] - df[close_col].shift(1))
        }).max(axis=1)
        df[new_col] = tr.rolling(window=window).mean()
        return df

    @staticmethod
    def add_high_low_range(df: pd.DataFrame, high_col: str = "high", low_col: str = "low", new_col: str = "High_Low_Range") -> pd.DataFrame:
        """Ajoute une colonne High-Low Range au DataFrame. Permet de mesurer l'amplitude des mouvements de prix."""
        df[new_col] = df[high_col] - df[low_col]
        return df

    @staticmethod
    def add_buy_pressure(df: pd.DataFrame, close_col: str = "close", high_col: str = "high", low_col: str = "low", new_col: str = "Buy_Pressure") -> pd.DataFrame:
        """Ajoute une colonne Buy Pressure au DataFrame. Permet de mesurer la pression d'achat par rapport à la plage de prix."""
        df[new_col] = (df[close_col] - df[low_col]) / (df[high_col] - df[low_col] + 1e-10)
        return df
    
    @staticmethod
    def add_volume_pressure(df: pd.DataFrame, taker_buy_col: str = "taker_buy_quote", total_volume_col: str = "quote_asset_volume", new_col: str = "Volume_Pressure") -> pd.DataFrame:
        """Ajoute une colonne Volume Pressure au DataFrame. Cela mesure la pression d'achat basée sur le volume."""
        df[new_col] = df[taker_buy_col] / (df[total_volume_col] + 1e-10)
        return df

    @staticmethod
    def add_realized_volatility(df: pd.DataFrame, returns_col: str = "return", window: int = 14, new_col: str = "Realized_Volatility") -> pd.DataFrame:
        """Ajoute une colonne de volatilité réalisée au DataFrame. Cela mesure la volatilité des rendements sur une période donnée."""
        df[new_col] = np.sqrt(df[returns_col].rolling(window=window).var())
        return df

    