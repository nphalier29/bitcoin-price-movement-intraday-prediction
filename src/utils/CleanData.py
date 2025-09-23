import pandas as pd

class CleanData:
    def __init__(self):
        self.useful_columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_quote"
        ]

    def clean_klines_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie un DataFrame de klines (Binance) :
        - Garde uniquement les colonnes utiles.
        - Convertit les timestamps en datetime.
        - Convertit les colonnes numériques en float.
        """

        cleaned_data = data[self.useful_columns].copy()

        cleaned_data["open_time"] = pd.to_datetime(cleaned_data["open_time"], unit="ms")
        cleaned_data["close_time"] = pd.to_datetime(cleaned_data["close_time"], unit="ms")

        numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume","number_of_trades", "taker_buy_quote"]
        for col in numeric_cols:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col])

        return cleaned_data
