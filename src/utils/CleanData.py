import pandas as pd

class CleanData:
    """Classe pour nettoyer et préparer le dataset."""
    
    def __init__(self):
        self.useful_columns = [
            "open_time", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_quote"
        ]

    def clean_klines_data(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data[self.useful_columns].copy()

        cleaned = cleaned.set_index("open_time")
        cleaned.index = pd.to_datetime(cleaned.index, unit="ms")
        cleaned.index.name = "open_time"

        numeric_cols = ["high", "low", "close", "volume","quote_asset_volume", "number_of_trades", "taker_buy_quote"]
        
        cleaned[numeric_cols] = cleaned[numeric_cols].apply(pd.to_numeric, errors="coerce")

        return cleaned

