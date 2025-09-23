import pandas as pd

class CleanData:
    def __init__(self):
        self.useful_columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_quote"
        ]

    def clean_klines_data(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data[self.useful_columns].copy()

        cleaned = cleaned.set_index("open_time")
        cleaned.index = pd.to_datetime(cleaned.index, unit="ms")
        cleaned.index.name = "open_time"

        cleaned["close_time"] = pd.to_datetime(cleaned["close_time"], unit="ms")

        numeric_cols = ["open", "high", "low", "close", "volume",
                        "quote_asset_volume", "number_of_trades", "taker_buy_quote"]
        cleaned[numeric_cols] = cleaned[numeric_cols].apply(pd.to_numeric, errors="coerce")

        return cleaned

