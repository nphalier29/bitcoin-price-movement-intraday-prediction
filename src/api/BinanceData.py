from binance.client import Client
import pandas as pd
import time
from datetime import datetime, timedelta

class BinanceData:
    """Classe pour interagir avec l'API Binance et récupérer des données de marché."""

    def __init__(self):
        self.client = Client()
        
    def load_data(self, symbol, days=30):

        try:
            interval = Client.KLINE_INTERVAL_5MINUTE
            max_limit = 1000  # Limite Binance par appel
            start_time = datetime.now() - timedelta(days=days)
            end_time = datetime.now()

            all_data = []
            current_time = start_time

            while current_time < end_time:
                # Récupérer 1000 bougies à partir de current_time
                data = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(current_time.timestamp() * 1000),
                    limit=max_limit
                )

                if not data:
                    break

                all_data.extend(data)

                # Le dernier "close_time" de la série actuelle
                last_close = data[-1][6]
                current_time = datetime.fromtimestamp(last_close / 1000)

                # Petite pause pour ne pas saturer l'API
                time.sleep(0.2)

            # Conversion en DataFrame
            df = pd.DataFrame(all_data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            return df.reset_index(drop=True)
        
        except Exception as e:
            print(f"Erreur lors de la récupération des données: {e}")
            return None
