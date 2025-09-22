from binance.client import Client
import pandas as pd

class BinanceData:

    def __init__(self):
        self.client = Client()
        
    def load_data(self, symbol, days=1):
        """ Récupère les données de bougies (klines) pour un symbole donné, sur des intervalles de 5 minutes, pour une journée. 
        symbol: Paire de trading (ex: "BTCUSDT") 
        days: Nombre de jours (1 par défaut) 
        return: DataFrame pandas avec les données 
        """
        try:
            max_limit = 1000 #limite de données par requête binance
            limit = min(288 * days, max_limit)

            data = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                limit=limit
            )

            # Conversion en DataFrame
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            return df
        
        except Exception as e:
            print(f"Erreur lors de la récupération des données: {e}")
            return None