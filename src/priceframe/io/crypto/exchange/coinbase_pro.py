from .abstract import Exchange
import requests
import asyncio
import aiohttp
from datetime import datetime

class CoinbasePro(Exchange):
    """
    Classe pour interagir avec l'API de Coinbase Pro.
    """

    name = "coinbase_pro"
    BASE_REST_URL = "https://api.exchange.coinbase.com"
    KLINE_URL = "/products/{symbol}/candles"
    SYMBOL_URL = "/products"
    LIMIT = 300
    VALID_INTERVALS = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "6h": 21600,
            "1d": 86400
        }
    
    @staticmethod
    def _process_klines(klines):
        """
        Convertit les données
        """
        klines = klines[::-1]
        
        return [
            {
                "ts": datetime.utcfromtimestamp(kline[0]).strftime('%Y-%m-%d %H:%M:%S'),
                "open": kline[3],
                "high": kline[2],
                "low": kline[1],
                "close": kline[4],
                "volume": ""
            }
            for kline in klines
        ]

    @classmethod
    async def get_historical_klines(cls, symbol, interval, start_time, end_time):
        """
        Récupère des chandelles historiques entre start_time et end_time.

        :param symbol: Symbole de trading (ex: 'BTC-USD').
        :param interval: Intervalle des chandelles (ex: '1m', '5m', '1h', '1d').
        :param start_time: Timestamp Unix (ms) de début.
        :param end_time: Timestamp Unix (ms) de fin.
        :return: Liste des chandelles.
        """
        # Vérifier si l'intervalle est valide
        if interval not in cls.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Valid intervals are: {', '.join(cls.VALID_INTERVALS.keys())}")

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            endpoint = f"{cls.BASE_REST_URL}{cls.KLINE_URL.format(symbol=symbol)}"
            klines = []

            # Convertir les timestamps en secondes (Coinbase Pro utilise des timestamps en secondes)
            start_time = start_time // 1000
            end_time = end_time // 1000
            granularity = cls.VALID_INTERVALS[interval]
            max_data_points = 300

            while start_time < end_time - granularity:
                
                # Calculer la fin de la plage de temps pour cette requête
                request_end_time = min(end_time, start_time + granularity * max_data_points)

                params = {
                    "start": datetime.utcfromtimestamp(start_time).isoformat(),
                    "end": datetime.utcfromtimestamp(request_end_time).isoformat(),
                    "granularity": granularity
                }

                async with session.get(endpoint, params=params) as response:
                    data = await response.json()
                    print(data)
                    if isinstance(data, list):
                        if len(data) < 2:
                            break
                        klines.extend(data)

                        # Avance le start_time à la fin de la dernière chandelle récupérée
                        last_candle_time = int(data[1][0])
                        start_time = last_candle_time + granularity
                        await asyncio.sleep(0.1)  # Petite pause pour éviter les limitations d'API
                    else:
                        print(data, "Error, retrying...")
                        await asyncio.sleep(5)  # Pause plus longue en cas d'erreur

            return cls._process_klines(klines)
        
    # async def get_historical_klines(self, symbol, interval, start_time, end_time):
    #     """
    #     Récupère des chandelles historiques entre start_time et end_time.

    #     :param symbol: Symbole de trading (ex: 'BTC-USD').
    #     :param interval: Intervalle des chandelles (ex: '1m', '5m', '1h', '1d').
    #     :param start_time: Timestamp Unix (ms) de début.
    #     :param end_time: Timestamp Unix (ms) de fin.
    #     :return: Liste des chandelles.
    #     """
    #     # Vérifier si l'intervalle est valide
    #     if interval not in self.valid_intervals:
    #         raise ValueError(f"Invalid interval '{interval}'. Valid intervals are: {', '.join(self.valid_intervals.keys())}")

    #     async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
    #         endpoint = f"{self.BASE_REST_URL}{self.KLINE_URL.format(symbol=symbol)}"
    #         klines = []

    #         # Convertir les timestamps en secondes (Coinbase Pro utilise des timestamps en secondes)
    #         start_time = start_time // 1000
    #         end_time = end_time // 1000
    #         granularity = self.valid_intervals[interval]
    #         print(granularity)

    #         while start_time < end_time:
    #             params = {
    #                 "start": datetime.utcfromtimestamp(start_time).isoformat(),
    #                 "end": datetime.utcfromtimestamp(end_time).isoformat(),
    #                 "granularity": granularity
    #             }

    #             async with session.get(endpoint, params=params) as response:
    #                 data = await response.json()

    #                 if isinstance(data, list):
    #                     if not len(data):
    #                         break
    #                     klines.extend(data)

    #                     # Avance le start_time à la fin de la dernière chandelle récupérée
    #                     last_candle_time = int(data[-1][0])
    #                     start_time = last_candle_time + granularity
    #                     await asyncio.sleep(0.1)  # Petite pause pour éviter les limitations d'API
    #                 else:
    #                     print(data, "Error, retrying...")
    #                     await asyncio.sleep(5)  # Pause plus longue en cas d'erreur

    #         return klines
        
    @classmethod
    def get_available_trading_pairs(cls):
        """
        Récupère la liste des paires de trading disponibles sur Coinbase Pro.
        """
        response = requests.get(cls.BASE_REST_URL + cls.SYMBOL_URL)
        if response.status_code == 200:
            data = response.json()
            return [product['id'] for product in data]
        else:
            raise Exception(f"Coinbase Pro API error: {response.status_code} - {response.text}")

    # async def subscribe_order_book(self, symbol: str, callback):
    #     """
    #     Connect to Coinbase Pro’s WebSocket and subscribe to level2 order book updates for a given symbol.
    #     The callback is called with standardized order book data.
        
    #     :param symbol: Trading pair symbol (e.g., 'BTC-USD')
    #     :param callback: A function to be called with the standardized order book dict.
    #     """
    #     ws_endpoint = "wss://ws-feed.exchange.coinbase.com"
    #     # Build a subscription message to the "level2" channel
    #     subscription_message = {
    #         "type": "subscribe",
    #         "channels": [{
    #             "name": "level2",
    #             "product_ids": [symbol]
    #         }]
    #     }
    #     async with aiohttp.ClientSession() as session:
    #         async with session.ws_connect(ws_endpoint) as ws:
    #             await ws.send_json(subscription_message)
    #             async for msg in ws:
    #                 if msg.type == aiohttp.WSMsgType.TEXT:
    #                     data = msg.json()
    #                     # Coinbase Pro sends a "snapshot" message (full order book) first and then "l2update" messages.
    #                     # A production system would merge the snapshot with incremental updates.
    #                     if data.get("type") == "snapshot":
    #                         standardized = {
    #                             "exchange": "coinbase_pro",
    #                             "symbol": data.get("product_id"),
    #                             "bids": [[float(price), float(size)] for price, size in data.get("bids", [])],
    #                             "asks": [[float(price), float(size)] for price, size in data.get("asks", [])],
    #                             "timestamp": None  # The snapshot does not include a timestamp, so you might use the current time.
    #                         }
    #                         callback(standardized)
    #                     elif data.get("type") == "l2update":
    #                         # l2update messages include only the changes.
    #                         changes = data.get("changes", [])
    #                         bids = []
    #                         asks = []
    #                         for change in changes:
    #                             side, price, size = change
    #                             if side.lower() == "buy":
    #                                 bids.append([float(price), float(size)])
    #                             else:
    #                                 asks.append([float(price), float(size)])
    #                         standardized = {
    #                             "exchange": "coinbase_pro",
    #                             "symbol": data.get("product_id", symbol),
    #                             "bids": bids,
    #                             "asks": asks,
    #                             "timestamp": data.get("time")  # typically an ISO timestamp
    #                         }
    #                         callback(standardized)
    #                 elif msg.type == aiohttp.WSMsgType.ERROR:
    #                     print("WebSocket error on Coinbase Pro")
    #                     break