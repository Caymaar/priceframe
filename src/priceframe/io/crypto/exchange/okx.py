from .abstract import Exchange
import requests
import asyncio
import aiohttp
from datetime import datetime

class OKX(Exchange):
    """
    Classe pour interagir avec l'API de OKX.
    """
    name = "okx"
    BASE_REST_URL = "https://www.okx.com"
    KLINE_URL = "/api/v5/market/candles"
    SYMBOL_URL = "/api/v5/public/instruments"
    LIMIT = 100
    VALID_INTERVALS = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6H",
        "12h": "12H",
        "1d": "1D",
        "2d": "2D",
        "3d": "3D",
        "1w": "1W",
        "1M": "1M",
        "3M": "3M",
    }

    @staticmethod
    def process_klines(klines):
        """
        Convertit les données
        """
        klines = klines[::-1]
        
        return [
            {
                "ts": datetime.utcfromtimestamp(int(kline[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
            }
            for kline in klines
        ]

    @classmethod
    async def get_historical_klines(cls, symbol, interval, start_time, end_time):
        """
        Récupère des chandelles historiques entre start_time et end_time.

        :param symbol: Symbole de trading (ex: 'BTC-USDT').
        :param interval: Intervalle des chandelles (ex: '1m', '5m', '1H', '1D').
        :param start_time: Timestamp Unix (ms) de début.
        :param end_time: Timestamp Unix (ms) de fin.
        :return: Liste des chandelles.
        """
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            endpoint = f"{cls.BASE_REST_URL}{cls.KLINE_URL}"
            klines = []
            old_data = {}
            while start_time < end_time:
                params = {
                    "instId": symbol,  # Instrument ID
                    "bar": cls.VALID_INTERVALS[interval],   # Intervalle
                    "after": end_time,  # Temps de début
                    "limit": cls.LIMIT  # Maximum par requête
                }
                async with session.get(endpoint, params=params) as response:
                    data = await response.json()

                    if "data" in data:
                        if not len(data["data"]) or data == old_data:
                            break
                        klines.extend(data["data"])

                        # Avance le start_time à la fin de la dernière chandelle récupérée
                        last_candle_time = int(data["data"][-1][0])  # Timestamp de la dernière chandelle
                        end_time = last_candle_time - 1
                        old_data = data
                        await asyncio.sleep(0.2)  # Petite pause pour éviter les limitations d'API
                    else:
                        print(data, "Error, retrying...")
                        await asyncio.sleep(5)  # Pause plus longue en cas d'erreur

            # Filtrer les chandelles avec un timestamp inférieur à start_time
            klines = [kline for kline in klines if int(kline[0]) >= start_time]
            return cls.process_klines(klines)

    @classmethod
    def get_available_trading_pairs(cls):
        """
        Récupère la liste des paires de trading disponibles sur OKX.
        """
        base_url = f"{cls.BASE_REST_URL}{cls.SYMBOL_URL}"
        params = {"instType": "SPOT"}  # Exemple pour récupérer uniquement les instruments Spot
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                return [instrument["instId"] for instrument in data["data"]]
            else:
                raise Exception("OKX API error: No data found in response.")
        else:
            raise Exception(f"OKX API error: {response.status_code} - {response.text}")

    # async def subscribe_order_book(self, symbol: str, callback):
    #     """
    #     Connect to OKX’s WebSocket and subscribe to order book updates for a given symbol.
    #     The callback is called with standardized order book data.
        
    #     :param symbol: Trading pair symbol (e.g., 'BTC-USDT')
    #     :param callback: A function to be called with the standardized order book dict.
    #     """
    #     # OKX WebSocket public endpoint
    #     ws_endpoint = "wss://ws.okx.com:8443/ws/v5/public"
    #     # Build a subscription message – here we use the "books5" channel (a 5‐level order book)
    #     subscription_message = {
    #         "op": "subscribe",
    #         "args": [{
    #             "channel": "books5",
    #             "instId": symbol
    #         }]
    #     }
    #     async with aiohttp.ClientSession() as session:
    #         async with session.ws_connect(ws_endpoint) as ws:
    #             # Send the subscription message
    #             await ws.send_json(subscription_message)
    #             async for msg in ws:
    #                 if msg.type == aiohttp.WSMsgType.TEXT:
    #                     data = msg.json()
    #                     # OKX typically sends messages in the format:
    #                     # {
    #                     #   "arg": {"channel": "books5", "instId": "BTC-USDT"},
    #                     #   "data": [{
    #                     #         "bids": [["9999", "0.2", "1"], ...],
    #                     #         "asks": [["10000", "0.1", "1"], ...],
    #                     #         "ts": "1578969180502"
    #                     #     }]
    #                     # }
    #                     if "data" in data:
    #                         order_data = data["data"][0]
    #                         standardized = {
    #                             "bids": [[float(item[0]), float(item[1])] for item in order_data.get("bids", [])],
    #                             "asks": [[float(item[0]), float(item[1])] for item in order_data.get("asks", [])],
    #                             "timestamp": order_data.get("ts")
    #                         }
    #                         callback(standardized)
    #                 elif msg.type == aiohttp.WSMsgType.ERROR:
    #                     print("WebSocket error on OKX")
    #                     break