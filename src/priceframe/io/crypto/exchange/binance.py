from .abstract import Exchange
import requests
import asyncio
import aiohttp
from datetime import datetime

import logging

logger = logging.getLogger(__name__)

class Binance(Exchange):

    name = "binance"
    
    BASE_REST_SPOT_URL = "https://api.binance.com"
    KLINE_URL = "/api/v3/klines"
    SYMBOLE_URL = "/api/v3/exchangeInfo"
    LIMIT = 1000
    VALID_INTERVALS = {
                "1m": 60,
                "3m": 180,
                "5m": 300,
                "15m": 900,
                "30m": 1800,
                "1h": 3600,
                "2h": 7200,
                "4h": 14400,
                "6h": 21600,
                "8h": 28800,
                "12h": 43200,
                "1d": 86400,
                "3d": 259200,
                "1w": 604800,
                "1M": 2592000
            }
    def __init__(self):


        self.valid_intervals = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
            "3d": 259200,
            "1w": 604800,
            "1M": 2592000
        }

    @staticmethod
    def process_klines(klines):
        """
        Convertit les données
        """
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

        :param symbol: Symbole de trading (ex: 'BTCUSDT').
        :param interval: Intervalle des chandelles (ex: '1m', '5m', '1h', '1d').
        :param start_time: Timestamp Unix (ms) de début.
        :param end_time: Timestamp Unix (ms) de fin.
        :param perpetual: Booléen indiquant si on utilise les données futures perpétuelles (par défaut: False).
        :return: DataFrame des chandelles.
        """
        if "-" in symbol:
            symbol = symbol.replace("-", "")

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            endpoint = f"{cls.BASE_REST_SPOT_URL}{cls.KLINE_URL}"
            klines = []
            while start_time < end_time:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": cls.LIMIT
                }
                async with session.get(endpoint, params=params) as response:
                    data = await response.json()
                    if isinstance(data, list):
                        if not len(data):
                            break
                        klines.extend(data)

                        # Avance le start_time à la fin de la dernière chandelle récupérée
                        start_time = data[-1][0] + 1
                        await asyncio.sleep(0.1)  # Petite pause pour éviter les limitations d'API
                    else:
                        logger.error(f"Error, retrying...: {data}")
                        await asyncio.sleep(5)  # Pause plus longue en cas d'erreur

            return cls.process_klines(klines)
    
    @classmethod
    def get_available_trading_pairs(cls):
        base_url = cls.BASE_REST_SPOT_URL + cls.SYMBOLE_URL
        response = requests.get(base_url)
        if response.status_code == 200:
            data = response.json()
            return [symbol_info['symbol'] for symbol_info in data['symbols']]
        else:
            raise Exception(f"Binance API error: {response.status_code} - {response.text}")

    # async def subscribe_order_book(self, symbol: str, callback):
    #     """
    #     Se connecte au WebSocket Binance pour le symbol donné (ex: BTCUSDT@depth10)
    #     et envoie les données du carnet d'ordres au callback jusqu'à ce qu'on 
    #     appelle unsubscribe_order_book(symbol).
    #     """
    #     # On génère le stream
    #     stream = f"{symbol.lower()}@depth10"
    #     url = f"{self.ws_url}/{stream}"

    #     # On crée l'Event s'il n'existe pas
    #     if symbol not in self.stop_events:
    #         self.stop_events[symbol] = asyncio.Event()
    #     stop_event = self.stop_events[symbol]
    #     stop_event.clear()  # On s'assure qu'il n'est pas déjà set

    #     # Connexion WS
    #     async with aiohttp.ClientSession() as session:
    #         async with session.ws_connect(url) as ws:
    #             print(f"[Binance] Subscribed to {symbol}@depth10")
    #             async for msg in ws:
    #                 # Check si on doit arrêter
    #                 if stop_event.is_set():
    #                     print(f"[Binance] Unsubscribing from {symbol}@depth10")
    #                     break

    #                 if msg.type == aiohttp.WSMsgType.TEXT:
    #                     data = msg.json()
    #                     # Première réception : snapshot
    #                     if "lastUpdateId" in data:
    #                         standardized = {
    #                             "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
    #                             "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
    #                             "timestamp": None
    #                         }
    #                         callback(standardized)
    #                     # Mises à jour incrémentielles (depthUpdate)
    #                     elif data.get("e") == "depthUpdate":
    #                         standardized = {
    #                             "bids": [[float(p), float(q)] for p, q in data.get("b", [])],
    #                             "asks": [[float(p), float(q)] for p, q in data.get("a", [])],
    #                             "timestamp": data.get("E")
    #                         }
    #                         callback(standardized)
    #                 elif msg.type == aiohttp.WSMsgType.ERROR:
    #                     print("[Binance] WebSocket error on", symbol)
    #                     break

    #             print(f"[Binance] Connection closed for {symbol}@depth10")

    # def unsubscribe_order_book(self, symbol: str):
    #     """
    #     Déclenche l'arrêt de la boucle d'écoute 
    #     (si subscribe_order_book tourne encore pour ce symbol).
    #     """
    #     if symbol in self.stop_events:
    #         self.stop_events[symbol].set()
    #     else:
    #         print(f"[Binance] No active subscription to unsubscribe for {symbol}")