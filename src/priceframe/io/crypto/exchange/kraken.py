from .abstract import Exchange
import requests
import asyncio
import aiohttp
from datetime import datetime

class Kraken(Exchange):
    name = "kraken"
    BASE_REST_URL = "https://api.kraken.com"
    KLINE_URL = "/0/public/OHLC"
    SYMBOL_URL = "/0/public/AssetPairs"
    LIMIT = 720  # Limite maximale par requête selon la documentation de Kraken
    VALID_INTERVALS = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
        "1M": 21600  # Attention : Kraken ne supporte pas forcément "1M" tel quel, adaptez au besoin.
    }

    @staticmethod
    def _process_klines(klines):
        """
        Convertit les données
        """
        return [
            {
                "ts": datetime.utcfromtimestamp(int(kline[0])).strftime('%Y-%m-%d %H:%M:%S'),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[6])
            }
            for kline in klines
        ]

    @classmethod
    def get_available_trading_pairs(cls):
        """
        Récupère la liste des paires de trading disponibles sur Kraken.
        """
        endpoint = f"{cls.BASE_REST_URL}{cls.SYMBOL_URL}"
        response = requests.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            if data.get("error"):
                raise Exception(f"Kraken API error: {data['error']}")
            pairs = []
            for key, value in data["result"].items():
                # Si le champ 'wsname' est présent, on l'utilise pour afficher le nom de la paire (ex: "XBT/USD")
                if "wsname" in value:
                    pairs.append(value["wsname"])
                else:
                    pairs.append(key)
            return pairs
        else:
            raise Exception(f"Kraken API error: {response.status_code} - {response.text}")

    @classmethod
    async def get_historical_klines(cls, symbol, interval, start_time, end_time):
        """
        Récupère les chandelles historiques (OHLC) depuis Kraken.
        
        :param symbol: Paire de trading (ex: 'XBT/USD'). À fournir dans le format attendu par Kraken.
        :param interval: Intervalle des chandelles (ex: '1m', '5m', etc.)
        :param start_time: Date de début en timestamp Unix (ms).
        :param end_time: Date de fin en timestamp Unix (ms).
        :return: Liste de chandelles standardisées.
        """
        if interval not in cls.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Valid intervals are: {', '.join(cls.VALID_INTERVALS.keys())}")
        ohlc_endpoint = f"{cls.BASE_REST_URL}{cls.KLINE_URL}"
        # Kraken attend un paramètre 'since' en secondes
        current_since = int(start_time / 1000)
        all_candles = []
        
        while current_since * 1000 < end_time:
            params = {
                "pair": symbol,
                "interval": cls.VALID_INTERVALS[interval],
                "since": current_since
            }
            response = requests.get(ohlc_endpoint, params=params)
            if response.status_code != 200:
                raise Exception(f"Kraken API error: {response.status_code} - {response.text}")
            data = response.json()
            if data.get("error"):
                raise Exception(f"Kraken API error: {data['error']}")
            result = data.get("result", {})
            # La clé correspondant à la paire peut différer de 'symbol'. On ignore la clé 'last'
            pair_key = None
            for key in result:
                if key != "last":
                    pair_key = key
                    break
            if not pair_key:
                break
            candles = result[pair_key]
            # La valeur 'last' servira pour le prochain appel
            new_since = int(result.get("last", current_since))
            if new_since == current_since:
                break  # Évite la boucle infinie s'il n'y a pas de nouvelles données
            for candle in candles:
                # Format d'une chandelle Kraken : [ time, open, high, low, close, vwap, volume, count ]
                ts = int(candle[0]) * 1000  # conversion en ms
                if ts < start_time or ts > end_time:
                    continue
                standardized = {
                    "ts": datetime.utcfromtimestamp(int(candle[0])).strftime('%Y-%m-%d %H:%M:%S'),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[6])
                }
                all_candles.append(standardized)
            current_since = new_since
            await asyncio.sleep(1)  # Petite pause pour éviter de saturer l'API
        return all_candles

    # async def subscribe_order_book(self, symbol: str, callback):
    #     """
    #     Se connecte au WebSocket de Kraken et s'abonne aux mises à jour du carnet d'ordres pour une paire donnée.
    #     Les mises à jour sont appliquées sur un carnet en mémoire qui met à jour la quantité pour chaque prix.
    #     Si la quantité devient 0, le niveau est supprimé.
    #     Lorsque le message heartbeat est reçu, le carnet (agrégé et trié) est envoyé via callback.
        
    #     :param symbol: Paire de trading (ex: 'XBT/USD')
    #     :param callback: Fonction à appeler avec le dictionnaire de données agrégées.
    #     """
    #     subscription_message = {
    #         "event": "subscribe",
    #         "pair": [symbol],
    #         "subscription": {
    #             "name": "book",
    #             "depth": 10  # Profondeur du carnet, modifiable selon vos besoins
    #         }
    #     }

    #     # Le carnet d'ordres en mémoire :
    #     # On utilise deux dictionnaires pour les bids et asks, avec pour clé le prix (float) et pour valeur la quantité (float)
    #     order_book = {
    #         "bids": {},
    #         "asks": {}
    #     }

    #     async with aiohttp.ClientSession() as session:
    #         async with session.ws_connect(self.ws_url) as ws:
    #             await ws.send_json(subscription_message)
    #             async for msg in ws:
    #                 if msg.type == aiohttp.WSMsgType.TEXT:
    #                     data = msg.json()
    #                     # Si c'est un message événementiel, on vérifie s'il s'agit d'un heartbeat.
    #                     if isinstance(data, dict):
    #                         if data.get("event") == "heartbeat":
    #                             # À la réception du heartbeat, on prépare la donnée agrégée.
    #                             # Pour faciliter la lecture, on convertit les dictionnaires en listes triées.
    #                             bids_list = [[price, qty] for price, qty in order_book["bids"].items()]
    #                             # Tri décroissant pour les bids (les meilleurs prix en premier)
    #                             bids_list.sort(key=lambda x: x[0], reverse=True)
    #                             asks_list = [[price, qty] for price, qty in order_book["asks"].items()]
    #                             # Tri croissant pour les asks
    #                             asks_list.sort(key=lambda x: x[0])
    #                             aggregated = {
    #                                 "bids": bids_list,
    #                                 "asks": asks_list,
    #                                 "timestamp": None  # Vous pouvez ajouter un timestamp ici si besoin
    #                             }
    #                             # Tri décroissant pour les bids (les meilleurs prix en premier)
    #                             aggregated["bids"].sort(key=lambda x: x[0], reverse=True)
    #                             # Tri croissant pour les asks
    #                             aggregated["asks"].sort(key=lambda x: x[0])
    #                             # Prendre 10 niveaux pour chaque côté du carnet
    #                             aggregated["bids"] = aggregated["bids"][:10]
    #                             aggregated["asks"] = aggregated["asks"][:10]

    #                             # Envoie les données agrégées via la callback
    #                             callback(aggregated)
    #                             # On continue sans modifier le carnet (les mises à jour continueront d'être appliquées)
    #                             continue
    #                         # On ignore les autres événements comme "subscriptionStatus"
    #                         elif data.get("event") in ["subscriptionStatus"]:
    #                             continue

    #                     # Les mises à jour du carnet arrivent sous forme de liste
    #                     elif isinstance(data, list):
    #                         if len(data) < 4:
    #                             continue
    #                         update = data[1]
    #                         # Traitement des bids
    #                         if "bs" in update:  # Snapshot initiale
    #                             for item in update["bs"]:
    #                                 price = float(item[0])
    #                                 qty = float(item[1])
    #                                 if qty > 0:
    #                                     order_book["bids"][price] = qty
    #                                 elif price in order_book["bids"]:
    #                                     del order_book["bids"][price]
    #                         elif "b" in update:  # Mises à jour incrémentielles
    #                             for item in update["b"]:
    #                                 price = float(item[0])
    #                                 qty = float(item[1])
    #                                 if qty > 0:
    #                                     order_book["bids"][price] = qty
    #                                 elif price in order_book["bids"]:
    #                                     del order_book["bids"][price]

    #                         # Traitement des asks
    #                         if "as" in update:  # Snapshot initiale
    #                             for item in update["as"]:
    #                                 price = float(item[0])
    #                                 qty = float(item[1])
    #                                 if qty > 0:
    #                                     order_book["asks"][price] = qty
    #                                 elif price in order_book["asks"]:
    #                                     del order_book["asks"][price]
    #                         elif "a" in update:  # Mises à jour incrémentielles
    #                             for item in update["a"]:
    #                                 price = float(item[0])
    #                                 qty = float(item[1])
    #                                 if qty > 0:
    #                                     order_book["asks"][price] = qty
    #                                 elif price in order_book["asks"]:
    #                                     del order_book["asks"][price]
    #                 elif msg.type == aiohttp.WSMsgType.ERROR:
    #                     print("WebSocket error on Kraken")
    #                     break