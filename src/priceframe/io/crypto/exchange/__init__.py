from .binance import Binance
from .okx import OKX
from .coinbase_pro import CoinbasePro
from .kraken import Kraken

__all__ = ["Binance", "OKX", "CoinbasePro", "Kraken"]

exchange_dict = {
    "binance": Binance(),
    "okx": OKX(),
    "coinbase_pro": CoinbasePro(),
    "kraken": Kraken()
}