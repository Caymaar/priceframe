from abc import ABC, abstractmethod

class Exchange(ABC):
    @abstractmethod
    async def get_historical_klines(self, symbol, interval, start_date=None, end_date=None):
        pass

    @abstractmethod
    def get_available_trading_pairs(self):
        pass