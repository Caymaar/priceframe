from datetime import datetime
from typing import Union
from priceframe.core import PriceFrame

try:
    from xbbg import blp
except ImportError or ModuleNotFoundError:
    raise ImportError("xbbg et blpapi sont requis pour utiliser le module priceframe.io.bloomberg. vous pouvez les installer via pip install priceframe[bloomberg]")

OHLVC_FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]
CLOSE_FIELD = ["PX_LAST"]
MAPPING_FIELDS = {
    "PX_OPEN": "open",
    "PX_HIGH": "high",
    "PX_LOW": "low",
    "PX_LAST": "close",
    "PX_VOLUME": "volume"
}

BLOOMBERG_DATE_FORMAT = "%Y-%m-%d"

def _bloomberg_request(tickers: Union[str, list[str]],
                       start_date: Union[datetime, str] = None,
                       end_date: Union[datetime, str] = None,
                       interval: str = None,
                       type_: str = "ohlcv", # 'close' | 'ohlcv' | 'ohlc'
                       currency: str = None, 
                       **kwargs) -> PriceFrame:
    """
    Effectue une requête Bloomberg via xbbg et retourne un PriceFrame.
    """

    

    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(start_date, datetime):
        start_date = start_date.strftime(BLOOMBERG_DATE_FORMAT)
    if isinstance(end_date, datetime):
        end_date = end_date.strftime(BLOOMBERG_DATE_FORMAT)
    if end_date is None:
        end_date = datetime.today().strftime(BLOOMBERG_DATE_FORMAT)

    fields = []

    if type_ == "close":
        fields = CLOSE_FIELD
    else:
        for s in type_:
            for key, value in MAPPING_FIELDS.items():
                if s == value[0].lower():
                    fields.append(key)

    if fields == []:
        fields = OHLVC_FIELDS

    if currency is not None:
        kwargs["CURRENCY"] = currency

    results = blp.bdh(
        tickers=tickers,
        flds=fields,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )

    # Ajuster le DataFrame pour correspondre au format attendu par PriceFrame
    df = results.stack(level=0).rename_axis(index=['ts', 'symbol']).reset_index()
    for col in df.columns[2:]:
        if col in MAPPING_FIELDS:
            df = df.rename(columns={col: MAPPING_FIELDS[col]})

    if interval is not None and interval != "1d":
        raise NotImplementedError("Seul l'intervalle '1d' est supporté pour le moment.")

    if interval is not None:
        df["interval"] = interval

    df = df[["ts", "symbol"] + ([ "interval"] if "interval" in df.columns else []) + list(c for c in MAPPING_FIELDS.values() if c in df.columns)]

    return PriceFrame.from_pandas(df, impute_ohlcv=False, interval=interval, source="bloomberg")
