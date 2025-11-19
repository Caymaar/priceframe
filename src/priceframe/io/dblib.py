from datetime import datetime
from typing import Union
from priceframe.core import PriceFrame

try:
    import DBLib as db
    from DBLib import __regions__
except ImportError:
    raise ImportError("DBLib est requis pour utiliser le module priceframe.io.dblib. vous pouvez l'installer via pip install priceframe[themateam] (Réservé aux utilisateurs de Themateam).")

ID_TO_REGION_DICT = {}
for region in __regions__:
    print(region)
    db.DBConnection(region)
    infos = db.Get_INFOS(region, "EQUITY", listfields=["ID"])
    ID_TO_REGION_DICT[region] = list(infos['ID'].unique())
db.DBConnection.close_all()

def _dblib_request(ids: Union[str, list[str]],
                   start_date: Union[datetime, str] = None,
                   end_date: Union[datetime, str] = None,
                   currency: str = None
                   ) -> PriceFrame:
    """
    Effectue une requête via DBLib et retourne un PriceFrame.
    """

    if isinstance(ids, str):
        ids = [ids]

    results = None
    for region, id_list in ID_TO_REGION_DICT.items():
        ids_in_region = [i for i in ids if i in id_list]
        if len(ids_in_region) == 0:
            continue

        if currency is not None:
            currency = currency.upper()
            addfx = True
        else:
            addfx = False

        db.DBConnection(region)
        df_region = db.Get_OHLC_DATA(
            region,
            "EQUITY",
            ids_in_region,
            datestart=start_date,
            dateend=end_date,
            addfx=addfx,
            refcurrency="EUR" if currency is None else currency
        )
       

        if df_region is None or len(df_region) == 0:
            continue
        else:
            df_region = df_region.reset_index()


        df_region = df_region.rename(columns={
            "DATE": "ts",
            "ID": "symbol",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume"
        })

        print(df_region.columns)

        if "FX" in df_region.columns:
            print("Converting prices using FX rates...")
            for col in ["open", "high", "low", "close"]:
                df_region[col] = df_region[col] / df_region["FX"]

            df_region = df_region.drop(columns=["FX", "CURRENCY"])
            
        df_region["interval"] = "1d"

        if results is None:
            results = PriceFrame.from_pandas(df_region, interval="1d", impute_ohlcv=False, source=region, quote_ccy=currency)
        else:
            results = results.add(PriceFrame.from_pandas(df_region, interval="1d", impute_ohlcv=False, source=region, quote_ccy=currency))
    db.DBConnection.close_all()
    return results
