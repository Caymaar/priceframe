from dataclasses import dataclass
from typing import Optional, Sequence, Literal, Callable, Union

import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
from datetime import datetime

from .features import FeatureSpec, build_feature_fn

# ---------------------------------------------------------------------
# Schéma canonique : OHLCV par (symbol, interval, ts)
# ---------------------------------------------------------------------

CANON_COLS = ["symbol", "interval", "ts", "open", "high", "low", "close", "volume"]

CANON_TYPES: dict[str, pa.DataType] = {
    "symbol": pa.string(),
    "interval": pa.string(),
    "ts": pa.timestamp("ms", tz="UTC"),
    "open": pa.float64(),
    "high": pa.float64(),
    "low": pa.float64(),
    "close": pa.float64(),
    "volume": pa.float64(),
}

SCHEMA_V1 = pa.schema(CANON_TYPES)


# ---------------------------------------------------------------------
# Helpers Timestamps
# ---------------------------------------------------------------------

def _ensure_ts_ms_utc(col: pa.Array) -> pa.Array:
    """
    Cast une colonne 'ts' vers timestamp[ms, tz='UTC'].

    Cas gérés :
    - timestamp naïf -> assume UTC, convert en ms.
    - timestamp avec tz != UTC -> convert en UTC, en ms.
    - numérique epoch (s ou ms) -> cast en UTC ms via heuristique.
    """
    t = col.type

    # 1) Cas timestamp
    if pa.types.is_timestamp(t):
        target = pa.timestamp("ms", tz="UTC")

        # Naïf: assume UTC
        if t.tz is None:
            col_ms = pc.cast(col, pa.timestamp("ms"))
            return pc.assume_timezone(col_ms, "UTC")

        # Déjà tz-aware
        col_tz = col
        if t.tz != "UTC":
            col_tz = pc.astimezone(col, "UTC")

        if t.unit != "ms":
            col_tz = pc.cast(col_tz, target)

        return col_tz

    # 2) Cas numérique (epoch s ou ms)
    if pa.types.is_integer(t) or pa.types.is_floating(t):
        sample = col.slice(0, min(10, len(col)))
        mx = max(sample.to_pylist()) if len(sample) > 0 else 0

        # heuristique : > 1e11 ⇒ ms
        if mx > 10**11:
            return pc.cast(col, pa.timestamp("ms", tz="UTC"))
        else:
            seconds = pc.cast(col, pa.int64())
            ms = pc.multiply(seconds, pc.scalar(1000, pa.int64()))
            return pc.cast(ms, pa.timestamp("ms", tz="UTC"))

    # 3) Fallback
    return pc.cast(col, pa.timestamp("ms", tz="UTC"))


def _cast_to_canon(t: pa.Table) -> pa.Table:
    """
    Applique le schéma canonique :
    - ts -> timestamp[ms, tz='UTC']
    - colonnes OHLCV -> float64 si possible.
    """
    if "ts" not in t.column_names:
        raise KeyError("Colonne 'ts' manquante dans la table.")

    # 1) ts
    ts_idx = t.schema.get_field_index("ts")
    ts_col = t.column("ts")
    ts_canon = _ensure_ts_ms_utc(ts_col)
    t = t.set_column(ts_idx, "ts", ts_canon)

    # 2) autres colonnes canoniques (tolérant)
    for name, target_type in CANON_TYPES.items():
        if name not in t.column_names:
            continue
        current_type = t.schema.field(name).type
        if current_type == target_type:
            continue
        try:
            casted = pc.cast(t.column(name), target_type, safe=False)
            t = t.set_column(t.schema.get_field_index(name), name, casted)
        except Exception:
            # ex: volume en int64 acceptable
            pass

    return t


def _sort_and_dedup(t: pa.Table) -> pa.Table:
    """
    Tri par (symbol, ts) et drop duplicates sur (symbol, interval, ts).
    """
    t = t.sort_by([("symbol", "ascending"), ("ts", "ascending")])

    # PyArrow récent : drop_duplicates
    try:
        t = pc.drop_duplicates(t, keys=["symbol", "interval", "ts"], keep="first")
    except Exception:
        # Fallback pandas
        df = t.to_pandas(types_mapper=pd.ArrowDtype)
        df = df.sort_values(["symbol", "interval", "ts"]).drop_duplicates(
            subset=["symbol", "interval", "ts"], keep="first"
        )
        t = pa.Table.from_pandas(df, preserve_index=False)

    return t


def _normalize_ts_like(x) -> Optional[pd.Timestamp]:
    """
    Convertit start/end entrés en pd.Timestamp UTC.
    """
    if x is None:
        return None
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


# ---------------------------------------------------------------------
# Helpers fréquence / interval
# ---------------------------------------------------------------------

def _interval_str_from_offset(off: pd.DateOffset) -> str:
    """
    Offsets pandas -> notation canonique '1m','1h','1d','1w','1mo','1q','1y', etc.
    """
    from pandas.tseries import offsets as po

    n = getattr(off, "n", 1)

    if isinstance(off, po.Second):
        return f"{n}s"
    if isinstance(off, po.Minute):
        return f"{n}m"
    if isinstance(off, po.Hour):
        return f"{n}h"
    if isinstance(off, po.Day):
        return f"{n}d"
    if isinstance(off, po.Week):
        return f"{n}w"
    if isinstance(
        off,
        (
            po.MonthEnd,
            po.MonthBegin,
            po.BMonthEnd,
            po.BMonthBegin,
            po.SemiMonthBegin,
            po.SemiMonthEnd,
        ),
    ):
        return f"{n}mo"
    if isinstance(
        off,
        (po.QuarterEnd, po.QuarterBegin, po.BQuarterEnd, po.BQuarterBegin),
    ):
        return f"{n}q"
    if isinstance(
        off,
        (po.YearEnd, po.YearBegin, po.BYearEnd, po.BYearBegin),
    ):
        return f"{n}y"

    return getattr(off, "name", str(off))


def infer_interval_from_index(index: pd.DatetimeIndex) -> tuple[pd.DateOffset, str]:
    """
    Infère la fréquence d'un DatetimeIndex :
    - retourne (offset pandas, 'interval_str' canonique).
    """
    from pandas.tseries.frequencies import to_offset

    if len(index) < 2:
        raise ValueError("Index trop court pour inférer une fréquence.")

    f = pd.infer_freq(index)
    if f is None:
        diffs = pd.Series(index[1:] - index[:-1]).dropna()
        if len(diffs) == 0:
            raise ValueError("Impossible d'inférer la fréquence (un seul point).")
        med = diffs.median()

        # heuristiques super simples
        if med.components.days >= 365:
            f = "A"
        elif med.components.days >= 90:
            f = "Q"
        elif med.components.days >= 28:
            f = "M"
        else:
            s = med.total_seconds()
            if abs(s - 1) < 1e-6:
                f = "S"
            elif abs(s - 5) < 1e-6:
                f = "5S"
            elif abs(s - 10) < 1e-6:
                f = "10S"
            elif abs(s - 15) < 1e-6:
                f = "15S"
            elif abs(s - 30) < 1e-6:
                f = "30S"
            elif abs(s - 60) < 1e-6:
                f = "T"
            elif abs(s - 300) < 1e-6:
                f = "5T"
            elif abs(s - 600) < 1e-6:
                f = "10T"
            elif abs(s - 900) < 1e-6:
                f = "15T"
            elif abs(s - 1800) < 1e-6:
                f = "30T"
            elif abs(s - 3600) < 1e-6:
                f = "H"
            elif abs(s - 86400) < 1e-6:
                f = "D"
            elif abs(s - 604800) < 1e-6:
                f = "W"
            else:
                raise ValueError(
                    f"Impossible d'inférer la fréquence (delta médian={med})."
                )

    off = to_offset(f)
    interval_str = _interval_str_from_offset(off)
    return off, interval_str


def _parse_interval(interval: str) -> tuple[int, str]:
    """
    '1m','5m','1h','1d','1w','1mo','1q','1y' -> (1,'m') etc.
    """
    m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", interval.strip())
    if not m:
        raise ValueError(f"Interval invalide: {interval!r}")
    n = int(m.group(1))
    u = m.group(2).lower()
    return n, u


def interval_to_offset(interval: str) -> tuple[pd.DateOffset, str]:
    """
    Convertit une string d'interval utilisateur vers (offset pandas, interval_str canonique).
    Accepte formats comme '1m','5m','1h','1d','1w','1mo','1q','1y'.
    """
    from pandas.tseries.frequencies import to_offset

    unit_map = {
        "s": "S",
        "m": "T",
        "h": "H",
        "d": "D",
        "w": "W",
        "mo": "M",
        "q": "Q",
        "y": "A",
    }

    try:
        n, u = _parse_interval(interval)
    except ValueError:
        # fallback : laisser pandas interpréter directement
        off = to_offset(interval)
        return off, _interval_str_from_offset(off)

    if u not in unit_map:
        off = to_offset(interval)
        return off, _interval_str_from_offset(off)

    pandas_code = unit_map[u]
    off = to_offset(f"{n}{pandas_code}")
    return off, _interval_str_from_offset(off)


def _open_shift_from_offset(off: pd.DateOffset) -> pd.DateOffset:
    """
    Si l'index représente le close, de combien on décale vers l'open ?
    Pour des offsets journaliers / horaires etc. : off lui-même.
    Pour MonthEnd/QuarterEnd/YearEnd : shift au début de période suivante.
    """
    from pandas.tseries import offsets as po

    if isinstance(off, (po.MonthEnd, po.BMonthEnd, po.SemiMonthEnd)):
        return po.MonthBegin(1)
    if isinstance(off, (po.QuarterEnd, po.BQuarterEnd)):
        return po.QuarterBegin(1, startingMonth=getattr(off, "startingMonth", 1))
    if isinstance(off, (po.YearEnd, po.BYearEnd)):
        return po.YearBegin(1)
    return off


# ---------------------------------------------------------------------
# Type pour les fonctions de features
# ---------------------------------------------------------------------

FeatureFn = Callable[[pl.DataFrame], pl.DataFrame]


# ---------------------------------------------------------------------
# Dataclass principale
# ---------------------------------------------------------------------

@dataclass
class PriceFrame:
    """
    Stockage canonique de prix :
    - une ligne = (symbol, interval, ts, OHLCV, colonnes extra éventuelles)
    - ts en UTC millisecondes
    - 'interval' : string canonique (ex: '1m','1h','1d','1w','1mo','1q','1y')

    Invariant recommandé : un PriceFrame = UNE seule fréquence
    (les méthodes temporelles supposent un interval unique et appellent
    ensure_single_interval()).
    """

    table: pa.Table
    schema_version: str = "ohlcv_v1"

    # -----------------------------------------------------------------
    # Constructeurs
    # -----------------------------------------------------------------
    @classmethod
    def from_pandas(cls, df: pd.DataFrame, 
                    interval: str = None, 
                    impute_ohlcv: bool = False,
                    source: Union[str, None] = None,
                    quote_ccy: Union[str, None] = None
                    ) -> 'PriceFrame':
        """
        Constructeur à partir d'un DataFrame long (symbol / interval / ts / OHLCV).
        Normalise les noms, types, tz, tri, dedup.
        """
        rename = {
            "date": "ts",
            "Date": "ts",
            "DATE": "ts",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Ajouter colonnes manquantes
        for c in [c for c in CANON_COLS if c not in df.columns]:
            if c in ["open", "high", "low", "close", "volume"] and impute_ohlcv:
                if c == "volume":
                    df[c] = 0.0
                else:
                    df[c] = df["close"].copy()
            elif c in ["open", "high", "low", "close", "volume"] and not impute_ohlcv:
                df[c] = np.nan

        # ts -> tz-aware UTC
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        # inférer interval si absent
        if interval is None and "interval" not in df.columns:
            # créer un DatetimeIndex sans doublons
            idx = pd.DatetimeIndex(sorted(df["ts"].unique()))
            _, interval_str = infer_interval_from_index(idx)
            df["interval"] = interval_str
        elif interval is not None and "interval" not in df.columns:
            _, interval_str = interval_to_offset(interval)
            df["interval"] = interval_str
        
        # numériques OHLCV
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # tri + dedup
        df = (
            df.sort_values(["symbol", "interval", "ts"])
            .drop_duplicates(subset=["symbol", "interval", "ts"])
        )
        if source is not None:
            df["source"] = source
        if quote_ccy is not None:
            df["quote_ccy"] = quote_ccy

        t = pa.Table.from_pandas(df, preserve_index=False)
        t = _cast_to_canon(t)
        t = _sort_and_dedup(t)
        return cls(t)

    @classmethod
    def from_polars(cls, pl_df: pl.DataFrame) -> 'PriceFrame':
        return cls.from_arrow(pl_df.to_arrow())

    @classmethod
    def from_arrow(cls, table: pa.Table) -> 'PriceFrame':
        """
        Construction à partir d'une table Arrow déjà en long-format.
        """
        rename_map = {
            "open_time": "ts",
            "Open time": "ts",
            "number_of_trades": "trades",
        }
        names = list(table.column_names)
        for old, new in rename_map.items():
            if old in names:
                names = [new if n == old else n for n in names]
        if names != table.column_names:
            table = table.rename_columns(names)

        missing = [c for c in CANON_COLS if c not in table.column_names]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        table = _cast_to_canon(table)
        table = _sort_and_dedup(table)
        return cls(table)

    # --- IO externes : à implémenter dans des modules dédiés (bloomberg, binance, ...) ---

    @classmethod
    def from_bloomberg(cls, 
                       tickers: Union[str, list[str]],
                       start_date: Union[datetime, str] = None,
                       end_date: Union[datetime, str] = None,
                       interval: str = "1d",
                       type_: str = "ohlcv", # 'close' | 'ohlcv' | 'ohlc'
                       currency: str = None, 
                       **kwargs) -> 'PriceFrame':
        """
        Stub : à implémenter dans un module séparé (ex: priceframe.io.bloomberg).
        """
        from .io.bloomberg import _bloomberg_request
        return _bloomberg_request(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            type_=type_,
            currency=currency,
            **kwargs
        )
    
    @classmethod
    def from_dblib(cls,
                   ids: Union[str, list[str]],
                   start_date: Union[datetime, str] = None,
                   end_date: Union[datetime, str] = None,
                   currency: str = None
                   ) -> 'PriceFrame':
        """
        Stub : à implémenter dans un module séparé (ex: priceframe.io.dblib).
        """        
        from .io.dblib import _dblib_request
        return _dblib_request(
            ids=ids,
            start_date=start_date,
            end_date=end_date,
            currency=currency
        )

    @classmethod
    def from_binance(cls, *args, **kwargs) -> 'PriceFrame':
        """
        Stub : à implémenter dans un module séparé (ex: priceframe.io.binance).
        """
        raise NotImplementedError("Utiliser un module IO dédié (ex: priceframe.io.binance).")

    # -----------------------------------------------------------------
    # Conversions
    # -----------------------------------------------------------------

    def to_arrow(self, columns: Optional[Sequence[str]] = None) -> pa.Table:
        return self.table.select(columns) if columns else self.table

    def to_polars(self) -> pl.DataFrame:
        return pl.from_arrow(self.table)

    def to_pandas(
        self,
        index: Optional[Literal["ts", "multi"]] = None,
        *,
        use_arrow_dtypes: bool = False,
    ) -> pd.DataFrame:
        """
        Conversion vers pandas.

        - use_arrow_dtypes=False (défaut) :
            DataFrame pandas "classique" (pas de dtypes [pyarrow]).
            * ts        -> datetime64[ns, UTC]
            * floats    -> float64
            * ints      -> Int64 (entiers nullable)
            * bool      -> bool
            * string    -> object

        - use_arrow_dtypes=True :
            renvoie les dtypes Arrow tels que produits par pyarrow.Table.to_pandas().
        """
        # 1) DataFrame brut issu d'Arrow
        raw = self.table.to_pandas()  # donne souvent des ArrowDtype en pandas 2.x

        if use_arrow_dtypes:
            df = raw
        else:
            df = raw.copy()

            for col in df.columns:
                dtype = df[col].dtype
                pa_type = getattr(dtype, "pyarrow_dtype", None)

                # Si pas un ArrowDtype, on laisse tel quel
                if pa_type is None:
                    continue

                # --- Cas particulier: ts ---
                if col == "ts":
                    # Forcer en datetime64[ns, UTC]
                    df[col] = pd.to_datetime(df[col], utc=True)
                    continue

                # --- Mapping générique Arrow -> pandas ---
                if pa.types.is_floating(pa_type):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                elif pa.types.is_integer(pa_type):
                    # Si tu préfères tout en float64, remplace par float64 ici
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif pa.types.is_boolean(pa_type):
                    df[col] = df[col].astype("bool")
                elif pa.types.is_string(pa_type):
                    df[col] = df[col].astype("object")
                else:
                    # Fallback : on bascule en object
                    df[col] = df[col].astype("object")

        # 2) Gestion de l'index
        if index == "ts":
            df = df.set_index("ts").sort_index()
        elif index == "multi":
            df = df.set_index(["symbol", "ts"]).sort_index()

        return df
    
    # def to_pandas(
    #     self,
    #     index: Optional[Literal["ts", "multi"]] = None,
    #     *,
    #     use_arrow_dtypes: bool = False,
    # ) -> pd.DataFrame:
    #     if use_arrow_dtypes:
    #         df = self.table.to_pandas(types_mapper=pd.ArrowDtype)
    #     else:
    #         df = self.table.to_pandas()

    #     if index == "ts":
    #         df = df.set_index("ts").sort_index()
    #         df.index = pd.to_datetime(df.index)
    #     elif index == "multi":
    #         df = df.set_index(["symbol", "ts"]).sort_index()

    #     return df

    # -----------------------------------------------------------------
    # Invariants & utilitaires de base
    # -----------------------------------------------------------------

    def ensure_single_interval(self) -> str:
        """
        Vérifie que la colonne 'interval' a une seule valeur.
        Retourne cette valeur.
        """
        uniques = pc.unique(self.table["interval"]).to_pylist()
        if len(uniques) != 1:
            raise ValueError(f"PriceFrame multi-fréquence non supporté pour cette opération: {uniques}")
        return uniques[0]

    def infer_interval(self) -> str:
        """
        Infère la fréquence à partir de ts, en supposant mono-fréquence.
        """
        df = self.to_pandas(index="ts")
        off, interval_str = infer_interval_from_index(df.index)
        _ = off  # pas utilisé pour l'instant, mais dispo si besoin
        return interval_str

    # -----------------------------------------------------------------
    # Accesseurs
    # -----------------------------------------------------------------

    def for_symbol(self, symbol: str) -> 'PriceFrame':
        mask = pc.equal(self.table["symbol"], pa.scalar(symbol))
        t = self.table.filter(mask)
        return PriceFrame(t, schema_version=self.schema_version)

    def only_close(self) -> 'PriceFrame':
        cols = [c for c in self.table.column_names if c in {"symbol", "interval", "ts", "close"}]
        return PriceFrame(self.table.select(cols), schema_version=self.schema_version)

    def feature_columns(self, prefix: str = "f_") -> list[str]:
        """
        Retourne les colonnes dont le nom commence par `prefix` (features).
        """
        return [c for c in self.table.column_names if c.startswith(prefix)]

    # -----------------------------------------------------------------
    # API Features / indicateurs
    # -----------------------------------------------------------------

    def with_features(self, fns: Sequence[FeatureFn]) -> "PriceFrame":
        """
        Applique une liste de fonctions de features Polars :
        chaque fn: pl.DataFrame -> pl.DataFrame (ajoute/modifie des colonnes).
        """
        pl_df = self.to_polars().sort(["symbol", "ts"])
        for fn in fns:
            pl_df = fn(pl_df)
        return PriceFrame.from_polars(pl_df)

    def with_indicators(self, *fns: FeatureFn) -> "PriceFrame":
        """
        Alias pour with_features, usage plus "finance".
        """
        return self.with_features(fns)

    def with_feature_specs(self, specs: Sequence[FeatureSpec]) -> "PriceFrame":
        """
        Applique une liste de FeatureSpec déclaratives sur ce PriceFrame.
        """
        fns = [build_feature_fn(spec) for spec in specs]
        return self.with_features(fns)
    # -----------------------------------------------------------------
    # Data access high-level
    # -----------------------------------------------------------------

    def ohlcv(self, symbol: str, engine: str = "pandas") -> Union[pd.DataFrame, pl.DataFrame]:
        pf = self.for_symbol(symbol)
        if engine == "pandas":
            df = pf.to_pandas(index="ts")
            return df[["open", "high", "low", "close", "volume"]]
        elif engine == "polars":
            pl_df = (
                pf.to_polars()
                .select(["ts", "open", "high", "low", "close", "volume"])
                .sort("ts")
            )
            return pl_df
        else:
            raise ValueError("engine ∈ {'pandas','polars'}")

    def close_series(self, symbol: str, engine: str = "pandas"):
        pf = self.for_symbol(symbol).only_close()
        if engine == "pandas":
            s = pf.to_pandas(index="ts")["close"]
            s.name = symbol
            return s
        elif engine == "polars":
            pl_df = pf.to_polars().select(["ts", "close"]).sort("ts")
            return pl_df.rename({"close": symbol})
        else:
            raise ValueError("engine ∈ {'pandas','polars'}")

    def close_matrix(
        self,
        symbols: Optional[list[str]] = None,
        how: str = "inner",        # 'inner' ou 'outer'
        fill: Union[float, None] = None,
        engine: str = "pandas",
    ):
        if engine == "pandas":
            df = self.to_pandas()  # long
            piv = df.pivot_table(
                index="ts",
                columns="symbol",
                values="close",
                aggfunc="first",
            )
            if symbols is not None:
                piv = piv[symbols]
            if how == "inner":
                piv = piv.dropna(how="any")
            if fill is not None:
                piv = piv.fillna(fill)
            piv.index = pd.to_datetime(piv.index)
            return piv.sort_index()

        elif engine == "polars":
            pl_df = self.to_polars().select(["ts", "symbol", "close"])
            wide = pl_df.pivot(
                values="close",
                index="ts",
                columns="symbol",
                aggregate_function="first",
            ).sort("ts")
            if symbols is not None:
                wide = wide.select(["ts"] + symbols)
            if how == "inner":
                wide = wide.drop_nulls()
            if fill is not None:
                wide = wide.fill_null(fill)
            return wide

        else:
            raise ValueError("engine ∈ {'pandas','polars'}")

    # -----------------------------------------------------------------
    # Filtres temporels
    # -----------------------------------------------------------------

    def range_date(
        self,
        start=None,
        end=None,
        *,
        closed: str = "both",  # "both" | "left" | "right" | "neither"
    ) -> 'PriceFrame':
        """
        Filtre sur [start, end] selon `closed`.
        """
        t = self.table

        s_ts = _normalize_ts_like(start)
        e_ts = _normalize_ts_like(end)

        conds = []

        if s_ts is not None:
            s_scalar = pa.scalar(s_ts.to_pydatetime(), type=pa.timestamp("ms", tz="UTC"))
            op = pc.greater_equal if closed in ("both", "left") else pc.greater
            conds.append(op(t["ts"], s_scalar))

        if e_ts is not None:
            e_scalar = pa.scalar(e_ts.to_pydatetime(), type=pa.timestamp("ms", tz="UTC"))
            op = pc.less_equal if closed in ("both", "right") else pc.less
            conds.append(op(t["ts"], e_scalar))

        if conds:
            mask = conds[0]
            for c in conds[1:]:
                mask = pc.and_(mask, c)
            t2 = t.filter(mask)
        else:
            t2 = t

        return PriceFrame(t2, schema_version=self.schema_version)

    # -----------------------------------------------------------------
    # Rebase & resample
    # -----------------------------------------------------------------

    def rebase(
        self,
        base: float = 100.0,
        *,
        anchor: str = "close",
        cols: tuple[str, ...] = ("open", "high", "low", "close"),
    ) -> 'PriceFrame':
        """
        Rebase OHLC par symbole :
        - 1er anchor non-NaN = base
        - les colonnes `cols` sont rescalées par le même facteur.
        """
        df = self.to_pandas().sort_values(["symbol", "ts"]).reset_index(drop=True)

        if anchor not in df.columns:
            raise KeyError(f"Colonne d'ancrage '{anchor}' absente.")

        for c in cols:
            if c not in df.columns:
                raise KeyError(f"Colonne '{c}' absente.")

        # Supprimer lignes avant premier anchor non-NaN par symbole
        has_obs_cum = (
            df.groupby("symbol")[anchor]
            .apply(lambda s: s.notna().cumsum() > 0)
            .reset_index(level=0, drop=True)
        )
        df = df[has_obs_cum].copy()

        # Facteur d'échelle par symbole
        first_anchor = df.groupby("symbol")[anchor].transform("first")
        scale = (float(base) / first_anchor).astype(float)

        for c in cols:
            df[c] = df[c] * scale

        return PriceFrame.from_pandas(df)

    def resample(
        self,
        target_interval: str,
        *,
        how: Literal["ohlcv"] = "ohlcv",
    ) -> "PriceFrame":
        """
        Resample OHLCV vers un intervalle plus large (downsampling) par symbole.

        Exemples valides :
        - 1d -> 1w
        - 1d -> 1mo
        - 1h -> 1d

        L'upsampling (ex: 1w -> 1d) est rejeté.

        Paramètres
        ----------
        target_interval : str
            Ex: "1w","1d","1mo","1q","1y","1h","5m",...
        how : "ohlcv"
            Pour l'instant, on ne supporte que l'agrégation OHLCV standard :
            open = first, high = max, low = min, close = last, volume = sum.
        """
        if how != "ohlcv":
            raise NotImplementedError("Pour l'instant, resample() supporte seulement how='ohlcv'.")

        # ---------- 1) Vérifier que le PriceFrame est mono-interval ----------
        t = self.table
        intervals = set(t.column("interval").to_pylist())
        if len(intervals) != 1:
            raise ValueError(f"resample() attend un PriceFrame mono-interval, trouvé {intervals}")
        current_interval = next(iter(intervals))

        # ---------- 2) Helpers pour parser et comparer les intervalles ----------
        def _parse_interval(s: str):
            """
            Transforme '1d','5m','1w','1mo','1q','1y' en (n, unit).
            """
            m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", s.strip())
            if not m:
                raise ValueError(f"Interval invalide: {s!r}")
            n = int(m.group(1))
            u = m.group(2).lower()
            return n, u

        # Approximate duration (en secondes) par unité
        unit_scale = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 7 * 86400,
            "mo": 30 * 86400,   # approx
            "q": 90 * 86400,    # approx
            "y": 365 * 86400,   # approx
        }

        def _duration_seconds(interval_str: str) -> float:
            n, u = _parse_interval(interval_str)
            if u not in unit_scale:
                raise ValueError(f"Unité d'intervalle inconnue: {u!r}")
            return n * unit_scale[u]

        # Interdit l'upsampling : target doit être plus large ou égal à current
        cur_dur = _duration_seconds(current_interval)
        tgt_dur = _duration_seconds(target_interval)
        if tgt_dur < cur_dur:
            raise ValueError(
                f"Upsampling non supporté: current={current_interval}, target={target_interval}"
            )

        # ---------- 3) Mapping vers une fréquence pandas ----------
        def _to_pandas_freq(interval_str: str) -> str:
            n, u = _parse_interval(interval_str)
            # mapping vers codes pandas
            mapping = {
                "s": "S",
                "m": "T",
                "h": "H",
                "d": "D",
                "w": "W",
                "mo": "M",
                "q": "Q",
                "y": "A",
            }
            if u not in mapping:
                raise ValueError(f"Unité d'intervalle inconnue: {u!r}")
            return f"{n}{mapping[u]}"

        pandas_freq = _to_pandas_freq(target_interval)

        # ---------- 4) Passage en pandas pour l'agrégation ----------
        df = self.to_pandas()
        # index = ts pour pouvoir resample
        df = df.set_index("ts").sort_index()

        result_frames = []

        # on resample par symbole
        for sym, g in df.groupby("symbol", sort=False):
            rs = g.resample(pandas_freq)

            # agrégation OHLCV classique
            agg = pd.DataFrame(
                {
                    "open": rs["open"].first(),
                    "high": rs["high"].max(),
                    "low": rs["low"].min(),
                    "close": rs["close"].last(),
                    "volume": rs["volume"].sum(),
                }
            )

            # on enlève les lignes vides (par ex. si aucune donnée dans une fenêtre)
            agg = agg.dropna(subset=["open", "high", "low", "close"], how="all")

            if agg.empty:
                continue

            agg["symbol"] = sym
            agg["interval"] = target_interval

            # remettre ts en colonne
            agg = agg.reset_index()  # ts redevient une colonne
            result_frames.append(agg)

        # ---------- 5) Retour en PriceFrame ----------
        if not result_frames:
            empty = pd.DataFrame(
                columns=["symbol", "interval", "ts", "open", "high", "low", "close", "volume"]
            )
            return PriceFrame.from_pandas(empty)

        out_df = pd.concat(result_frames, ignore_index=True)
        return PriceFrame.from_pandas(out_df)
    # def resample(
    #     self,
    #     target_interval: str,
    #     *,
    #     closed: str = "right",
    #     label: str = "right",
    # ) -> 'PriceFrame':
    #     """
    #     Downsample le PriceFrame vers `target_interval`.

    #     Hypothèses :
    #     - PriceFrame mono-fréquence (ou en pratique on ne vérifie que la granularité).
    #     - target_interval plus large ou égal à l'interval courant.
    #     - OHLC: open=1ère, high=max, low=min, close=dernière, volume = somme.

    #     Ex: 1m -> 5m, 1h, 1d, 1mo.
    #     """
    #     # On compare les granularités
    #     cur_interval = self.ensure_single_interval()
    #     off_cur, _ = interval_to_offset(cur_interval)
    #     off_tgt, tgt_str = interval_to_offset(target_interval)

    #     if off_tgt.nanos < off_cur.nanos:
    #         raise ValueError(
    #             f"Upsampling non supporté: current={cur_interval} -> target={target_interval}"
    #         )

    #     if off_tgt.nanos == off_cur.nanos:
    #         # même fréquence => rien à faire (on pourrait normaliser interval si besoin)
    #         return self

    #     df = self.to_pandas()
    #     df = df.set_index("ts").sort_index()

    #     pieces = []
    #     for symbol, g in df.groupby("symbol", group_keys=False):
    #         # g.index = ts
    #         # resample avec agrégation OHLCV standard
    #         resampled = (
    #             g.resample(off_tgt, closed=closed, label=label)
    #              .apply(
    #                  lambda x: pd.Series(
    #                      {
    #                          "open": x["open"].iloc[0],
    #                          "high": x["high"].max(),
    #                          "low": x["low"].min(),
    #                          "close": x["close"].iloc[-1],
    #                          "volume": x["volume"].sum(),
    #                      }
    #                  )
    #              )
    #              .dropna(subset=["close"])
    #         )
    #         resampled["symbol"] = symbol
    #         pieces.append(resampled)

    #     if not pieces:
    #         empty = pd.DataFrame(columns=CANON_COLS)
    #         return PriceFrame.from_pandas(empty)

    #     out = pd.concat(pieces)
    #     out = out.reset_index().rename(columns={"index": "ts"})
    #     out["interval"] = tgt_str
    #     return PriceFrame.from_pandas(out)

    # -----------------------------------------------------------------
    # Portefeuille naïf
    # -----------------------------------------------------------------

    def naive_portfolio(
        self,
        symbols: list[str],
        weights: list[float],
        *,
        base: float = 100.0,
        name: Union[str, None] = None,
        align: str = "inner",        # "inner" | "ffill_union"
        as_: str = "series",         # "series" | "priceframe"
    ):
        """
        Portefeuille naïf rebalancé à chaque barre (somme pondérée des returns).
        """
        # Matrice de closes alignée
        if align == "ffill_union":
            M = (
                self.only_close()
                .close_matrix(symbols, how="outer", engine="pandas")
                .sort_index()
                .ffill()
            )
        else:
            M = (
                self.only_close()
                .close_matrix(symbols, how="inner", engine="pandas")
                .sort_index()
            )

        # Vérifs
        missing = [s for s in symbols if s not in M.columns]
        if missing:
            raise KeyError(f"Symbol(s) manquant(s) dans la matrice: {missing}")

        W = np.asarray(weights, dtype=float)
        if W.ndim != 1 or len(W) != len(symbols):
            raise ValueError("`weights` doit être un vecteur de même longueur que `symbols`.")
        if not np.isfinite(W).all():
            raise ValueError("`weights` contient des valeurs non finies.")

        M = M[symbols]
        R = M.pct_change().fillna(0.0)
        r_p = (R * W).sum(axis=1)
        idx = (1.0 + r_p).cumprod() * float(base)

        if name is None:
            parts = [f"{int(round(w * 100))}%{sym}" for sym, w in zip(symbols, W)]
            name = "PF[" + "+".join(parts) + "]"
        idx.name = name

        if as_ == "series":
            return idx

        # as_ == "priceframe"
        df = idx.to_frame(name)

        return PriceFrame.from_close_matrix(
            df,
            interval=None,
            ts_is_close=True,      # index daté au close
            impute_ohlc=True,
            volume_value=0.0,
            source="naive_portfolio",
        )

    # -----------------------------------------------------------------
    # Merge / union de PriceFrame
    # -----------------------------------------------------------------

    def add(
        self,
        other: 'PriceFrame',
        *,
        require_same_interval: bool = True,
    ) -> 'PriceFrame':
        """
        Concatène deux PriceFrame et dé-dup sur (symbol, interval, ts).
        Retourne un **nouveau** PriceFrame (non-mutant).
        """
        if not isinstance(other, PriceFrame):
            raise TypeError("`other` doit être un PriceFrame")

        t1, t2 = self.table, other.table

        for col in ("symbol", "interval", "ts"):
            if col not in t1.column_names:
                raise KeyError(f"Colonne requise absente dans self: '{col}'")
            if col not in t2.column_names:
                raise KeyError(f"Colonne requise absente dans other: '{col}'")

        # Vérif d'interval homogène
        if require_same_interval:
            def one_interval(t: pa.Table) -> set[str]:
                vals = pc.unique(t["interval"]).to_pylist()
                return set(vals)

            u1, u2 = one_interval(t1), one_interval(t2)
            if len(u1) != 1 or len(u2) != 1 or u1 != u2:
                raise ValueError(
                    f"Interval(s) incompatibles: self={sorted(u1)} vs other={sorted(u2)}"
                )

        # Unification des schémas
        uni_schema = pa.unify_schemas([t1.schema, t2.schema])

        def _align(table: pa.Table, schema: pa.Schema) -> pa.Table:
            n = len(table)
            out = []
            present = set(table.column_names)
            for field in schema:
                name, target = field.name, field.type
                if name in present:
                    col = table[name]
                    if col.type != target:
                        if pa.types.is_timestamp(col.type) and pa.types.is_timestamp(target):
                            arr = col
                            if col.type.tz is None and target.tz is not None:
                                arr = pc.assume_timezone(arr, target.tz)
                            elif col.type.tz and target.tz and col.type.tz != target.tz:
                                arr = pc.astimezone(arr, target.tz)
                            col = pc.cast(
                                arr,
                                target,
                                options=pc.CastOptions(target, allow_time_truncate=True),
                            )
                        else:
                            col = pc.cast(col, target, safe=False)
                    out.append(col)
                else:
                    out.append(pa.nulls(n, type=target))
            return pa.Table.from_arrays(out, schema=schema)

        a1 = _align(t1, uni_schema)
        a2 = _align(t2, uni_schema)

        try:
            # PyArrow récent
            comb = pa.concat_tables([a1, a2], promote_options="default")
        except TypeError:
            # PyArrow plus ancien (pas de promote_options)
            comb = pa.concat_tables([a1, a2], promote=True)
        comb = _sort_and_dedup(comb)

        return PriceFrame(comb, schema_version=self.schema_version)

    # -----------------------------------------------------------------
    # from_close_matrix
    # -----------------------------------------------------------------

    @classmethod
    def from_close_matrix(
        cls,
        close_df: pd.DataFrame,
        *,
        interval: Union[str, None] = None,
        ts_is_close: bool = True,
        tz: str = "UTC",
        round_unit: Union[str, None] = "ms",
        drop_all_na_rows: bool = True,
        drop_all_na_cols: bool = True,
        source: Union[str, None] = None,
        quote_ccy: Union[str, None] = None,
    ) -> 'PriceFrame':
        """
        Ingestion d'une close-matrix (index=dates, colonnes=symbols, valeurs=close)
        vers le format canonique long PriceFrame.
        """
        if not isinstance(close_df, pd.DataFrame):
            raise TypeError("from_close_matrix attend un pandas.DataFrame.")

        if close_df.empty:
            empty = pd.DataFrame(columns=["symbol", "interval", "ts", "close"])
            return cls.from_pandas(empty)

        # 1) Index -> DatetimeIndex UTC
        idx = close_df.index
        if not pd.api.types.is_datetime64_any_dtype(idx):
            idx = pd.to_datetime(idx, utc=False)

        if idx.tz is None:
            idx = idx.tz_localize(tz).tz_convert("UTC")
        else:
            idx = idx.tz_convert("UTC")

        if round_unit:
            idx = idx.round(round_unit)

        mat = close_df.copy()
        mat.index = idx

        if drop_all_na_cols:
            mat = mat.dropna(axis=1, how="all")
        if drop_all_na_rows:
            mat = mat.dropna(axis=0, how="all")

        if mat.empty:
            empty = pd.DataFrame(columns=["symbol", "interval", "ts", "close"])
            return cls.from_pandas(empty)

        # 2) Déterminer offset / interval
        if interval is None:
            _, interval_str = infer_interval_from_index(mat.index)
        else:
            _, interval_str = interval_to_offset(interval)

        # --- 4) Passage en long ---
        # stack (ts, symbol) -> close
        try:
            stacked = mat.stack(future_stack=True)  # pandas >= 2.1
        except TypeError:
            # pandas < 2.1 ne connaît pas future_stack
            stacked = mat.stack(dropna=True)

        long = (
            stacked
            .rename_axis(index=["ts", "symbol"])
            .reset_index(name="close")
        )

        long["ts"] = pd.to_datetime(long["ts"], utc=True)
        if round_unit:
            long["ts"] = long["ts"].dt.round(round_unit)

        return cls.from_pandas(long, interval=interval_str, source=source, quote_ccy=quote_ccy)
