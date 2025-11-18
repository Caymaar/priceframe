from dataclasses import dataclass
from typing import Optional, Sequence, Literal
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd

from datetime import datetime
import polars as pl
from xbbg import blp

from .features import FeatureFn, FeatureSpec, build_feature_fn

CANON_COLS = ["symbol","interval","ts","open","high","low","close","volume"]
CANON_TYPES = {
    "symbol": pa.string(),
    "interval": pa.string(),
    "ts": pa.timestamp("ms", tz="UTC"),
    "open": pa.float64(),
    "high": pa.float64(),
    "low": pa.float64(),
    "close": pa.float64(),
    "volume": pa.float64(),
}


SCHEMA_V1 = pa.schema({**CANON_TYPES, **{}})  # ajoute champs optionnels si tu veux

@dataclass
class PriceFrame:
    table: pa.Table               # stockage canon interne
    schema_version: str = "ohlcv_v1"

    # ---------- Constructors ----------
    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "PriceFrame":
        # normaliser noms : open_time -> ts, number_of_trades -> trades, etc.
        rename = {"open_time":"ts","Open time":"ts","number_of_trades":"trades"}
        df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
        # lowercase columns
        df.columns = [c.lower() for c in df.columns]
        # garantir colonnes minimales
        missing = [c for c in CANON_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        # ts tz-UTC en ms
        ts = pd.to_datetime(df["ts"], utc=True)
        df = df.copy()
        df["ts"] = ts
        # types numériques
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # tri & drop dup
        df = df.sort_values(["symbol","ts"]).drop_duplicates(subset=["symbol","interval","ts"])
        # vers Arrow (pandas tz-aware -> timestamp[ns, tz=UTC] → Arrow)
        t = pa.Table.from_pandas(df, preserve_index=False)
        # cast canon (→ timestamp ms utc)
        t = _cast_to_canon(t)
        return cls(t)

    @classmethod
    def from_polars(cls, pl_df: pl.DataFrame) -> "PriceFrame":
        return cls.from_arrow(pl_df.to_arrow())

    @classmethod
    def from_arrow(cls, table: pa.Table) -> "PriceFrame":
        # harmoniser noms et caster
        cols = {c: c for c in table.column_names}
        rename_pairs = []
        for k,v in {"open_time":"ts","Open time":"ts","number_of_trades":"trades"}.items():
            if k in cols:
                rename_pairs.append((k,v))
        if rename_pairs:
            for src,dst in rename_pairs:
                table = table.rename_columns([dst if name==src else name for name in table.column_names])
        # check colonnes minimales
        missing = [c for c in CANON_COLS if c not in table.column_names]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        table = _cast_to_canon(table)
        table = _sort_and_dedup(table)
        return cls(table)
    
    @classmethod
    def close_from_bloomberg(cls, tickers: Sequence[str], start: datetime = None, end: datetime = None, freq: str = "1d") -> "PriceFrame":

        if end is None:
            end = "today"

        data = blp.bdh(
            tickers,
            flds=["PX_LAST"],
            start_date=start,
            end_date=end
        )
        # data: MultiIndex DataFrame (date, ticker) x columns (flds)
        records = []

        print(data)
        
        for (dt, ticker), row in data.iterrows():
            record = {
                "symbol": ticker,
                "interval": freq,
                "ts": pd.Timestamp(dt, tz="UTC"),
                "close": row["PX_LAST"],
            }
            records.append(record)
        df = pd.DataFrame.from_records(records)
        return cls.from_pandas(df)

    @classmethod
    def ohlcv_from_bloomberg(cls, tickers: Sequence[str], start: datetime = None, end: datetime = None, freq: str = "1d", **kwargs) -> "PriceFrame":


        data = blp.bdh(
            tickers,
            flds=["OPEN","HIGH","LOW","PX_LAST","VOLUME"],
            start_date=start,
            end_date=end,
            Per= freq,
            Fill="P",
            **kwargs
        )
        # data: MultiIndex DataFrame (date, ticker) x columns (flds)
        records = []
        for (dt, ticker), row in data.iterrows():
            record = {
                "symbol": ticker,
                "interval": freq,
                "ts": pd.Timestamp(dt, tz="UTC"),
                "open": row["OPEN"],
                "high": row["HIGH"],
                "low": row["LOW"],
                "close": row["PX_LAST"],
                "volume": row["VOLUME"],
            }
            records.append(record)
        df = pd.DataFrame.from_records(records)
        return cls.from_pandas(df)


    # ---------- Conversions ----------
    def to_arrow(self, columns: Optional[Sequence[str]]=None) -> pa.Table:
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

    # ---------- Utilities ----------
    def ensure_freq(self, expected_interval: str) -> "PriceFrame":
        uniques = set(self.table.column("interval").to_pylist())
        if len(uniques) != 1 or expected_interval not in uniques:
            raise ValueError(f"Interval non homogène: {uniques}")
        return self

    def with_indicators(self, *extras: str) -> "PriceFrame":
        # exemple: rajouter ret_log_1, ma_20
        df = self.to_polars()
        df = df.sort(["symbol","ts"])
        df = df.with_columns([
            (pl.col("close").log() - pl.col("close").log().shift(1)).over("symbol").alias("ret_log_1"),
            pl.col("close").rolling_mean(window_size=20).over("symbol").alias("ma_20"),
        ])
        return PriceFrame.from_polars(df)
    

    # ---------- Features API ----------

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

    # ---------- Data Accessors ----------
    def for_symbol(self, symbol: str) -> "PriceFrame":
            t = self.table.filter(pc.equal(self.table["symbol"], pa.scalar(symbol)))
            return PriceFrame(t, schema_version=self.schema_version)

    # Réduction aux seules colonnes utiles (ex: close-only)
    def only_close(self) -> "PriceFrame":
        cols = [c for c in self.table.column_names if c in {"symbol","interval","ts","close"}]
        return PriceFrame(self.table.select(cols), schema_version=self.schema_version)
    
    def ohlcv(self, symbol: str, engine: str = "pandas") -> pd.DataFrame | pl.DataFrame:
        pf = self.for_symbol(symbol)
        if engine == "pandas":
            df = pf.to_pandas(index="ts")
            
            return df[["open","high","low","close","volume"]]
        elif engine == "polars":
            pl_df = pf.to_polars().select(["ts","open","high","low","close","volume"]).sort("ts")
            return pl_df
        else:
            raise ValueError("engine ∈ {'pandas','polars'}")

    # Série de close pour un symbole (Pandas/Polars)
    def close_series(self, symbol: str, engine: str = "pandas"):
        pf = self.for_symbol(symbol).only_close()
        if engine == "pandas":
            s = pf.to_pandas(index="ts")["close"]
            s.name = symbol
            return s
        elif engine == "polars":
            import polars as pl
            df = pf.to_polars().select(["ts","close"]).sort("ts")
            return df.rename({"close": symbol})
        else:
            raise ValueError("engine ∈ {'pandas','polars'}")

    # Matrice wide (dates x symbols) pour backtests (close)
    def close_matrix(
        self, 
        symbols: list[str] = None,
        how: str = "inner",        # 'inner' (intersection) ou 'outer' (union)
        fill: float | None = None, # ex: forward-fill ensuite côté pandas
        engine: str = "pandas"
    ):
        if engine == "pandas":
            df = self.to_pandas()   # long
            piv = df.pivot_table(index="ts", columns="symbol", values="close", aggfunc="first")
            if symbols is not None:
                piv = piv[symbols]
            if how == "outer":
                piv = piv  # déjà outer par défaut sur ts
            else:  # inner = intersection stricte
                # garde les dates présentes pour tous les symbols
                piv = piv.dropna(how="any")
            if fill is not None:
                piv = piv.fillna(fill)
            piv.index = pd.to_datetime(piv.index)
            return piv.sort_index()
        elif engine == "polars":
            import polars as pl
            pl_df = self.to_polars().select(["ts","symbol","close"])
            # pivot polars
            wide = pl_df.pivot(values="close", index="ts", columns="symbol", aggregate_function="first").sort("ts")
            if symbols is not None:
                wide = wide.select(["ts"] + symbols)
            if how == "inner":
                wide = wide.drop_nulls()
            if fill is not None:
                wide = wide.fill_null(fill)
            return wide
        else:
            raise ValueError("engine ∈ {'pandas','polars'}")

    def range_date(
        self,
        start=None,               # str | datetime | None
        end=None,                 # str | datetime | None
        *,
        closed: str = "both",     # "both" | "left" | "right" | "neither"
    ) -> "PriceFrame":
        """Filtre le PriceFrame sur [start, end] selon closed ; retourne un nouveau PriceFrame."""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.compute as pc

        t = self.table

        def _to_pa_ts_ms_utc(x):
            if x is None:
                return None
            ts = pd.Timestamp(x)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            return pa.scalar(ts.to_pydatetime(), type=pa.timestamp("ms", tz="UTC"))

        s = _to_pa_ts_ms_utc(start)
        e = _to_pa_ts_ms_utc(end)

        conds = []
        if s is not None:
            op = pc.greater_equal if closed in ("both", "left") else pc.greater
            conds.append(op(t["ts"], s))
        if e is not None:
            op = pc.less_equal if closed in ("both", "right") else pc.less
            conds.append(op(t["ts"], e))

        if conds:
            mask = conds[0]
            for c in conds[1:]:
                mask = pc.and_(mask, c)
            t2 = t.filter(mask)
        else:
            t2 = t
        return PriceFrame(t2, schema_version=self.schema_version)

    def rebase(
        self,
        base: float = 100.0,
        *,
        anchor: str = "close",                     # colonne d'ancrage du rebase
        cols: tuple[str, ...] = ("open","high","low","close"),  # colonnes à rebaser
    ) -> "PriceFrame":
        """
        Rebase OHLC par symbole : remplace les valeurs existantes
        (ex: première barre close = `base`, et O/H/L sont mis à l'échelle avec le même facteur).
        - Supprime les lignes avant le premier `anchor` non-NaN pour chaque symbole.
        """

        import numpy as np
        import pandas as pd

        df = self.to_pandas().sort_values(["symbol", "ts"]).reset_index(drop=True)

        if anchor not in df.columns:
            raise KeyError(f"Colonne d'ancrage '{anchor}' absente.")
        for c in cols:
            if c not in df.columns:
                raise KeyError(f"Colonne '{c}' absente.")

        # 1) Supprimer les lignes avant le premier anchor non-NaN pour chaque symbole
        #    (garantit que la première barre observée vaut exactement `base`)
        has_obs_cum = (
            df.groupby("symbol")[anchor]
            .apply(lambda s: s.notna().cumsum() > 0)
            .reset_index(level=0, drop=True)
        )
        df = df[has_obs_cum].copy()

        # 2) Calcul du facteur d'échelle par ligne (base / premier close observé du symbole)
        first_anchor = df.groupby("symbol")[anchor].transform("first")
        scale = (float(base) / first_anchor).astype(float)

        # 3) Mise à l'échelle des colonnes OHLC (NaN restent NaN)
        for c in cols:
            df[c] = df[c] * scale

        # 4) Retour en PriceFrame (mêmes colonnes, valeurs écrasées)
        return PriceFrame.from_pandas(df)

    def naive_portfolio(
        self,
        symbols: list[str],
        weights: list[float],
        *,
        base: float = 100.0,
        name: str | None = None,
        align: str = "inner",        # "inner" | "ffill_union"
        as_: str = "series",         # "series" | "priceframe"
    ) :
        """
        Portefeuille naïf rebalancé chaque barre (somme pondérée des returns).
        - symbols: tickers à inclure
        - weights: mêmes longueurs que symbols (normalisés si besoin)
        - base: valeur initiale de l'indice (100 par défaut)
        - align:
            * "inner": intersection stricte des dates communes
            * "ffill_union": union des dates + ffill avant calcul (pratique multi-sources)
        - as_:
            * "series": pd.Series (par défaut)
            * "priceframe": PriceFrame mono-symbole via from_close_matrix
        """
        import numpy as np
        import pandas as pd

        # Matrice de closes alignée
        if align == "ffill_union":
            M = self.only_close().close_matrix(symbols, how="outer", engine="pandas").sort_index().ffill()
        else:
            M = self.only_close().close_matrix(symbols, how="inner", engine="pandas").sort_index()

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
        R = M.pct_change().fillna(0.0)  # rendements périodiques (0 au 1er indice)
        r_p = (R * W).sum(axis=1)
        idx = (1.0 + r_p).cumprod() * float(base)

        if name is None:
            parts = [f"{int(round(w*100))}%{sym}" for sym, w in zip(symbols, W)]
            name = "PF[" + "+".join(parts) + "]"
        idx.name = name

        
        if as_ == "series":
            return idx

        # as_ == "priceframe" : on encapsule cette série comme un PriceFrame mono-symbole
        df = idx.to_frame(name)  # colonnes = [name]

        return PriceFrame.from_close_matrix(
            df,
            interval=None,           # infère la fréquence
            ts_is_close=False,        # l'indice est daté au close de la barre
            impute_ohlc=True,        # crée OHLC = close, volume=0
            volume_value=0.0,
            source="naive_portfolio",
        )

    def add(
        self,
        other: "PriceFrame",
        *,
        require_same_interval: bool = True,
    ) -> "PriceFrame":
        import pyarrow as pa
        import pyarrow.compute as pc

        if not isinstance(other, PriceFrame):
            raise TypeError("`other` doit être un PriceFrame")

        t1, t2 = self.table, other.table
        for col in ("symbol", "interval", "ts"):
            if col not in t1.column_names:
                raise KeyError(f"Colonne requise absente dans self: '{col}'")
            if col not in t2.column_names:
                raise KeyError(f"Colonne requise absente dans other: '{col}'")

        # --- Vérif d’interval (mono-interval identique) ---
        if require_same_interval:
            def one_interval(t):
                vals = pc.unique(t["interval"]).to_pylist()
                return set(vals)
            u1, u2 = one_interval(t1), one_interval(t2)
            if len(u1) != 1 or len(u2) != 1 or u1 != u2:
                raise ValueError(f"Interval(s) incompatibles: self={sorted(u1)} vs other={sorted(u2)}")

        # --- Unifier les schémas ---
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
                            # harmoniser tz
                            if col.type.tz is None and target.tz is not None:
                                arr = pc.assume_timezone(arr, target.tz)
                            elif col.type.tz and target.tz and col.type.tz != target.tz:
                                arr = pc.astimezone(arr, target.tz)
                            col = pc.cast(arr, target, options=pc.CastOptions(target, allow_time_truncate=True))
                        else:
                            col = pc.cast(col, target, safe=False)
                    out.append(col)
                else:
                    out.append(pa.nulls(n, type=target))
            return pa.Table.from_arrays(out, schema=schema)

        a1 = _align(t1, uni_schema)
        a2 = _align(t2, uni_schema)

        # --- Concat ---
        comb = pa.concat_tables([a1, a2], promote=True)

        # --- Dé-dup sur (symbol, interval, ts), garder la 1ère occurrence (self d’abord) ---
        try:
            comb = pc.drop_duplicates(comb, keys=["symbol", "interval", "ts"], keep="first")
        except Exception:
            # Fallback universel via pandas si version PyArrow trop ancienne
            df = comb.to_pandas(types_mapper=pd.ArrowDtype)
            df = df.sort_values(["symbol", "ts"])
            df = df.drop_duplicates(subset=["symbol", "interval", "ts"], keep="first")
            self.table = PriceFrame.from_pandas(df).table
            return
        # --- Tri final ---
        comb = comb.sort_by([("symbol", "ascending"), ("ts", "ascending")])
        self.table = comb
        return
        #return PriceFrame(comb, schema_version=self.schema_version)

    @classmethod
    def from_close_matrix(
        cls,
        close_df: "pd.DataFrame",
        *,
        interval: str | None = None,       # ex: "1m","5m","1h","1d","1w","1mo","1q","1y"; si None -> tentative d'inférence
        ts_is_close: bool = True,           # True = index est daté au close => on translate en open
        tz: str = "UTC",                    # si index naïf, on assume ce tz avant conversion UTC
        round_unit: str | None = "ms",      # arrondit l'index avant export Arrow (évite us->ms errors)
        drop_all_na_rows: bool = True,      # drop lignes où tous les symbols sont NaN
        drop_all_na_cols: bool = True,      # drop colonnes (symbols) 100% NaN
        impute_ohlc: bool = True,           # crée open/high/low = close (utile si ton PriceFrame exige OHLCV)
        volume_value: float | None = 0.0,   # valeur de volume si imputation (None => pas de colonne volume)
        source: str | None = None,          # métadonnées optionnelles
        quote_ccy: str | None = None,
    ) -> "PriceFrame":
        """
        Ingestion d'une close-matrix (index=dates, colonnes=symbols, valeurs=close)
        vers le format canonique long du PriceFrame.
        """

        import pandas as pd
        from pandas.tseries.frequencies import to_offset
        from pandas.tseries import offsets as po

        if not isinstance(close_df, pd.DataFrame):
            raise TypeError("from_close_matrix attend un pandas.DataFrame (close matrix).")

        if close_df.empty:
            # table Arrow vide mais avec schema canon minimal
            empty = pd.DataFrame(columns=["symbol","interval","ts","close"])
            return cls.from_pandas(empty)

        # --- 1) Nettoyage de base / timezone ---
        idx = close_df.index
        if not pd.api.types.is_datetime64_any_dtype(idx):
            idx = pd.to_datetime(idx, utc=False)

        if idx.tz is None:
            # localise puis convertit UTC
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
            empty = pd.DataFrame(columns=["symbol","interval","ts","close"])
            return cls.from_pandas(empty)

        # --- 2) Déterminer la fréquence / interval ---
        def _infer_freq_str(index: "pd.DatetimeIndex") -> str | None:
            # essaie infer_freq, sinon médiane des diffs (pour S/min/h/j/sem)
            f = pd.infer_freq(index)
            if f is not None:
                return f
            diffs = pd.Series(index[1:] - index[:-1]).dropna()
            if len(diffs) == 0:
                return None
            med = diffs.median()
            # map quelques durées usuelles
            if med.components.days >= 365:
                return "A"                     # annuel
            if med.components.days >= 90:
                return "Q"                     # trimestriel approx
            if med.components.days >= 28:
                return "M"                     # mensuel approx
            # secondes / minutes / heures / jours
            s = med.total_seconds()
            if abs(s - 1) < 1e-6:   return "S"
            if abs(s - 5) < 1e-6:   return "5S"
            if abs(s - 10) < 1e-6:  return "10S"
            if abs(s - 15) < 1e-6:  return "15S"
            if abs(s - 30) < 1e-6:  return "30S"
            if abs(s - 60) < 1e-6:  return "T"   # minute
            if abs(s - 300) < 1e-6: return "5T"
            if abs(s - 600) < 1e-6: return "10T"
            if abs(s - 900) < 1e-6: return "15T"
            if abs(s - 1800) < 1e-6:return "30T"
            if abs(s - 3600) < 1e-6:return "H"
            if abs(s - 86400) < 1e-6:return "D"
            if abs(s - 604800) < 1e-6:return "W"
            return None

        def _interval_str_from_offset(off) -> str:
            # map l'offset pandas vers une notation concise type "1m","1h","1d","1w","1mo","1q","1y"
            n = getattr(off, "n", 1)
            # granularités fixes
            if isinstance(off, po.Second):   return f"{n}s"
            if isinstance(off, po.Minute):   return f"{n}m"
            if isinstance(off, po.Hour):     return f"{n}h"
            if isinstance(off, po.Day):      return f"{n}d"
            if isinstance(off, po.Week):     return f"{n}w"
            # périodes variables
            if isinstance(off, (po.MonthEnd, po.MonthBegin, po.BMonthEnd, po.BMonthBegin, po.SemiMonthBegin, po.SemiMonthEnd)):
                return f"{n}mo"
            if isinstance(off, (po.QuarterEnd, po.QuarterBegin, po.BQuarterEnd, po.BQuarterBegin)):
                return f"{n}q"
            if isinstance(off, (po.YearEnd, po.YearBegin, po.BYearEnd, po.BYearBegin)):
                return f"{n}y"
            # fallback générique
            name = getattr(off, "name", None) or str(off)
            return name

        def _open_shift_from_offset(off):
            # si l'index est daté au close, de combien on décale pour obtenir l'open
            if isinstance(off, (po.MonthEnd, po.BMonthEnd, po.SemiMonthEnd)):
                return po.MonthBegin(1)
            if isinstance(off, (po.QuarterEnd, po.BQuarterEnd)):
                return po.QuarterBegin(1, startingMonth=getattr(off, "startingMonth", 1))
            if isinstance(off, (po.YearEnd, po.BYearEnd)):
                return po.YearBegin(1)
            # pour offsets fixes (S/min/h/j/sem), utiliser l'offset tel quel
            return off

        # 2.a choisir un offset
        pandas_freq = None
        if interval is None:
            pandas_freq = _infer_freq_str(mat.index)
            if pandas_freq is None:
                raise ValueError(
                    "Impossible d'inférer la fréquence de la close-matrix. "
                    "Passe `interval=` (ex: '1m','1h','1d','1w','1mo','1q','1y')."
                )
            off = to_offset(pandas_freq)
            interval_str = _interval_str_from_offset(off)
        else:
            # on transforme le string utilisateur en offset pandas si possible
            # ex '1m','5m','1h','1d','1w','1mo','1q','1y'
            _map = {
                "s":"S","m":"T","h":"H","d":"D","w":"W","mo":"M","q":"Q","y":"A"
            }
            # découpe '10m' -> ('10','m')
            import re
            m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", interval.strip())
            if not m:
                # laisser pandas deviner (T/H/D/W/M/Q/A)
                off = to_offset(interval)
                interval_str = _interval_str_from_offset(off)
            else:
                n, u = int(m.group(1)), m.group(2).lower()
                if u not in _map:
                    off = to_offset(interval)  # laisser pandas gérer custom
                    interval_str = _interval_str_from_offset(off)
                else:
                    pandas_code = _map[u]
                    off = to_offset(f"{n}{pandas_code}")
                    interval_str = _interval_str_from_offset(off)

        # --- 3) Re-libeller en open si l'index représente le close ---
        if ts_is_close:
            shift = _open_shift_from_offset(off)
            mat.index = mat.index - shift

        # --- 4) Passage en long ---
        # stack (ts, symbol) -> close
        try:
            stacked = mat.stack(dropna=True, future_stack=True)  # pandas >= 2.1
        except TypeError:
            # pandas < 2.1 ne connaît pas future_stack
            stacked = mat.stack(dropna=True)

        long = (
            stacked
            .rename_axis(index=["ts", "symbol"])
            .reset_index(name="close")
        )
        long["interval"] = interval_str

        # métadonnées optionnelles
        if source is not None:
            long["source"] = source
        if quote_ccy is not None:
            long["quote_ccy"] = quote_ccy

        # imputation OHLC/volume si demandé (utile si ton PriceFrame attend OHLCV)
        if impute_ohlc:
            long["open"] = long["close"]
            long["high"] = long["close"]
            long["low"]  = long["close"]
            if volume_value is not None:
                long["volume"] = volume_value

        # garantir dtype/ordre colonnes minimales
        # (la normalisation/cast + tri/dedup seront faits dans from_pandas)
        # ts doit être tz-aware UTC
        long["ts"] = pd.to_datetime(long["ts"], utc=True)
        if round_unit:
            long["ts"] = long["ts"].dt.round(round_unit)

        return cls.from_pandas(long)

# --------- Helpers internes ---------
def _cast_to_canon(t: pa.Table) -> pa.Table:
    # ts -> ms utc
    if pa.types.is_timestamp(t.schema.field("ts").type):
        # convert tz to UTC + ms
        ts = t.column("ts")
        # cast to ms retains tz if set; ensure tz=UTC
        if ts.type.tz is None:
            # suppose UTC naïf
            t = t.set_column(t.schema.get_field_index("ts"),
                             "ts", pc.cast(ts, pa.timestamp("ms", tz="UTC")))
        elif ts.type.tz != "UTC" or ts.type.unit != "ms":
            t = t.set_column(t.schema.get_field_index("ts"),
                             "ts", pc.cast(pc.assume_timezone(ts, "UTC") if ts.type.tz is None else pc.cast(ts, pa.timestamp("ms", tz="UTC")),
                                           pa.timestamp("ms", tz="UTC")))
    else:
        # numérique epoch → timestamp ms utc
        ts = t.column("ts")
        # heuristique: si entier très grand -> ms; sinon s
        arr = ts.slice(0, min(10, len(ts)))
        mx = max(arr.to_pylist()) if len(arr)>0 else 0
        unit = "ms" if mx > 10**11 else "s"
        val = ts if unit=="s" else pc.divide(ts, pc.scalar(1000))
        t = t.set_column(t.schema.get_field_index("ts"),
                         "ts", pc.cast(val, pa.timestamp("ms", tz="UTC")))
    # caster types canoniques
    desired = {**CANON_TYPES}
    for name, typ in desired.items():
        if name in t.column_names and t.schema.field(name).type != typ:
            try:
                t = t.set_column(t.schema.get_field_index(name), name, pc.cast(t.column(name), typ))
            except Exception:
                pass  # si cast impossible, on laisse (ex: volume int64 acceptable)
    return t

def _sort_and_dedup(t: pa.Table) -> pa.Table:
    t = t.sort_by([("symbol","ascending"),("ts","ascending")])
    # drop duplicates on (symbol, interval, ts)
    key = pc.binary_join_element_wise(
        pc.binary_join_element_wise(t.column("symbol"), pc.scalar("|"), t.column("interval")),
        pc.scalar("|"), pc.cast(t.column("ts"), pa.timestamp("ms"))
    )
    # keep first occurrence
    _, indices = pc.unique(key, keep="first", indices_only=True)
    return t.take(indices)

