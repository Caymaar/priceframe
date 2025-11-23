import os
import glob
import datetime as dt
from typing import List, Optional, Sequence, Union

import duckdb


PATH = "/Users/julesmourgues/Documents/crypto-hedge-fund/caymar-crypto-data/binance/data/spot/klines/{symbol}/{interval}"


class KlineLoader:
    _TIME_CANDIDATES = ("open_time", "openTime", "timestamp", "time", "ts")
    _DEFAULT_TIME_UNIT = "ms"  # "ms" ou "s"

    def __init__(
        self,
        base_path: str = PATH,
        db: str = ":memory:",
        threads: Optional[int] = os.cpu_count(),
        memory_limit: Optional[str] = None,
    ):
        self.base_path = base_path
        self.con = duckdb.connect(database=db)
        if threads is not None:
            self.con.execute(f"PRAGMA threads={int(threads)};")
        if memory_limit:
            self.con.execute(f"SET memory_limit='{memory_limit}';")
        self.union_by_name = True

    def load(
        self,
        symbols: Sequence[str],
        interval: str,
        start: Optional[Union[str, dt.datetime]] = None,
        end: Optional[Union[str, dt.datetime]] = None,
        *,
        time_col: Optional[str] = None,
        time_unit: str = _DEFAULT_TIME_UNIT,  # "ms" ou "s" si NUMÉRIQUE
        select_cols: Optional[Sequence[str]] = None,
        drop_duplicates: bool = True,
        order: bool = True,
        as_: str = "pandas",  # "pandas" | "pyarrow" | "polars"
    ):
        files = self._build_file_list(symbols, interval)
        if not files:
            raise FileNotFoundError(
                f"Aucun fichier parquet trouvé pour symbols={symbols} interval={interval}"
            )

        # Détection colonne + type
        time_col = time_col or self._detect_time_column(files)
        if not time_col:
            raise ValueError(
                f"Colonne temps introuvable (candidats={self._TIME_CANDIDATES}). "
                "Spécifie `time_col=` si besoin."
            )
        time_kind, duck_type = self._detect_time_kind(files, time_col)
        # time_kind ∈ {"timestamp","numeric","string"}

        sql = self._build_sql(
            files=files,
            interval=interval,
            time_col=time_col,
            time_kind=time_kind,
            time_unit=time_unit,
            start=start,
            end=end,
            select_cols=select_cols,
            drop_duplicates=drop_duplicates,
            order=order,
        )

        if as_ == "pandas":
            return self.con.execute(sql).fetchdf()
        elif as_ == "pyarrow":
            return self.con.execute(sql).fetch_arrow_table()
        elif as_ == "polars":
            import polars as pl
            return pl.from_arrow(self.con.execute(sql).fetch_arrow_table())
        else:
            raise ValueError("Paramètre `as_` invalide. Utilise 'pandas' | 'pyarrow' | 'polars'.")

    def available_symbols(self, interval: str) -> List[str]:
        base = self.base_path.format(symbol="*", interval=interval)
        root = os.path.dirname(base)
        parent = os.path.dirname(root)
        if not os.path.isdir(parent):
            return []
        syms = []
        for sym in os.listdir(parent):
            p = self.base_path.format(symbol=sym, interval=interval)
            if os.path.isdir(p) and glob.glob(os.path.join(p, "*.parquet")):
                syms.append(sym)
        return sorted(syms)

    # -------------------------- Internals ---------------------------

    def _build_file_list(self, symbols: Sequence[str], interval: str) -> List[str]:
        globs = [
            os.path.join(self.base_path.format(symbol=s, interval=interval), "*.parquet")
            for s in symbols
        ]
        any_match = any(glob.glob(g) for g in globs)
        return globs if any_match else []

    def _detect_time_column(self, files: Sequence[str]) -> Optional[str]:
        files_sql = self._list_literal_sql(files)
        desc = self.con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({files_sql}, union_by_name={str(self.union_by_name).lower()}) LIMIT 0"
        ).fetchdf()
        cols = [c.lower() for c in desc["column_name"].tolist()]
        for cand in self._TIME_CANDIDATES:
            if cand.lower() in cols:
                return desc.loc[cols.index(cand.lower()), "column_name"]
        return None

    def _detect_time_kind(self, files: Sequence[str], time_col: str):
        """
        Retourne (kind, duck_type)
        kind ∈ {"timestamp","numeric","string"}
        """
        files_sql = self._list_literal_sql(files)
        desc = self.con.execute(
            f"DESCRIBE SELECT {self._quote_ident(time_col)} FROM read_parquet({files_sql}, union_by_name={str(self.union_by_name).lower()}) LIMIT 0"
        ).fetchdf()
        tname = str(desc.loc[0, "column_type"]).upper()
        if "TIMESTAMP" in tname:   # ex: TIMESTAMP_NS, TIMESTAMP
            return "timestamp", tname
        if any(num in tname for num in ("BIGINT", "HUGEINT", "INTEGER", "DECIMAL", "DOUBLE", "REAL", "UBIGINT", "SMALLINT", "TINYINT")):
            return "numeric", tname
        return "string", tname

    def _build_sql(
        self,
        files: Sequence[str],
        interval: str,
        time_col: str,
        time_kind: str,
        time_unit: str,
        start: Optional[Union[str, dt.datetime]],
        end: Optional[Union[str, dt.datetime]],
        select_cols: Optional[Sequence[str]],
        drop_duplicates: bool,
        order: bool,
    ) -> str:
        files_sql = self._list_literal_sql(files)

        # 1) Expression ts normalisée (TIMESTAMP) selon le type détecté
        if time_kind == "timestamp":
            # la colonne est déjà un TIMESTAMP (ex: TIMESTAMP_NS)
            to_ts_expr = f"CAST({self._quote_ident(time_col)} AS TIMESTAMP)"
            # Préparer WHERE sur la colonne source (meilleur pushdown)
            where_parts = []
            if start:
                start_iso = self._as_utc_iso(start)
                where_parts.append(f"{self._quote_ident(time_col)} >= TIMESTAMP '{start_iso}'")
            if end:
                end_iso = self._as_utc_iso(end)
                where_parts.append(f"{self._quote_ident(time_col)} <= TIMESTAMP '{end_iso}'")
            where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        elif time_kind == "numeric":
            # colonne = epoch s|ms (guidé par `time_unit`)
            if time_unit not in ("ms", "s"):
                raise ValueError("time_unit doit être 'ms' ou 's' pour une colonne temporelle numérique.")
            to_ts_expr = (
                f"to_timestamp({self._quote_ident(time_col)} / 1000.0)" if time_unit == "ms"
                else f"to_timestamp({self._quote_ident(time_col)})"
            )
            # WHERE en unités natives de la colonne pour pushdown
            where_parts = []
            if start:
                sec = self._as_epoch_seconds(start)
                left = int(sec * 1000) if time_unit == "ms" else int(sec)
                where_parts.append(f"{self._quote_ident(time_col)} >= {left}")
            if end:
                sec = self._as_epoch_seconds(end)
                right = int(sec * 1000) if time_unit == "ms" else int(sec)
                where_parts.append(f"{self._quote_ident(time_col)} <= {right}")
            where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        else:  # "string"
            # on suppose un ISO string → CAST
            to_ts_expr = f"CAST({self._quote_ident(time_col)} AS TIMESTAMP)"
            where_parts = []
            if start:
                start_iso = self._as_utc_iso(start)
                where_parts.append(f"CAST({self._quote_ident(time_col)} AS TIMESTAMP) >= TIMESTAMP '{start_iso}'")
            if end:
                end_iso = self._as_utc_iso(end)
                where_parts.append(f"CAST({self._quote_ident(time_col)} AS TIMESTAMP) <= TIMESTAMP '{end_iso}'")
            where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # 2) Colonnes
        symbol_expr = r"regexp_extract(filename, '/klines/([^/]+)/', 1)"
        base_select = [
            f"{to_ts_expr} AS ts",
            f"{symbol_expr} AS symbol",
            f"'{interval}'::VARCHAR AS interval",
        ]
        if select_cols is None:
            data_cols = ["t.*"]
        else:
            data_cols = [self._quote_ident(c) for c in select_cols]

        dedup = (
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol, ts ORDER BY ts) = 1"
            if drop_duplicates
            else ""
        )
        order_sql = "ORDER BY symbol, ts" if order else ""

        sql = f"""
WITH src AS (
  SELECT
    {", ".join(base_select + data_cols)}
  FROM read_parquet({files_sql}, union_by_name={str(self.union_by_name).lower()}, filename=true) AS t
),
filt AS (
  SELECT
    {", ".join(["ts", "symbol", "interval"] + (data_cols if select_cols else ["*"]))}
  FROM src
  {where_sql}
  {dedup}
)
SELECT
  {"* EXCLUDE(filename)" if select_cols is None else "*"}
FROM filt
{order_sql}
;
"""
        return sql

    @staticmethod
    def _list_literal_sql(files: Sequence[str]) -> str:
        esc = lambda p: p.replace("'", "''")
        return "[" + ", ".join(f"'{esc(p)}'" for p in files) + "]"

    @staticmethod
    def _as_utc_iso(x: Union[str, dt.datetime]) -> str:
        if isinstance(x, str):
            return x
        if x.tzinfo is None:
            x = x.replace(tzinfo=dt.timezone.utc)
        return x.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _as_epoch_seconds(x: Union[str, dt.datetime]) -> float:
        if isinstance(x, str):
            # on interprète comme UTC naïf
            dtobj = dt.datetime.fromisoformat(x.replace("Z", "")).replace(tzinfo=dt.timezone.utc)
        else:
            dtobj = x if x.tzinfo else x.replace(tzinfo=dt.timezone.utc)
        return dtobj.timestamp()

    @staticmethod
    def _quote_ident(name: str) -> str:
        safe = name.replace('"', '""')
        return f'"{safe}"'