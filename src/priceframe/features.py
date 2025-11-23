from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import polars as pl


# -------------------------------------------------------------------
# Types & Spec
# -------------------------------------------------------------------

# Feature function: takes a Polars DataFrame, returns a Polars DataFrame
FeatureFn = Callable[[pl.DataFrame], pl.DataFrame]

# Feature builder: main logic used in the registry
FeatureBuilder = Callable[[pl.DataFrame, "FeatureSpec"], pl.DataFrame]


@dataclass
class FeatureSpec:
    """
    Declarative feature specification.

    Simple example (MA 20 on close):
        FeatureSpec(
            name="f_ma_20",
            func="ma",
            input_col="close",
            params={"window": 20},
        )

    Multi-column example (ATR):
        FeatureSpec(
            name="f_atr_14",
            func="atr",
            input_cols={"high": "high", "low": "low", "close": "close"},
            params={"window": 14, "method": "wilder"},
        )
    """

    # Output column name in the DataFrame
    name: str

    # Logical feature name (key in FEATURE_REGISTRY)
    func: str

    # Simple case: single input column
    input_col: Optional[str] = None

    # Multi-column case: logical -> actual column name mapping
    # E.g., {"high": "high_bid", "low": "low_bid", "close": "mid_close"}
    input_cols: Dict[str, str] = field(default_factory=dict)

    # Free parameters for the feature
    params: Dict[str, Any] = field(default_factory=dict)

    # Convenient helper to retrieve a column name
    def col(self, key: str, default: Optional[str] = None) -> str:
        """
        Retrieve the column name for a logical key.
        
        Priority order:
          - input_cols[key]
          - input_col (if defined)
          - default (if provided)
        """
        if key in self.input_cols:
            return self.input_cols[key]
        if self.input_col is not None:
            return self.input_col
        if default is not None:
            return default
        raise KeyError(f"Column '{key}' not specified in FeatureSpec {self.name!r}.")


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------

FEATURE_REGISTRY: Dict[str, FeatureBuilder] = {}


def register_feature(
    name: str,
    builder: FeatureBuilder,
    *,
    overwrite: bool = False,
) -> None:
    """
    Register a feature in the global registry.
    
    Args:
        name: Logical identifier (e.g., "ma", "ret", "atr", ...).
        builder: Function pl.DataFrame x FeatureSpec -> pl.DataFrame.
        overwrite: Allow overwriting existing feature.
    """
    if not overwrite and name in FEATURE_REGISTRY:
        raise ValueError(f"Feature '{name}' already registered.")
    FEATURE_REGISTRY[name] = builder

def feature(name: str, *, overwrite: bool = False):
    """
    Decorator to register a feature in the global registry.
    
    Usage:
        @feature("ma")
        def _feat_ma(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
            # implementation...
    
    Args:
        name: Logical identifier (e.g., "ma", "ret", "atr", ...).
        overwrite: Allow overwriting existing feature.
    """
    def decorator(builder: FeatureBuilder) -> FeatureBuilder:
        if not overwrite and name in FEATURE_REGISTRY:
            raise ValueError(f"Feature '{name}' already registered.")
        FEATURE_REGISTRY[name] = builder
        return builder
    return decorator


def build_feature_fn(spec: FeatureSpec) -> FeatureFn:
    """
    Transform a FeatureSpec into a pl.DataFrame -> pl.DataFrame function.
    
    Retrieves the builder from FEATURE_REGISTRY.
    
    Args:
        spec: FeatureSpec instance.
        
    Returns:
        FeatureFn: Feature function.
    """
    if spec.func not in FEATURE_REGISTRY:
        raise KeyError(f"Feature func '{spec.func}' not registered.")

    builder = FEATURE_REGISTRY[spec.func]

    def fn(df: pl.DataFrame) -> pl.DataFrame:
        return builder(df, spec)

    return fn


def list_registered_features() -> List[str]:
    """
    Return the list of registered feature names.
    
    Returns:
        List[str]: Sorted list of feature names.
    """
    return sorted(FEATURE_REGISTRY.keys())


# -------------------------------------------------------------------
# Base Feature Implementations
# -------------------------------------------------------------------

# ---------- Returns ----------

@feature("ret")
def _feat_ret(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Simple return: (price_t / price_{t-h} - 1) by symbol.

    Params:
        horizon: int (>=1), return horizon (default: 1).
    """
    h = int(spec.params.get("horizon", 1))
    col = spec.col("price", default="close")

    return df.with_columns(
        (
            pl.col(col) / pl.col(col).shift(h) - 1.0
        ).over("symbol").alias(spec.name)
    )

@feature("logret")
def _feat_logret(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Log-return: log(price_t) - log(price_{t-h}) by symbol.

    Params:
        horizon: int (>=1), return horizon (default: 1).
    """
    h = int(spec.params.get("horizon", 1))
    col = spec.col("price", default="close")

    return df.with_columns(
        (
            pl.col(col).log() - pl.col(col).log().shift(h)
        ).over("symbol").alias(spec.name)
    )


# ---------- Moving Averages ----------

@feature("ma")
def _feat_ma(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Simple Moving Average (SMA) on a column, by symbol.

    Params:
        window: int, window size (required).
    """
    if "window" not in spec.params:
        raise ValueError(f"Feature '{spec.name}' (ma) requires params['window'].")

    w = int(spec.params["window"])
    col = spec.col("price", default="close")

    return df.with_columns(
        pl.col(col)
        .rolling_mean(window_size=w)
        .over("symbol")
        .alias(spec.name)
    )

@feature("ema")
def _feat_ema(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Exponential Moving Average (EMA) on a column, by symbol.

    Params:
        alpha: float, OR
        span: float -> alpha = 2 / (span + 1)
    """
    col = spec.col("price", default="close")
    alpha = spec.params.get("alpha")
    span = spec.params.get("span")

    if alpha is None:
        if span is None:
            raise ValueError(
                f"Feature '{spec.name}' (ema) requires params['alpha'] or params['span']."
            )
        alpha = 2.0 / (float(span) + 1.0)

    return df.with_columns(
        pl.col(col)
        .ewm_mean(alpha=float(alpha), adjust=False)
        .over("symbol")
        .alias(spec.name)
    )


# ---------- Rolling Volatility ----------

@feature("rolling_vol")
def _feat_rolling_vol(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Rolling volatility (standard deviation) of a series (often returns).

    Params:
        window: int, window size (required).
    """
    if "window" not in spec.params:
        raise ValueError(
            f"Feature '{spec.name}' (rolling_vol) requires params['window']."
        )

    w = int(spec.params["window"])
    col = spec.col("value", default="f_ret_1")  # e.g., ret_1

    return df.with_columns(
        pl.col(col)
        .rolling_std(window_size=w)
        .over("symbol")
        .alias(spec.name)
    )


# ---------- Cross-Sectional (by Date) ----------

@feature("cs_rank")
def _feat_cs_rank(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Cross-sectional rank of a column at each timestamp.

    Params:
        method: Polars rank method ("dense", "average", "ordinal", etc.).
    """
    col = spec.col("value", default="f_ret_1")
    method = spec.params.get("method", "dense")

    return df.with_columns(
        pl.col(col)
        .rank(method)
        .over("ts")
        .alias(spec.name)
    )

@feature("cs_zscore")
def _feat_cs_zscore(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Cross-sectional Z-score of a column at each timestamp.
    
    Formula: (x - mean_ts) / std_ts
    """
    col = spec.col("value", default="f_ret_1")

    mean_ts = pl.col(col).mean().over("ts")
    std_ts = pl.col(col).std().over("ts")

    return df.with_columns(
        ((pl.col(col) - mean_ts) / std_ts).alias(spec.name)
    )


# ---------- ATR: Average True Range ----------

@feature("atr")
def _feat_atr(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Average True Range (ATR) by symbol.

    True Range (TR_t):
        max(
            high_t - low_t,
            |high_t - close_{t-1}|,
            |low_t - close_{t-1}|
        )

    Params:
        window: int, ATR length (e.g., 14).
        method: "sma" or "wilder" (default: "sma").
    """
    high_col = spec.col("high", default="high")
    low_col = spec.col("low", default="low")
    close_col = spec.col("close", default="close")

    window = int(spec.params.get("window", 14))
    method = spec.params.get("method", "sma")

    high = pl.col(high_col)
    low = pl.col(low_col)
    close_prev = pl.col(close_col).shift(1)

    tr_expr = pl.max_horizontal(
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    )

    # True Range by symbol
    df = df.with_columns(
        tr_expr.over("symbol").alias("_tr_tmp")
    )

    if method == "sma":
        df = df.with_columns(
            pl.col("_tr_tmp")
            .rolling_mean(window_size=window)
            .over("symbol")
            .alias(spec.name)
        )

    elif method == "wilder":
        # Wilder approximation via EWMA alpha=1/window
        alpha = 1.0 / float(window)
        df = df.with_columns(
            pl.col("_tr_tmp")
            .ewm_mean(alpha=alpha, adjust=False)
            .over("symbol")
            .alias(spec.name)
        )
    else:
        raise ValueError(
            f"Unknown ATR method '{method}', expected 'sma' or 'wilder'."
        )

    df = df.drop("_tr_tmp")
    return df


# ---------- RSI: Relative Strength Index ----------

@feature("rsi")
def _feat_rsi(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    RSI (Relative Strength Index) on 'close' (default), by symbol.

    Params:
        window: int, period (e.g., 14).
        method: "sma" or "wilder" (default: "wilder").
    """
    close_col = spec.col("close", default="close")
    window = int(spec.params.get("window", 14))
    method = spec.params.get("method", "wilder")

    delta = pl.col(close_col) - pl.col(close_col).shift(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

    df = df.with_columns(
        gain.over("symbol").alias("_rsi_gain_tmp"),
        loss.over("symbol").alias("_rsi_loss_tmp"),
    )

    if method == "sma":
        avg_gain = (
            pl.col("_rsi_gain_tmp")
            .rolling_mean(window_size=window)
            .over("symbol")
        )
        avg_loss = (
            pl.col("_rsi_loss_tmp")
            .rolling_mean(window_size=window)
            .over("symbol")
        )
    elif method == "wilder":
        alpha = 1.0 / float(window)
        avg_gain = (
            pl.col("_rsi_gain_tmp")
            .ewm_mean(alpha=alpha, adjust=False)
            .over("symbol")
        )
        avg_loss = (
            pl.col("_rsi_loss_tmp")
            .ewm_mean(alpha=alpha, adjust=False)
            .over("symbol")
        )
    else:
        raise ValueError(
            f"Unknown RSI method '{method}', expected 'sma' or 'wilder'."
        )

    rs = avg_gain / avg_loss
    rsi_expr = 100.0 - 100.0 / (1.0 + rs)

    df = df.with_columns(
        rsi_expr.alias(spec.name)
    )

    df = df.drop("_rsi_gain_tmp", "_rsi_loss_tmp")
    return df


# ---------- Bollinger Bands ----------

@feature("bollinger")
def _feat_bollinger(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Bollinger Bands on a series (default: close).

    Params:
        window: int, window size (e.g., 20).
        n_std: float, standard deviation multiplier (e.g., 2.0).
        band: "mid", "upper", "lower" (output of this feature).
    """
    if "window" not in spec.params:
        raise ValueError(
            f"Feature '{spec.name}' (bollinger) requires params['window']."
        )

    col = spec.col("price", default="close")
    window = int(spec.params["window"])
    n_std = float(spec.params.get("n_std", 2.0))
    band = spec.params.get("band", "mid")

    mid = (
        pl.col(col)
        .rolling_mean(window_size=window)
        .over("symbol")
    )
    std = (
        pl.col(col)
        .rolling_std(window_size=window)
        .over("symbol")
    )

    upper = mid + n_std * std
    lower = mid - n_std * std

    if band == "mid":
        expr = mid
    elif band == "upper":
        expr = upper
    elif band == "lower":
        expr = lower
    else:
        raise ValueError("Bollinger band must be 'mid','upper' or 'lower'.")

    return df.with_columns(
        expr.alias(spec.name)
    )


# ---------- MACD ----------

@feature("macd")
def _feat_macd(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Classic MACD on a series (default: close), by symbol.

    Params:
        fast: int, fast EMA period (e.g., 12).
        slow: int, slow EMA period (e.g., 26).
        signal: int, MACD line EMA period (e.g., 9).
        component: "macd", "signal", "hist".
    """
    col = spec.col("price", default="close")

    fast = int(spec.params.get("fast", 12))
    slow = int(spec.params.get("slow", 26))
    signal = int(spec.params.get("signal", 9))

    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)
    signal_alpha = 2.0 / (signal + 1.0)

    # Fast and slow EMAs
    df = df.with_columns(
        pl.col(col)
        .ewm_mean(alpha=fast_alpha, adjust=False)
        .over("symbol")
        .alias("_macd_fast"),
        pl.col(col)
        .ewm_mean(alpha=slow_alpha, adjust=False)
        .over("symbol")
        .alias("_macd_slow"),
    )

    df = df.with_columns(
        (pl.col("_macd_fast") - pl.col("_macd_slow")).alias("_macd_line")
    )

    df = df.with_columns(
        pl.col("_macd_line")
        .ewm_mean(alpha=signal_alpha, adjust=False)
        .over("symbol")
        .alias("_macd_signal")
    )

    df = df.with_columns(
        (pl.col("_macd_line") - pl.col("_macd_signal")).alias("_macd_hist")
    )

    comp = spec.params.get("component", "macd")
    comp_map = {
        "macd": "_macd_line",
        "signal": "_macd_signal",
        "hist": "_macd_hist",
    }
    if comp not in comp_map:
        raise ValueError("MACD component must be 'macd','signal' or 'hist'.")

    df = df.with_columns(
        pl.col(comp_map[comp]).alias(spec.name)
    )

    # Cleanup temporary columns
    df = df.drop(
        "_macd_fast",
        "_macd_slow",
        "_macd_line",
        "_macd_signal",
        "_macd_hist",
    )
    return df