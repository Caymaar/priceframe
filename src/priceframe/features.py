from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import polars as pl


# -------------------------------------------------------------------
# Types & Spec
# -------------------------------------------------------------------

# Fonction de feature : prend un DataFrame Polars, renvoie un DataFrame Polars
FeatureFn = Callable[[pl.DataFrame], pl.DataFrame]

# Builder d'une feature : logique principale utilisée dans le registry
FeatureBuilder = Callable[[pl.DataFrame, "FeatureSpec"], pl.DataFrame]


@dataclass
class FeatureSpec:
    """
    Spécification déclarative d'une feature.

    Exemple simple (MA 20 sur le close) :
        FeatureSpec(
            name="f_ma_20",
            func="ma",
            input_col="close",
            params={"window": 20},
        )

    Exemple multi-colonnes (ATR) :
        FeatureSpec(
            name="f_atr_14",
            func="atr",
            input_cols={"high": "high", "low": "low", "close": "close"},
            params={"window": 14, "method": "wilder"},
        )
    """

    # Nom de la colonne de sortie dans le DataFrame
    name: str

    # Nom logique de la feature (clé dans FEATURE_REGISTRY)
    func: str

    # Cas simple : une colonne d'entrée unique
    input_col: Optional[str] = None

    # Cas multi-colonnes : mapping logique -> nom de colonne réelle
    # Ex: {"high": "high_bid", "low": "low_bid", "close": "mid_close"}
    input_cols: Dict[str, str] = field(default_factory=dict)

    # Paramètres libres pour la feature
    params: Dict[str, Any] = field(default_factory=dict)

    # Helper pratique pour récupérer un nom de colonne
    def col(self, key: str, default: Optional[str] = None) -> str:
        """
        Récupère le nom de colonne pour une clé logique.
        Ordre de priorité :
          - input_cols[key]
          - input_col (si défini)
          - default (si fourni)
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
    Enregistre une feature dans le registry global.
    - name: identifiant logique (ex. "ma", "ret", "atr", ...)
    - builder: fonction pl.DataFrame x FeatureSpec -> pl.DataFrame
    """
    if not overwrite and name in FEATURE_REGISTRY:
        raise ValueError(f"Feature '{name}' déjà enregistrée.")
    FEATURE_REGISTRY[name] = builder

def feature(name: str, *, overwrite: bool = False):
    """
    Décorateur pour enregistrer une feature dans le registry global.
    
    Usage:
        @feature("ma")
        def _feat_ma(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
            # implementation...
    
    - name: identifiant logique (ex. "ma", "ret", "atr", ...)
    - overwrite: permet d'écraser une feature existante
    """
    def decorator(builder: FeatureBuilder) -> FeatureBuilder:
        if not overwrite and name in FEATURE_REGISTRY:
            raise ValueError(f"Feature '{name}' déjà enregistrée.")
        FEATURE_REGISTRY[name] = builder
        return builder
    return decorator


def build_feature_fn(spec: FeatureSpec) -> FeatureFn:
    """
    Transforme un FeatureSpec en fonction pl.DataFrame -> pl.DataFrame
    en allant chercher le builder dans FEATURE_REGISTRY.
    """
    if spec.func not in FEATURE_REGISTRY:
        raise KeyError(f"Feature func '{spec.func}' non enregistrée.")

    builder = FEATURE_REGISTRY[spec.func]

    def fn(df: pl.DataFrame) -> pl.DataFrame:
        return builder(df, spec)

    return fn


def list_registered_features() -> List[str]:
    """Retourne la liste des noms de features enregistrées."""
    return sorted(FEATURE_REGISTRY.keys())


# -------------------------------------------------------------------
# Implémentations de features "de base"
# -------------------------------------------------------------------

# ---------- Returns ----------

@feature("ret")
def _feat_ret(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Return simple : (price_t / price_{t-h} - 1) par symbol.

    params:
        horizon: int (>=1), horizon de retour (par défaut 1).
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
    Log-return : log(price_t) - log(price_{t-h}) par symbol.

    params:
        horizon: int (>=1), horizon de retour (par défaut 1).
    """
    h = int(spec.params.get("horizon", 1))
    col = spec.col("price", default="close")

    return df.with_columns(
        (
            pl.col(col).log() - pl.col(col).log().shift(h)
        ).over("symbol").alias(spec.name)
    )


# ---------- Moving averages ----------

@feature("ma")
def _feat_ma(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Simple Moving Average (SMA) sur une colonne, par symbol.

    params:
        window: int, taille de fenêtre (obligatoire).
    """
    if "window" not in spec.params:
        raise ValueError(f"Feature '{spec.name}' (ma) nécessite params['window'].")

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
    Exponential Moving Average (EMA) sur une colonne, par symbol.

    params:
        alpha: float, OU
        span: float -> alpha = 2 / (span + 1)
    """
    col = spec.col("price", default="close")
    alpha = spec.params.get("alpha")
    span = spec.params.get("span")

    if alpha is None:
        if span is None:
            raise ValueError(
                f"Feature '{spec.name}' (ema) nécessite params['alpha'] ou params['span']."
            )
        alpha = 2.0 / (float(span) + 1.0)

    return df.with_columns(
        pl.col(col)
        .ewm_mean(alpha=float(alpha), adjust=False)
        .over("symbol")
        .alias(spec.name)
    )


# ---------- Rolling volatility ----------

@feature("rolling_vol")
def _feat_rolling_vol(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Volatilité rolling (écart-type) d'une série (souvent des returns).

    params:
        window: int, taille de fenêtre (obligatoire).
    """
    if "window" not in spec.params:
        raise ValueError(
            f"Feature '{spec.name}' (rolling_vol) nécessite params['window']."
        )

    w = int(spec.params["window"])
    col = spec.col("value", default="f_ret_1")  # par ex. ret_1

    return df.with_columns(
        pl.col(col)
        .rolling_std(window_size=w)
        .over("symbol")
        .alias(spec.name)
    )


# ---------- Cross-sectional (par date) ----------

@feature("cs_rank")
def _feat_cs_rank(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Rank cross-sectionnel d'une colonne à chaque timestamp.

    params:
        method: méthode de rank Polars ("dense","average","ordinal", etc.).
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
    Z-score cross-sectionnel d'une colonne à chaque timestamp :
    (x - mean_ts) / std_ts
    """
    col = spec.col("value", default="f_ret_1")

    mean_ts = pl.col(col).mean().over("ts")
    std_ts = pl.col(col).std().over("ts")

    return df.with_columns(
        ((pl.col(col) - mean_ts) / std_ts).alias(spec.name)
    )


# ---------- ATR : Average True Range ----------

@feature("atr")
def _feat_atr(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    Average True Range (ATR) par symbol.

    True Range (TR_t) :
        max(
            high_t - low_t,
            |high_t - close_{t-1}|,
            |low_t - close_{t-1}|
        )

    params:
        window: int, longueur ATR (ex. 14)
        method: "sma" ou "wilder" (par défaut "sma")
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

    # True Range par symbol
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
        # Approximation Wilder via EWMA alpha=1/window
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


# ---------- RSI : Relative Strength Index ----------

@feature("rsi")
def _feat_rsi(df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    """
    RSI (Relative Strength Index) sur le 'close' (par défaut), par symbol.

    params:
        window: int, période (ex. 14)
        method: "sma" ou "wilder" (par défaut "wilder")
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
    Bollinger Bands sur une série (par défaut close).

    params:
        window : int, taille de fenêtre (ex. 20)
        n_std  : float, multiplicateur d'écart-type (ex. 2.0)
        band   : "mid", "upper", "lower" (output de cette feature)
    """
    if "window" not in spec.params:
        raise ValueError(
            f"Feature '{spec.name}' (bollinger) nécessite params['window']."
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
    MACD classique sur une série (par défaut close), par symbol.

    params:
        fast   : int, période EMA rapide (ex. 12)
        slow   : int, période EMA lente (ex. 26)
        signal : int, période EMA de la ligne MACD (ex. 9)
        component : "macd", "signal", "hist"
    """
    col = spec.col("price", default="close")

    fast = int(spec.params.get("fast", 12))
    slow = int(spec.params.get("slow", 26))
    signal = int(spec.params.get("signal", 9))

    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)
    signal_alpha = 2.0 / (signal + 1.0)

    # EMA rapides et lentes
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

    # Nettoyage colonnes temporaires
    df = df.drop(
        "_macd_fast",
        "_macd_slow",
        "_macd_line",
        "_macd_signal",
        "_macd_hist",
    )
    return df


# -------------------------------------------------------------------
# Registration des features de base au chargement du module
# -------------------------------------------------------------------

# register_feature("ret", _feat_ret)
# register_feature("logret", _feat_logret)
# register_feature("ma", _feat_ma)
# register_feature("ema", _feat_ema)
# register_feature("rolling_vol", _feat_rolling_vol)
# register_feature("cs_rank", _feat_cs_rank)
# register_feature("cs_zscore", _feat_cs_zscore)
# register_feature("atr", _feat_atr)
# register_feature("rsi", _feat_rsi)
# register_feature("bollinger", _feat_bollinger)
# register_feature("macd", _feat_macd)
