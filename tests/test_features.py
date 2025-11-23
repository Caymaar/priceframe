import numpy as np
import pandas as pd
import polars as pl

from priceframe.core import PriceFrame
from priceframe.features import FeatureSpec, list_registered_features


def make_pf_single_symbol():
    """
    PriceFrame for a single symbol, daily, with a simple path.
    """
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    data = []
    close_vals = np.linspace(100, 110, len(idx))  # tendance haussière
    for ts, c in zip(idx, close_vals):
        o = c - 0.5
        h = c + 1.0
        l = c - 1.0
        v = 1000
        data.append(
            {
                "symbol": "AAA",
                "interval": "1d",
                "ts": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )
    df = pd.DataFrame(data)
    return PriceFrame.from_pandas(df)


def test_registry_contains_expected_features():
    names = list_registered_features()
    expected = {"ret", "logret", "ma", "ema", "rolling_vol", "cs_rank",
                "cs_zscore", "atr", "rsi", "bollinger", "macd"}
    # At least all of these must be present
    for e in expected:
        assert e in names


def test_ret_and_rolling_vol_with_specs():
    pf = make_pf_single_symbol()

    specs = [
        FeatureSpec(
            name="f_ret_1",
            func="ret",
            input_col="close",
            params={"horizon": 1},
        ),
        FeatureSpec(
            name="f_vol_3",
            func="rolling_vol",
            input_col="f_ret_1",
            params={"window": 3},
        ),
    ]

    pf_feat = pf.with_feature_specs(specs)
    out = pf_feat.to_pandas().sort_values("ts")

    # Check f_ret_1 ~ close/close.shift(1)-1
    close = out["close"].values
    ret_manual = np.empty_like(close)
    ret_manual[0] = np.nan
    ret_manual[1:] = close[1:] / close[:-1] - 1.0

    f_ret = out["f_ret_1"].values
    # Ignore first NaN value
    np.testing.assert_allclose(
        f_ret[1:], ret_manual[1:], rtol=1e-12, atol=1e-12
    )

    # Rolling vol > 0 after a few points
    assert np.nanmax(out["f_vol_3"].values) > 0


def test_atr_sma_matches_manual():
    """
    Build a small OHLC path, calculate ATR(3) via "atr" spec
    (SMA method), and compare to equivalent pandas calculation.
    """
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = []
    highs = [10, 11, 12, 13, 14]
    lows = [9, 9.5, 10, 11, 12]
    closes = [9.5, 10.5, 11, 12.5, 13]

    for ts, h, l, c in zip(idx, highs, lows, closes):
        data.append(
            {
                "symbol": "AAA",
                "interval": "1d",
                "ts": ts,
                "open": (h + l) / 2,
                "high": h,
                "low": l,
                "close": c,
                "volume": 0,
            }
        )
    df = pd.DataFrame(data)
    pf = PriceFrame.from_pandas(df)

    specs = [
        FeatureSpec(
            name="f_atr_3",
            func="atr",
            input_cols={"high": "high", "low": "low", "close": "close"},
            params={"window": 3, "method": "sma"},
        )
    ]

    pf_feat = pf.with_feature_specs(specs)
    out = pf_feat.to_pandas().sort_values("ts")

    # Manual TR + ATR(3) calculation with pandas
    g = df.sort_values("ts").copy()
    prev_close = g["close"].shift(1)

    tr = pd.concat(
        [
            g["high"] - g["low"],
            (g["high"] - prev_close).abs(),
            (g["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_manual = tr.rolling(3).mean()

    print(out["f_atr_3"].values)
    print(atr_manual.values)
    np.testing.assert_allclose(
        out["f_atr_3"].values,
        atr_manual.values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_rsi_trending_up_gives_high_value():
    pf = make_pf_single_symbol()

    specs = [
        FeatureSpec(
            name="f_rsi_14",
            func="rsi",
            input_cols={"close": "close"},
            params={"window": 5, "method": "wilder"},  # fenêtre courte pour test
        )
    ]

    pf_feat = pf.with_feature_specs(specs)
    out = pf_feat.to_pandas().sort_values("ts")

    # Last RSI should be > 70 on a strong uptrend
    last_rsi = out["f_rsi_14"].iloc[-1]
    assert last_rsi > 70


def test_bollinger_bands_consistency():
    pf = make_pf_single_symbol()

    specs = [
        FeatureSpec(
            name="f_bb_mid",
            func="bollinger",
            input_col="close",
            params={"window": 3, "n_std": 2.0, "band": "mid"},
        ),
        FeatureSpec(
            name="f_bb_up",
            func="bollinger",
            input_col="close",
            params={"window": 3, "n_std": 2.0, "band": "upper"},
        ),
        FeatureSpec(
            name="f_bb_low",
            func="bollinger",
            input_col="close",
            params={"window": 3, "n_std": 2.0, "band": "lower"},
        ),
    ]

    pf_feat = pf.with_feature_specs(specs)
    out = pf_feat.to_pandas().sort_values("ts")

    print(out.dtypes)
    # upper >= mid >= lower for all points
    bb_mid = out["f_bb_mid"].dropna().values
    bb_up = out["f_bb_up"].dropna().values
    bb_low = out["f_bb_low"].dropna().values

    assert np.all(bb_up >= bb_mid - 1e-12)
    assert np.all(bb_mid >= bb_low - 1e-12)


def test_macd_hist_non_trivial_on_trend():
    pf = make_pf_single_symbol()

    specs = [
        FeatureSpec(
            name="f_macd_hist",
            func="macd",
            input_col="close",
            params={"fast": 3, "slow": 6, "signal": 3, "component": "hist"},
        )
    ]

    pf_feat = pf.with_feature_specs(specs)
    out = pf_feat.to_pandas().sort_values("ts")

    hist = out["f_macd_hist"].values
    # We want at least one non-zero / non-NaN point after a few steps
    assert np.any(np.isfinite(hist[3:]) & (np.abs(hist[3:]) > 0))


def test_with_features_vs_with_feature_specs_equivalence_on_ret():
    """
    Verify that ret_1 via FeatureSpec gives the same result
    as a custom Polars implementation passed to with_features.
    """
    pf = make_pf_single_symbol()

    # 1) Via FeatureSpec
    spec = FeatureSpec(
        name="f_ret_1_spec",
        func="ret",
        input_col="close",
        params={"horizon": 1},
    )
    pf_spec = pf.with_feature_specs([spec])

    # 2) Via custom Polars function
    def f_ret_1(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (
                pl.col("close") / pl.col("close").shift(1) - 1.0
            ).over("symbol").alias("f_ret_1_fn")
        )

    pf_fn = pf.with_features([f_ret_1])

    df_spec = pf_spec.to_pandas().sort_values(["symbol", "ts"])
    df_fn = pf_fn.to_pandas().sort_values(["symbol", "ts"])

    # Compare from 2nd point (1st is NaN)
    s1 = df_spec["f_ret_1_spec"].values[1:]
    s2 = df_fn["f_ret_1_fn"].values[1:]

    np.testing.assert_allclose(s1, s2, rtol=1e-12, atol=1e-12)
