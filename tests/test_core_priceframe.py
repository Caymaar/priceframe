import numpy as np
import pandas as pd
import pytest

from priceframe.core import PriceFrame


def make_basic_df():
    """
    Small daily OHLCV DataFrame for 2 symbols.
    """
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = []
    for sym in ["AAA", "BBB"]:
        for i, ts in enumerate(idx):
            o = 100 + i
            h = o + 1
            l = o - 1
            c = o + 0.5
            v = 1000 + 10 * i
            data.append(
                {
                    "symbol": sym,
                    "interval": "1d",
                    "ts": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )
    return pd.DataFrame(data)


def test_from_pandas_round_trip():
    df = make_basic_df()
    pf = PriceFrame.from_pandas(df)

    out = pf.to_pandas()

    # Minimal columns present
    for col in ["symbol", "interval", "ts", "open", "high", "low", "close", "volume"]:
        assert col in out.columns

    # ts in datetime64[ns, UTC]
    print(str(out["ts"].dtype))
    assert str(out["ts"].dtype) == "datetime64[ns, UTC]"

    # No duplicates on (symbol, interval, ts)
    assert (
        out.duplicated(subset=["symbol", "interval", "ts"]).sum() == 0
    )

    # Close values preserved
    pd.testing.assert_series_equal(
        df.sort_values(["symbol", "interval", "ts"])["close"].reset_index(drop=True),
        out.sort_values(["symbol", "interval", "ts"])["close"].reset_index(drop=True),
        check_names=False,
    )


def test_range_date_filters_correctly():
    df = make_basic_df()
    pf = PriceFrame.from_pandas(df)

    start = "2024-01-02"
    end = "2024-01-04"
    pf2 = pf.range_date(start=start, end=end)

    out = pf2.to_pandas()
    assert out["ts"].min() >= pd.Timestamp(start, tz="UTC")
    assert out["ts"].max() <= pd.Timestamp(end, tz="UTC")


def test_close_matrix_basic_pandas():
    df = make_basic_df()
    pf = PriceFrame.from_pandas(df)

    cm = pf.close_matrix(engine="pandas")
    # index = ts, columns = symbols
    assert list(sorted(cm.columns)) == ["AAA", "BBB"]
    # 5 dates
    assert len(cm) == 5
    # AAA close[0] = 100.5
    assert np.isclose(cm["AAA"].iloc[0], 100.5)


def test_resample_daily_to_weekly():
    # 5 days of a single symbol
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = []
    for i, ts in enumerate(idx):
        o = 100 + i
        h = o + 2
        l = o - 2
        c = o + 0.5
        v = 10 + i
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
    pf = PriceFrame.from_pandas(df)

    # Resample to weekly
    pf_w = pf.resample("1w")
    out = pf_w.to_pandas().sort_values("ts")

    # Single weekly bar
    assert len(out) == 1
    row = out.iloc[0]

    # open = first day's open
    assert np.isclose(row["open"], df.iloc[0]["open"])
    # close = last day's close
    assert np.isclose(row["close"], df.iloc[-1]["close"])
    # high = max high
    assert np.isclose(row["high"], df["high"].max())
    # low = min low
    assert np.isclose(row["low"], df["low"].min())
    # volume = sum
    assert np.isclose(row["volume"], df["volume"].sum())
    # interval updated
    assert row["interval"] == "1w"


def test_naive_portfolio_series():
    # 3 days, 2 symbols
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    data = []
    # AAA: 100, 101, 102
    for i, ts in enumerate(idx):
        data.append(
            {
                "symbol": "AAA",
                "interval": "1d",
                "ts": ts,
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 100 + i,
                "volume": 0,
            }
        )
    # BBB: 200, 198, 204
    closes_bbb = [200, 198, 204]
    for i, ts in enumerate(idx):
        data.append(
            {
                "symbol": "BBB",
                "interval": "1d",
                "ts": ts,
                "open": 0,
                "high": 0,
                "low": 0,
                "close": closes_bbb[i],
                "volume": 0,
            }
        )
    df = pd.DataFrame(data)
    pf = PriceFrame.from_pandas(df)

    symbols = ["AAA", "BBB"]
    weights = [0.5, 0.5]
    base = 100.0

    pf_idx = pf.naive_portfolio(symbols, weights, base=base, as_="series")

    # index length = 3
    assert len(pf_idx) == 3
    # first value = base
    assert np.isclose(pf_idx.iloc[0], base)

    # Manual calculation
    cm = pf.close_matrix(symbols=symbols, engine="pandas")
    R = cm.pct_change().fillna(0.0)
    r_p = (R * np.array(weights)).sum(axis=1)
    idx_manual = (1 + r_p).cumprod() * base

    np.testing.assert_allclose(
        pf_idx.values,
        idx_manual.values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_add_merge_and_dedup():
    # pf1: AAA avec 3 dates
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    data1 = []
    for ts in idx:
        data1.append(
            {
                "symbol": "AAA",
                "interval": "1d",
                "ts": ts,
                "open": 1,
                "high": 2,
                "low": 0,
                "close": 1.5,
                "volume": 10,
            }
        )
    df1 = pd.DataFrame(data1)
    pf1 = PriceFrame.from_pandas(df1)

    # pf2: AAA avec recouvrement sur les 2 dernières dates + une nouvelle
    idx2 = pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC")
    data2 = []
    for ts in idx2:
        data2.append(
            {
                "symbol": "AAA",
                "interval": "1d",
                "ts": ts,
                "open": 10,
                "high": 20,
                "low": 0,
                "close": 15,
                "volume": 100,
            }
        )
    df2 = pd.DataFrame(data2)
    pf2 = PriceFrame.from_pandas(df2)

    pf_comb = pf1.add(pf2)
    out = pf_comb.to_pandas().sort_values("ts")

    # Should contain 4 dates (1x 01, 1x 02, 1x 03, 1x 04)
    assert len(out["ts"].unique()) == 4

    # First occurrence (pf1) should be kept for 02 & 03
    sub = out[out["ts"] == pd.Timestamp("2024-01-02", tz="UTC")]
    assert len(sub) == 1
    # close is 1.5 (pf1), not 15 (pf2)
    assert np.isclose(sub["close"].iloc[0], 1.5)


def test_add_require_same_interval_error():
    # pf1 interval 1d, pf2 interval 1h
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    df1 = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "interval": "1d",
                "ts": ts,
                "open": 1,
                "high": 2,
                "low": 0,
                "close": 1.5,
                "volume": 10,
            }
        ]
    )
    df2 = df1.copy()
    df2["interval"] = "1h"

    pf1 = PriceFrame.from_pandas(df1)
    pf2 = PriceFrame.from_pandas(df2)

    with pytest.raises(ValueError):
        _ = pf1.add(pf2, require_same_interval=True)


def test_from_close_matrix_basic():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    close_df = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0],
            "BBB": [50.0, 49.0, 51.0],
        },
        index=idx,
    )

    # ts_is_close=False to keep index = ts
    pf = PriceFrame.from_close_matrix(close_df)

    out = pf.to_pandas()

    # Columns OK
    assert set(["symbol", "interval", "ts", "close"]).issubset(out.columns)

    # Inferred interval = 1d
    assert out["interval"].unique().tolist() == ["1d"]

    # 3 dates * 2 symbols
    assert len(out) == 6

    # AAA close
    aaa = out[out["symbol"] == "AAA"].sort_values("ts")
    np.testing.assert_allclose(
        aaa["close"].values,
        [100.0, 101.0, 102.0],
    )

    # ts corresponds to original index
    assert set(aaa["ts"]) == set(idx)
