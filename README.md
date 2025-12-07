# PriceFrame

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![Version](https://img.shields.io/github/v/tag/Caymaar/priceframe?label=version)
[![Downloads](https://pepy.tech/badge/priceframe)](https://pepy.tech/project/priceframe)

A modern, efficient price data management library for Python, built on top of PyArrow, Polars, and pandas.

## Overview

PriceFrame provides a canonical, high-performance way to store, manipulate, and analyze financial price data (OHLCV). It combines the speed of PyArrow and Polars with the familiarity of pandas, offering a flexible feature system for technical indicators and quantitative analysis.

### Key Features

- **High Performance**: Built on PyArrow for efficient columnar storage and operations
- **Multiple Engine Support**: Seamlessly switch between pandas and Polars
- **Rich Feature System**: Declarative technical indicators (MA, EMA, RSI, ATR, MACD, Bollinger Bands, etc.)
- **Multiple Data Sources**: Bloomberg, custom databases, and more
- **Flexible Time Handling**: Automatic timezone normalization and resampling
- **Portfolio Analysis**: Built-in naive portfolio construction and rebalancing
- **Type-Safe**: Canonical schema with automatic type casting and validation

## Installation

### Basic Installation

```bash
pip install priceframe
```

### With Optional Dependencies

For Bloomberg Terminal support:
```bash
pip install priceframe[bloomberg]
```

For cryptocurrency data (via CCXT):
```bash
pip install priceframe[crypto]
```

All optional dependencies:
```bash
pip install priceframe[bloomberg,crypto]
```

## Quick Start

### Creating a PriceFrame

```python
import pandas as pd
from priceframe import PriceFrame

# From pandas DataFrame
df = pd.DataFrame({
    'symbol': ['AAPL', 'AAPL', 'AAPL'],
    'ts': pd.date_range('2024-01-01', periods=3, tz='UTC'),
    'open': [150.0, 151.0, 152.0],
    'high': [151.0, 152.0, 153.0],
    'low': [149.0, 150.0, 151.0],
    'close': [150.5, 151.5, 152.5],
    'volume': [1000000, 1100000, 1050000]
})

pf = PriceFrame.from_pandas(df, interval='1d')
```

### Loading from Bloomberg

```python
# Single ticker
pf = PriceFrame.from_bloomberg(
    tickers='AAPL US Equity',
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1d'
)

# Multiple tickers
pf = PriceFrame.from_bloomberg(
    tickers=['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity'],
    start_date='2024-01-01',
    type_='ohlcv',
    currency='USD'
)
```

### Adding Technical Indicators

```python
from priceframe.features import FeatureSpec

# Define features declaratively
features = [
    # 20-day moving average
    FeatureSpec(
        name='f_ma_20',
        func='ma',
        input_col='close',
        params={'window': 20}
    ),
    
    # RSI(14)
    FeatureSpec(
        name='f_rsi_14',
        func='rsi',
        input_cols={'close': 'close'},
        params={'window': 14, 'method': 'wilder'}
    ),
    
    # ATR(14)
    FeatureSpec(
        name='f_atr_14',
        func='atr',
        input_cols={'high': 'high', 'low': 'low', 'close': 'close'},
        params={'window': 14, 'method': 'sma'}
    ),
    
    # Daily returns
    FeatureSpec(
        name='f_ret_1',
        func='ret',
        input_col='close',
        params={'horizon': 1}
    )
]

# Apply features
pf_with_features = pf.with_feature_specs(features)

# Convert to pandas for analysis
df_result = pf_with_features.to_pandas()
```

### Data Access

```python
# Get OHLCV for a specific symbol
ohlcv = pf.ohlcv('AAPL US Equity', engine='pandas')

# Get close prices only
closes = pf.close_series('AAPL US Equity')

# Create a close price matrix (timestamps × symbols)
close_matrix = pf.close_matrix(
    symbols=['AAPL US Equity', 'MSFT US Equity'],
    how='inner'
)
```

### Time Operations

```python
# Filter by date range
pf_filtered = pf.range_date(
    start='2024-03-01',
    end='2024-06-30',
    closed='both'
)

# Resample to weekly
pf_weekly = pf.resample('1w')

# Rebase to 100
pf_rebased = pf.rebase(base=100.0, anchor='close')
```

### Portfolio Construction

```python
# Create a naive portfolio
portfolio = pf.naive_portfolio(
    symbols=['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity'],
    weights=[0.4, 0.3, 0.3],
    base=100.0,
    name='Tech Portfolio',
    as_='series'
)
```

### Merging PriceFrames

```python
# Combine two PriceFrames
pf_combined = pf1.add(pf2, require_same_interval=True)
```

## Available Technical Indicators

PriceFrame includes a comprehensive feature registry with the following indicators:

### Returns
- **ret**: Simple returns
- **logret**: Log returns

### Moving Averages
- **ma**: Simple Moving Average (SMA)
- **ema**: Exponential Moving Average (EMA)

### Volatility
- **rolling_vol**: Rolling standard deviation

### Cross-Sectional
- **cs_rank**: Cross-sectional rank
- **cs_zscore**: Cross-sectional z-score

### Technical Indicators
- **atr**: Average True Range
- **rsi**: Relative Strength Index
- **bollinger**: Bollinger Bands (mid, upper, lower)
- **macd**: MACD (macd, signal, histogram)

### Custom Features

You can register your own features:

```python
from priceframe.features import feature
import polars as pl

@feature("my_custom_indicator")
def _feat_custom(df: pl.DataFrame, spec) -> pl.DataFrame:
    """Custom indicator implementation."""
    window = spec.params.get('window', 10)
    col = spec.col('price', default='close')
    
    return df.with_columns(
        pl.col(col).rolling_mean(window).over('symbol').alias(spec.name)
    )

# Use it
spec = FeatureSpec(
    name='f_custom_10',
    func='my_custom_indicator',
    input_col='close',
    params={'window': 10}
)
```

## Data Format

PriceFrame uses a canonical long-format schema:

| Column   | Type                     | Description                    |
|----------|--------------------------|--------------------------------|
| symbol   | string                   | Instrument identifier          |
| interval | string                   | Time interval (e.g., '1d', '1h')|
| ts       | timestamp[ms, tz='UTC']  | Timestamp                      |
| open     | float64                  | Opening price                  |
| high     | float64                  | Highest price                  |
| low      | float64                  | Lowest price                   |
| close    | float64                  | Closing price                  |
| volume   | float64                  | Trading volume                 |

Additional columns (features, metadata) are preserved.

## Conversion Between Formats

```python
# To pandas
df_pandas = pf.to_pandas(index='ts')  # or index='multi' for MultiIndex

# To Polars
df_polars = pf.to_polars()

# To PyArrow
table = pf.to_arrow()

# To Excel
pf.to_excel('output.xlsx')

# From close matrix
close_df = pd.DataFrame({
    'AAPL': [150, 151, 152],
    'MSFT': [300, 301, 302]
}, index=pd.date_range('2024-01-01', periods=3, tz='UTC'))

pf = PriceFrame.from_close_matrix(close_df, interval='1d')
```

## Architecture

PriceFrame is built on three main components:

1. **Core (`priceframe.core`)**: The `PriceFrame` class and data manipulation functions
2. **Features (`priceframe.features`)**: Declarative feature system with registry
3. **IO (`priceframe.io`)**: Data source connectors (Bloomberg, databases, etc.)

### Design Principles

- **Immutability**: Most operations return new PriceFrame instances
- **Type Safety**: Automatic type casting and validation
- **Performance**: PyArrow backend for efficient columnar operations
- **Flexibility**: Support for both pandas and Polars workflows

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Roadmap

- [ ] Additional data sources (Yahoo Finance, Binance, S3 storage, etc.)
- [ ] More technical indicators
- [ ] Backtesting framework integration (NautilusTrader)
- [ ] Performance optimizations
- [ ] Quick visualisation with mlpfinance
