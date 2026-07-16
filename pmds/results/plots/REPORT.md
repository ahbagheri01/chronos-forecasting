# Chronos Forecasting Benchmark Report

## 📊 Datasets

### Chronos M1 Yearly

**Domain**: Finance/Economics

Monash M1 Competition yearly dataset

**Key Characteristics**:
- Yearly frequency data
- 20 different time series
- 6-step ahead forecasting horizon
- Low seasonality (1)
- Long-term trends and patterns

**Use Case**: Annual business and economic forecasting

---

### Chronos M4 Hourly

**Domain**: Energy/Traffic

M4 Competition hourly dataset

**Key Characteristics**:
- Hourly frequency data
- 20 different time series
- 48-step ahead forecasting horizon (2 days)
- Strong hourly seasonality (24)
- Repeating daily patterns

**Use Case**: Short-term operational forecasting (energy demand, traffic)

---

### Chronos Weather

**Domain**: Meteorology

Monash weather dataset

**Key Characteristics**:
- Daily frequency data
- 20 different weather stations
- 30-day forecasting horizon
- Weekly seasonality (7)
- Temporal weather patterns

**Use Case**: Weather forecasting for multiple locations

---

### National Illness

**Domain**: Healthcare

National illness dataset from Time-Series-Library

**Key Characteristics**:
- Weekly frequency data
- 1 single target variable (OT)
- 24-week forecasting horizon (~6 months)
- Yearly seasonality (52 weeks)
- Epidemic patterns

**Use Case**: Disease tracking and epidemiological forecasting

---

### Traffic

**Domain**: Transportation

Traffic flow dataset from Time-Series-Library

**Key Characteristics**:
- Hourly frequency data
- 5 different traffic sensors
- 24-hour ahead forecasting
- Daily seasonality (24)
- Vehicle flow patterns

**Use Case**: Traffic management and congestion prediction

---

### PSM (Power Supply Monitoring)

**Domain**: Manufacturing/Energy

Power Supply Monitoring dataset

**Key Characteristics**:
- Hourly frequency data
- 5 different power metrics
- 24-hour ahead forecasting
- Daily seasonality (24)
- Equipment monitoring data

**Use Case**: Industrial equipment performance monitoring

---

## 📈 Evaluation Metrics

| Metric | Direction | Unit | Interpretation |
|--------|-----------|------|----------------|
| **Mean Absolute Error (MAE)** | ↓ Lower Better | Same unit as target variable | Penalizes all errors equally, not sensitive to outliers |
| **Root Mean Squared Error (RMSE)** | ↓ Lower Better | Same unit as target variable | More sensitive to outliers and large errors than MAE |
| **Symmetric Mean Absolute Percentage Error (SMAPE)** | ↓ Lower Better | Percentage (%) | Good for comparing across series with different scales. Range: 0-200% |
| **Mean Absolute Scaled Error (MASE)** | ↓ Lower Better | Dimensionless ratio | MASE < 1 means better than naive baseline, > 1 means worse |
| **Weighted Quantile Loss (WQL)** | ↓ Lower Better | Same unit as target variable | Evaluates entire forecast distribution, not just point estimates |

### Detailed Metric Explanations

#### Mean Absolute Error (MAE)
**Formula**: `mean(|actual - predicted|)`

Average magnitude of errors. Simple and interpretable.

**Interpretation**: Penalizes all errors equally, not sensitive to outliers

#### Root Mean Squared Error (RMSE)
**Formula**: `sqrt(mean((actual - predicted)²))`

Square root of average squared errors. Penalizes large errors more heavily.

**Interpretation**: More sensitive to outliers and large errors than MAE

#### Symmetric Mean Absolute Percentage Error (SMAPE)
**Formula**: `mean(2 * |actual - predicted| / (|actual| + |predicted|)) * 100`

Scale-independent percentage error that is symmetric between actual and predicted.

**Interpretation**: Good for comparing across series with different scales. Range: 0-200%

#### Mean Absolute Scaled Error (MASE)
**Formula**: `mean(|actual - predicted|) / mean(|actual[t] - actual[t-seasonality]|)`

Scales errors relative to naive seasonal forecast. MASE=1 means same as naive.

**Interpretation**: MASE < 1 means better than naive baseline, > 1 means worse

#### Weighted Quantile Loss (WQL)
**Formula**: `Weighted sum of quantile losses across [0.1, 0.2, ..., 0.9]`

Loss function for probabilistic forecasts across multiple quantiles.

**Interpretation**: Evaluates entire forecast distribution, not just point estimates

## 🤖 Forecasting Models

### Chronos T5 Tiny

**Type**: Deep Learning - Pretrained Transformer

**Parameters**: ~8M

Tiny Chronos model - pretrained T5 transformer for time series forecasting

**Features**:
- Language model architecture (T5)
- Pretrained on large time series datasets
- Zero-shot forecasting capability
- Probabilistic forecasts (100 samples)
- CPU-friendly (8M parameters)
- Quantization support available

**Strengths**:
- ✅ Good performance across diverse datasets
- ✅ Fast inference
- ✅ Transfer learning from pretraining
- ✅ Handles multiple time series naturally

**Weaknesses**:
- ❌ Less powerful than larger Chronos variants
- ❌ Still heavier than traditional methods

---

### Seasonal Naive

**Type**: Baseline - Naive

**Parameters**: ~0

Baseline model - predicts by repeating last known seasonal period

**Features**:
- No model training required
- Repeats values from same season
- Fast computation
- No parameters to tune
- Deterministic forecasts

**Strengths**:
- ✅ Excellent baseline for seasonal data
- ✅ Very fast
- ✅ Simple to understand
- ✅ Often beats more complex methods on seasonal data

**Weaknesses**:
- ❌ Cannot capture trends
- ❌ Ignores recent observations
- ❌ Poor for data without clear seasonality

---

### AR(2)

**Type**: Statistical - Autoregressive

**Parameters**: ~2

Autoregressive model of order 2 - predicts based on last 2 observations

**Features**:
- Depends on previous 2 time steps
- No differencing (stationary data)
- Linear constant trend
- Statsmodels implementation

**Strengths**:
- ✅ Captures short-term dependencies
- ✅ Fast and lightweight
- ✅ Interpretable coefficients

**Weaknesses**:
- ❌ Only uses 2 previous values
- ❌ No seasonality modeling
- ❌ Requires stationary data

---

### MA(2)

**Type**: Statistical - Moving Average

**Parameters**: ~2

Moving Average model of order 2 - models forecast errors of last 2 steps

**Features**:
- Models residual errors from last 2 steps
- No differencing
- Linear constant trend
- Statsmodels implementation

**Strengths**:
- ✅ Good for irregular shocks
- ✅ Can smooth noise
- ✅ Lightweight

**Weaknesses**:
- ❌ Purely reactive to past errors
- ❌ No seasonality
- ❌ Only 2 lagged errors

---

### ARMA(2,2)

**Type**: Statistical - Mixed

**Parameters**: ~4

AutoRegressive-Moving-Average model combining AR(2) and MA(2)

**Features**:
- 2 autoregressive terms + 2 moving average terms
- Combines AR and MA benefits
- No differencing
- Linear constant trend

**Strengths**:
- ✅ Flexible modeling of dependencies
- ✅ Combines memory and error correction
- ✅ Better than pure AR or MA

**Weaknesses**:
- ❌ More parameters to estimate
- ❌ No seasonality
- ❌ Can be harder to fit

---

### ARIMA(2,1,2)

**Type**: Statistical - Integrated

**Parameters**: ~4

AutoRegressive Integrated Moving-Average - includes 1-step differencing for non-stationary data

**Features**:
- 2 AR terms + 2 MA terms
- 1 differencing step (I=1)
- Handles non-stationary trends
- No trend component

**Strengths**:
- ✅ Handles trending data
- ✅ Differencing removes trend automatically
- ✅ Balanced AR and MA

**Weaknesses**:
- ❌ Ignores seasonality
- ❌ Differencing loses information
- ❌ Limited to linear trends

---

### Prophet

**Type**: Statistical - Time Series Decomposition

**Parameters**: ~~10-20

Facebook's Prophet - decomposes time series into trend, seasonality, and holidays

**Features**:
- Additive seasonality
- Linear trend model
- Automatic seasonality detection
- Holiday effects support
- Bayesian inference
- 200 posterior samples for uncertainty

**Strengths**:
- ✅ Good at capturing multiple seasonalities
- ✅ Robust to missing data
- ✅ Good uncertainty estimates
- ✅ Interpretable components

**Weaknesses**:
- ❌ Slower than simpler methods
- ❌ Assumes linear trend
- ❌ May overfit on small datasets

---

## 📋 Summary

This benchmark compares forecasting models across diverse time series datasets.

### Model Categories

1. **Deep Learning**: Chronos T5 - Pretrained transformer model
2. **Baselines**: Seasonal Naive - Simple repeating pattern baseline
3. **Statistical**: AR/MA/ARMA/ARIMA/Prophet - Classical time series methods

### Key Insights

- **Seasonal Naive** serves as an important baseline for seasonal data
- **Chronos T5** leverages deep learning and pretraining for better generalization
- **Statistical methods** (ARIMA, Prophet) provide interpretability and can work well on specific data types
- **Lower metrics are better** for MAE, RMSE, SMAPE, MASE, and WQL
- **MASE=1** represents the baseline seasonal naive forecast performance
- **WQL** evaluates entire probabilistic forecast distributions

---

*Generated from PMDS Chronos Forecasting Benchmark*
