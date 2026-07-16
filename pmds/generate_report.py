#!/usr/bin/env python
"""Generate comprehensive report explaining datasets, metrics, and models."""

import json
from pathlib import Path
from typing import Any


# Dataset descriptions
DATASET_DESCRIPTIONS = {
    "chronos_m1_yearly": {
        "name": "Chronos M1 Yearly",
        "description": "Monash M1 Competition yearly dataset",
        "domain": "Finance/Economics",
        "characteristics": [
            "Yearly frequency data",
            "20 different time series",
            "6-step ahead forecasting horizon",
            "Low seasonality (1)",
            "Long-term trends and patterns"
        ],
        "use_case": "Annual business and economic forecasting"
    },
    "chronos_m4_hourly": {
        "name": "Chronos M4 Hourly",
        "description": "M4 Competition hourly dataset",
        "domain": "Energy/Traffic",
        "characteristics": [
            "Hourly frequency data",
            "20 different time series",
            "48-step ahead forecasting horizon (2 days)",
            "Strong hourly seasonality (24)",
            "Repeating daily patterns"
        ],
        "use_case": "Short-term operational forecasting (energy demand, traffic)"
    },
    "chronos_weather": {
        "name": "Chronos Weather",
        "description": "Monash weather dataset",
        "domain": "Meteorology",
        "characteristics": [
            "Daily frequency data",
            "20 different weather stations",
            "30-day forecasting horizon",
            "Weekly seasonality (7)",
            "Temporal weather patterns"
        ],
        "use_case": "Weather forecasting for multiple locations"
    },
    "external_national_illness": {
        "name": "National Illness",
        "description": "National illness dataset from Time-Series-Library",
        "domain": "Healthcare",
        "characteristics": [
            "Weekly frequency data",
            "1 single target variable (OT)",
            "24-week forecasting horizon (~6 months)",
            "Yearly seasonality (52 weeks)",
            "Epidemic patterns"
        ],
        "use_case": "Disease tracking and epidemiological forecasting"
    },
    "external_traffic": {
        "name": "Traffic",
        "description": "Traffic flow dataset from Time-Series-Library",
        "domain": "Transportation",
        "characteristics": [
            "Hourly frequency data",
            "5 different traffic sensors",
            "24-hour ahead forecasting",
            "Daily seasonality (24)",
            "Vehicle flow patterns"
        ],
        "use_case": "Traffic management and congestion prediction"
    },
    "external_psm": {
        "name": "PSM (Power Supply Monitoring)",
        "description": "Power Supply Monitoring dataset",
        "domain": "Manufacturing/Energy",
        "characteristics": [
            "Hourly frequency data",
            "5 different power metrics",
            "24-hour ahead forecasting",
            "Daily seasonality (24)",
            "Equipment monitoring data"
        ],
        "use_case": "Industrial equipment performance monitoring"
    }
}

# Metric descriptions
METRIC_DESCRIPTIONS = {
    "mae": {
        "name": "Mean Absolute Error (MAE)",
        "formula": "mean(|actual - predicted|)",
        "lower_is_better": True,
        "unit": "Same unit as target variable",
        "description": "Average magnitude of errors. Simple and interpretable.",
        "interpretation": "Penalizes all errors equally, not sensitive to outliers"
    },
    "rmse": {
        "name": "Root Mean Squared Error (RMSE)",
        "formula": "sqrt(mean((actual - predicted)²))",
        "lower_is_better": True,
        "unit": "Same unit as target variable",
        "description": "Square root of average squared errors. Penalizes large errors more heavily.",
        "interpretation": "More sensitive to outliers and large errors than MAE"
    },
    "smape": {
        "name": "Symmetric Mean Absolute Percentage Error (SMAPE)",
        "formula": "mean(2 * |actual - predicted| / (|actual| + |predicted|)) * 100",
        "lower_is_better": True,
        "unit": "Percentage (%)",
        "description": "Scale-independent percentage error that is symmetric between actual and predicted.",
        "interpretation": "Good for comparing across series with different scales. Range: 0-200%"
    },
    "mase": {
        "name": "Mean Absolute Scaled Error (MASE)",
        "formula": "mean(|actual - predicted|) / mean(|actual[t] - actual[t-seasonality]|)",
        "lower_is_better": True,
        "unit": "Dimensionless ratio",
        "description": "Scales errors relative to naive seasonal forecast. MASE=1 means same as naive.",
        "interpretation": "MASE < 1 means better than naive baseline, > 1 means worse"
    },
    "wql": {
        "name": "Weighted Quantile Loss (WQL)",
        "formula": "Weighted sum of quantile losses across [0.1, 0.2, ..., 0.9]",
        "lower_is_better": True,
        "unit": "Same unit as target variable",
        "description": "Loss function for probabilistic forecasts across multiple quantiles.",
        "interpretation": "Evaluates entire forecast distribution, not just point estimates"
    }
}

# Model descriptions
MODEL_DESCRIPTIONS = {
    "chronos_t5_tiny": {
        "name": "Chronos T5 Tiny",
        "type": "Deep Learning - Pretrained Transformer",
        "family": "Chronos",
        "parameters": "8M",
        "description": "Tiny Chronos model - pretrained T5 transformer for time series forecasting",
        "features": [
            "Language model architecture (T5)",
            "Pretrained on large time series datasets",
            "Zero-shot forecasting capability",
            "Probabilistic forecasts (100 samples)",
            "CPU-friendly (8M parameters)",
            "Quantization support available"
        ],
        "strengths": [
            "Good performance across diverse datasets",
            "Fast inference",
            "Transfer learning from pretraining",
            "Handles multiple time series naturally"
        ],
        "weaknesses": [
            "Less powerful than larger Chronos variants",
            "Still heavier than traditional methods"
        ]
    },
    "seasonal_naive": {
        "name": "Seasonal Naive",
        "type": "Baseline - Naive",
        "family": "Simple",
        "parameters": "0",
        "description": "Baseline model - predicts by repeating last known seasonal period",
        "features": [
            "No model training required",
            "Repeats values from same season",
            "Fast computation",
            "No parameters to tune",
            "Deterministic forecasts"
        ],
        "strengths": [
            "Excellent baseline for seasonal data",
            "Very fast",
            "Simple to understand",
            "Often beats more complex methods on seasonal data"
        ],
        "weaknesses": [
            "Cannot capture trends",
            "Ignores recent observations",
            "Poor for data without clear seasonality"
        ]
    },
    "ar_2": {
        "name": "AR(2)",
        "type": "Statistical - Autoregressive",
        "family": "ARIMA",
        "parameters": "2",
        "description": "Autoregressive model of order 2 - predicts based on last 2 observations",
        "features": [
            "Depends on previous 2 time steps",
            "No differencing (stationary data)",
            "Linear constant trend",
            "Statsmodels implementation"
        ],
        "strengths": [
            "Captures short-term dependencies",
            "Fast and lightweight",
            "Interpretable coefficients"
        ],
        "weaknesses": [
            "Only uses 2 previous values",
            "No seasonality modeling",
            "Requires stationary data"
        ]
    },
    "ma_2": {
        "name": "MA(2)",
        "type": "Statistical - Moving Average",
        "family": "ARIMA",
        "parameters": "2",
        "description": "Moving Average model of order 2 - models forecast errors of last 2 steps",
        "features": [
            "Models residual errors from last 2 steps",
            "No differencing",
            "Linear constant trend",
            "Statsmodels implementation"
        ],
        "strengths": [
            "Good for irregular shocks",
            "Can smooth noise",
            "Lightweight"
        ],
        "weaknesses": [
            "Purely reactive to past errors",
            "No seasonality",
            "Only 2 lagged errors"
        ]
    },
    "arma_2_2": {
        "name": "ARMA(2,2)",
        "type": "Statistical - Mixed",
        "family": "ARIMA",
        "parameters": "4",
        "description": "AutoRegressive-Moving-Average model combining AR(2) and MA(2)",
        "features": [
            "2 autoregressive terms + 2 moving average terms",
            "Combines AR and MA benefits",
            "No differencing",
            "Linear constant trend"
        ],
        "strengths": [
            "Flexible modeling of dependencies",
            "Combines memory and error correction",
            "Better than pure AR or MA"
        ],
        "weaknesses": [
            "More parameters to estimate",
            "No seasonality",
            "Can be harder to fit"
        ]
    },
    "arima_2_1_2": {
        "name": "ARIMA(2,1,2)",
        "type": "Statistical - Integrated",
        "family": "ARIMA",
        "parameters": "4",
        "description": "AutoRegressive Integrated Moving-Average - includes 1-step differencing for non-stationary data",
        "features": [
            "2 AR terms + 2 MA terms",
            "1 differencing step (I=1)",
            "Handles non-stationary trends",
            "No trend component"
        ],
        "strengths": [
            "Handles trending data",
            "Differencing removes trend automatically",
            "Balanced AR and MA"
        ],
        "weaknesses": [
            "Ignores seasonality",
            "Differencing loses information",
            "Limited to linear trends"
        ]
    },
    "prophet": {
        "name": "Prophet",
        "type": "Statistical - Time Series Decomposition",
        "family": "Facebook Prophet",
        "parameters": "~10-20",
        "description": "Facebook's Prophet - decomposes time series into trend, seasonality, and holidays",
        "features": [
            "Additive seasonality",
            "Linear trend model",
            "Automatic seasonality detection",
            "Holiday effects support",
            "Bayesian inference",
            "200 posterior samples for uncertainty"
        ],
        "strengths": [
            "Good at capturing multiple seasonalities",
            "Robust to missing data",
            "Good uncertainty estimates",
            "Interpretable components"
        ],
        "weaknesses": [
            "Slower than simpler methods",
            "Assumes linear trend",
            "May overfit on small datasets"
        ]
    }
}


def create_markdown_report(output_path: Path):
    """Create markdown report."""
    report = "# Chronos Forecasting Benchmark Report\n\n"
    
    # Datasets section
    report += "## 📊 Datasets\n\n"
    for dataset_key, dataset_info in DATASET_DESCRIPTIONS.items():
        report += f"### {dataset_info['name']}\n\n"
        report += f"**Domain**: {dataset_info['domain']}\n\n"
        report += f"{dataset_info['description']}\n\n"
        report += "**Key Characteristics**:\n"
        for char in dataset_info['characteristics']:
            report += f"- {char}\n"
        report += f"\n**Use Case**: {dataset_info['use_case']}\n\n"
        report += "---\n\n"
    
    # Metrics section
    report += "## 📈 Evaluation Metrics\n\n"
    report += "| Metric | Direction | Unit | Interpretation |\n"
    report += "|--------|-----------|------|----------------|\n"
    for metric_key, metric_info in METRIC_DESCRIPTIONS.items():
        direction = "↓ Lower Better" if metric_info['lower_is_better'] else "↑ Higher Better"
        report += f"| **{metric_info['name']}** | {direction} | {metric_info['unit']} | {metric_info['interpretation']} |\n"
    
    report += "\n### Detailed Metric Explanations\n\n"
    for metric_key, metric_info in METRIC_DESCRIPTIONS.items():
        report += f"#### {metric_info['name']}\n"
        report += f"**Formula**: `{metric_info['formula']}`\n\n"
        report += f"{metric_info['description']}\n\n"
        report += f"**Interpretation**: {metric_info['interpretation']}\n\n"
    
    # Models section
    report += "## 🤖 Forecasting Models\n\n"
    for model_key, model_info in MODEL_DESCRIPTIONS.items():
        report += f"### {model_info['name']}\n\n"
        report += f"**Type**: {model_info['type']}\n\n"
        report += f"**Parameters**: ~{model_info['parameters']}\n\n"
        report += f"{model_info['description']}\n\n"
        
        report += "**Features**:\n"
        for feature in model_info['features']:
            report += f"- {feature}\n"
        
        report += "\n**Strengths**:\n"
        for strength in model_info['strengths']:
            report += f"- ✅ {strength}\n"
        
        report += "\n**Weaknesses**:\n"
        for weakness in model_info['weaknesses']:
            report += f"- ❌ {weakness}\n"
        
        report += "\n---\n\n"
    
    # Summary
    report += "## 📋 Summary\n\n"
    report += "This benchmark compares forecasting models across diverse time series datasets.\n\n"
    report += "### Model Categories\n\n"
    report += "1. **Deep Learning**: Chronos T5 - Pretrained transformer model\n"
    report += "2. **Baselines**: Seasonal Naive - Simple repeating pattern baseline\n"
    report += "3. **Statistical**: AR/MA/ARMA/ARIMA/Prophet - Classical time series methods\n\n"
    
    report += "### Key Insights\n\n"
    report += "- **Seasonal Naive** serves as an important baseline for seasonal data\n"
    report += "- **Chronos T5** leverages deep learning and pretraining for better generalization\n"
    report += "- **Statistical methods** (ARIMA, Prophet) provide interpretability and can work well on specific data types\n"
    report += "- **Lower metrics are better** for MAE, RMSE, SMAPE, MASE, and WQL\n"
    report += "- **MASE=1** represents the baseline seasonal naive forecast performance\n"
    report += "- **WQL** evaluates entire probabilistic forecast distributions\n"
    
    report += "\n---\n\n"
    report += "*Generated from PMDS Chronos Forecasting Benchmark*\n"
    
    return report


def create_html_report(output_path: Path):
    """Create HTML report."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chronos Forecasting Benchmark Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 10px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        section {
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        h3 {
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        h4 {
            color: #555;
            margin-top: 15px;
            margin-bottom: 10px;
        }
        .dataset-card, .model-card {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .dataset-card h3 {
            color: #667eea;
            margin-top: 0;
        }
        .model-card h3 {
            color: #764ba2;
            margin-top: 0;
        }
        .metric {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .metric-name {
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }
        .direction {
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .direction.lower {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .direction.higher {
            background-color: #fff3e0;
            color: #e65100;
        }
        ul, ol {
            margin-left: 20px;
            margin-bottom: 10px;
        }
        li {
            margin-bottom: 5px;
        }
        .strength {
            color: #2e7d32;
        }
        .weakness {
            color: #c62828;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #667eea;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f5f5f5;
        }
        .domain {
            display: inline-block;
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            margin: 5px 0;
        }
        .characteristics {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .hr-text {
            text-align: center;
            margin: 30px 0;
            color: #999;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Chronos Forecasting Benchmark</h1>
            <p>Comprehensive Report on Datasets, Metrics, and Models</p>
        </header>

        <!-- Datasets Section -->
        <section>
            <h2>📊 Datasets Overview</h2>
            <p>This benchmark evaluates models across diverse time series datasets spanning multiple domains and frequencies.</p>
"""
    
    for dataset_key, dataset_info in DATASET_DESCRIPTIONS.items():
        html += f"""
            <div class="dataset-card">
                <h3>{dataset_info['name']}</h3>
                <div class="domain">{dataset_info['domain']}</div>
                <p><strong>{dataset_info['description']}</strong></p>
                
                <h4>Key Characteristics:</h4>
                <ul class="characteristics">
"""
        for char in dataset_info['characteristics']:
            html += f"                    <li>{char}</li>\n"
        html += f"""                </ul>
                <p><strong>Use Case:</strong> {dataset_info['use_case']}</p>
            </div>
"""
    
    html += """        </section>

        <!-- Metrics Section -->
        <section>
            <h2>📈 Evaluation Metrics</h2>
            <p>Five complementary metrics are used to evaluate forecast accuracy and quality.</p>
"""
    
    for metric_key, metric_info in METRIC_DESCRIPTIONS.items():
        direction_class = "lower" if metric_info['lower_is_better'] else "higher"
        direction_text = "↓ Lower is Better" if metric_info['lower_is_better'] else "↑ Higher is Better"
        
        html += f"""
            <div class="metric">
                <div class="metric-header">
                    <span class="metric-name">{metric_info['name']}</span>
                    <span class="direction {direction_class}">{direction_text}</span>
                </div>
                <p><strong>Formula:</strong> <code>{metric_info['formula']}</code></p>
                <p><strong>Unit:</strong> {metric_info['unit']}</p>
                <p>{metric_info['description']}</p>
                <p><strong>📌 Interpretation:</strong> {metric_info['interpretation']}</p>
            </div>
"""
    
    html += """        </section>

        <!-- Models Section -->
        <section>
            <h2>🤖 Forecasting Models</h2>
            <p>Six different forecasting models spanning deep learning, baselines, and statistical methods.</p>
"""
    
    for model_key, model_info in MODEL_DESCRIPTIONS.items():
        html += f"""
            <div class="model-card">
                <h3>{model_info['name']}</h3>
                <p><strong>Type:</strong> {model_info['type']}</p>
                <p><strong>Parameters:</strong> ~{model_info['parameters']}</p>
                <p>{model_info['description']}</p>
                
                <h4>Features:</h4>
                <ul>
"""
        for feature in model_info['features']:
            html += f"                    <li>{feature}</li>\n"
        
        html += """                </ul>
                
                <h4>Strengths:</h4>
                <ul>
"""
        for strength in model_info['strengths']:
            html += f'                    <li class="strength">✅ {strength}</li>\n'
        
        html += """                </ul>
                
                <h4>Weaknesses:</h4>
                <ul>
"""
        for weakness in model_info['weaknesses']:
            html += f'                    <li class="weakness">❌ {weakness}</li>\n'
        
        html += """                </ul>
            </div>
"""
    
    html += """        </section>

        <!-- Summary Section -->
        <section>
            <h2>📋 Summary & Key Insights</h2>
            
            <h3>Model Categories</h3>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Model</th>
                    <th>Characteristics</th>
                </tr>
                <tr>
                    <td><strong>Deep Learning</strong></td>
                    <td>Chronos T5 Tiny</td>
                    <td>Pretrained transformer model with zero-shot capabilities</td>
                </tr>
                <tr>
                    <td><strong>Baseline</strong></td>
                    <td>Seasonal Naive</td>
                    <td>Simple repeating pattern baseline for benchmarking</td>
                </tr>
                <tr>
                    <td rowspan="4"><strong>Statistical</strong></td>
                    <td>AR(2)</td>
                    <td>Autoregressive model - memory of last 2 values</td>
                </tr>
                <tr>
                    <td>MA(2)</td>
                    <td>Moving average - models forecast errors</td>
                </tr>
                <tr>
                    <td>ARMA(2,2)</td>
                    <td>Combined AR and MA</td>
                </tr>
                <tr>
                    <td>ARIMA(2,1,2)</td>
                    <td>Integrated - handles trends via differencing</td>
                </tr>
                <tr>
                    <td><strong>Hybrid</strong></td>
                    <td>Prophet</td>
                    <td>Bayesian time series decomposition with trend & seasonality</td>
                </tr>
            </table>
            
            <h3>Key Insights</h3>
            <ul>
                <li><strong>Seasonal Naive</strong> provides essential baseline, particularly strong on seasonal data</li>
                <li><strong>Chronos T5</strong> leverages deep learning and pretraining for superior cross-dataset performance</li>
                <li><strong>Statistical Methods</strong> (ARIMA, Prophet) excel at interpretability and specific data patterns</li>
                <li><strong>Metric Hierarchy</strong>: MASE=1 represents seasonal naive baseline performance</li>
                <li><strong>Probabilistic vs Point</strong>: WQL evaluates entire forecast distributions; MAE/RMSE focus on point accuracy</li>
                <li><strong>Scale Invariance</strong>: SMAPE enables fair comparison across datasets with different scales</li>
            </ul>
        </section>

        <footer>
            <p><em>Generated from PMDS Chronos Forecasting Benchmark</em></p>
            <p>For detailed results, see the individual dataset plots and CSV summary files.</p>
        </footer>
    </div>
</body>
</html>
"""
    return html


def main():
    output_dir = Path("pmds/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown report
    md_report = create_markdown_report(output_dir)
    md_path = output_dir / "REPORT.md"
    with open(md_path, 'w') as f:
        f.write(md_report)
    print(f"✅ Saved: {md_path}")
    
    # Create HTML report
    html_report = create_html_report(output_dir)
    html_path = output_dir / "REPORT.html"
    with open(html_path, 'w') as f:
        f.write(html_report)
    print(f"✅ Saved: {html_path}")
    
    print(f"\nReports generated successfully!")
    print(f"  - Markdown: {md_path}")
    print(f"  - HTML: {html_path}")
    print(f"\nOpen {html_path} in a browser for an interactive view.")


if __name__ == '__main__':
    main()
