#!/usr/bin/env python
"""Generate plots from compare_detailed.csv results."""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


def load_config(config_path: Path) -> dict:
    """Load config to get metrics list."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def plot_results(csv_path: Path, output_base: Path, config_path: Path = None):
    """
    Read CSV and generate plots for each dataset.
    
    Args:
        csv_path: Path to compare_detailed.csv
        output_base: Base output directory for plots
        config_path: Path to config.json to get metrics list
    """
    # Load config to get metrics
    if config_path is None:
        config_path = Path("pmds/config.json")
    
    config = load_config(config_path)
    metrics = config.get("evaluation", {}).get("metrics", ["mae", "rmse", "smape", "mase", "wql"])
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Remove rows with errors
    df = df[df['error_type'].isna() | (df['error_type'] == '')]
    
    # Group by dataset
    datasets = df['dataset'].unique()
    
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(datasets)} datasets")
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        dataset_df = df[df['dataset'] == dataset]
        
        # Create dataset-specific output directory
        dataset_dir = output_base / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Aggregate metrics by model (take mean across items)
        model_stats = dataset_df.groupby('model')[metrics].mean()
        
        print(f"  Models: {', '.join(model_stats.index.tolist())}")
        print(f"  Items: {dataset_df['item_id'].nunique()}")
        
        # Create a plot for each metric
        for metric in metrics:
            if metric not in model_stats.columns:
                print(f"  Warning: Metric '{metric}' not found in data")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get data for this metric
            data = model_stats[metric].sort_values()
            
            # Create bar plot
            data.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', alpha=0.8)
            
            # Customize plot
            ax.set_title(f'{dataset.replace("_", " ").title()}\n{metric.upper()} by Model', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(data.values):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = dataset_dir / f'{metric}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
            plt.close()
        
        # Create a comparison plot with all metrics normalized (bar chart)
        if len(metrics) > 1:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Normalize metrics for comparison (0-1 scale per metric)
            normalized_stats = model_stats.copy()
            scaling_factors = {}
            for metric in metrics:
                if metric in normalized_stats.columns:
                    min_val = normalized_stats[metric].min()
                    max_val = normalized_stats[metric].max()
                    scaling_factors[metric] = {'min': min_val, 'max': max_val, 'range': max_val - min_val}
                    if max_val > min_val:
                        normalized_stats[metric] = (normalized_stats[metric] - min_val) / (max_val - min_val)
            
            # Plot grouped bar chart
            normalized_stats.plot(kind='bar', ax=ax, width=0.8)
            
            ax.set_title(f'{dataset.replace("_", " ").title()}\nNormalized Metrics Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax.set_ylabel('Normalized Value (0-1)', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            
            # Save comparison plot
            plot_path = dataset_dir / 'all_metrics_normalized.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
            plt.close()
        
        # Create dot plot for all metrics with scaling factors annotated
        if len(metrics) > 1:
            fig, ax = plt.subplots(figsize=(16, 9))
            
            # Prepare data for scatter plot
            models = list(model_stats.index)
            
            # Use distinct color palette
            colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            color_map = {metric: colors_palette[i % len(colors_palette)] for i, metric in enumerate(metrics)}
            
            # Use different markers for metrics
            markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
            marker_map = {metric: markers[i % len(markers)] for i, metric in enumerate(metrics)}
            
            # Normalize metrics and collect data
            normalized_data = []
            scaling_factors = {}
            
            for metric in metrics:
                if metric not in model_stats.columns:
                    continue
                    
                metric_values = model_stats[metric].values
                min_val = metric_values.min()
                max_val = metric_values.max()
                scaling_range = max_val - min_val
                scaling_factors[metric] = {'min': min_val, 'max': max_val, 'range': scaling_range}
                
                if scaling_range > 0:
                    normalized_values = (metric_values - min_val) / scaling_range
                else:
                    normalized_values = metric_values
                
                for i, (model, norm_val) in enumerate(zip(models, normalized_values)):
                    normalized_data.append({
                        'model_idx': i,
                        'model': model,
                        'metric': metric,
                        'normalized_value': norm_val,
                        'min': min_val,
                        'max': max_val,
                        'range': scaling_range
                    })
            
            # Plot dots with offset for each metric to avoid overlap
            offset_step = 0.08
            metric_offsets = {metric: (i - len(metrics)/2) * offset_step for i, metric in enumerate(metrics)}
            
            for metric in metrics:
                metric_data = [d for d in normalized_data if d['metric'] == metric]
                model_indices = [d['model_idx'] + metric_offsets[metric] for d in metric_data]
                norm_values = [d['normalized_value'] for d in metric_data]
                
                ax.scatter(model_indices, norm_values, s=500, 
                          label=metric, color=color_map[metric], 
                          marker=marker_map[metric],
                          alpha=0.8, edgecolors='black', linewidth=2, zorder=3)
            
            # Customize plot
            ax.set_title(f'{dataset.replace("_", " ").title()}\nAll Metrics Comparison (Normalized with Scaling Factors)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Model', fontsize=13, fontweight='bold')
            ax.set_ylabel('Normalized Value (0-1)', fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
            ax.set_ylim(-0.15, 1.2)
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.7)
            ax.legend(title='Metrics', loc='upper left', fontsize=11, title_fontsize=12, 
                     framealpha=0.95, edgecolor='black')
            
            # Add scaling factor legend text
            legend_text = "Scaling Factors [min, max]:\n" + "─" * 35 + "\n"
            for metric in metrics:
                if metric in scaling_factors:
                    sf = scaling_factors[metric]
                    legend_text += f"{metric.upper():8s}: [{sf['min']:.3e}, {sf['max']:.3e}]\n"
            
            ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='black', linewidth=1),
                   family='monospace')
            
            plt.tight_layout()
            
            # Save dot plot
            plot_path = dataset_dir / 'all_metrics_dot_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
            plt.close()
    
    print(f"\nAll plots saved to: {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from compare_detailed.csv")
    parser.add_argument(
        '--csv',
        type=Path,
        default=Path('pmds/results/compare_detailed.csv'),
        help='Path to compare_detailed.csv'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('pmds/results/plots'),
        help='Output directory for plots'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('pmds/config.json'),
        help='Path to config.json'
    )
    
    args = parser.parse_args()
    
    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    plot_results(args.csv, args.output, args.config)


if __name__ == '__main__':
    main()
