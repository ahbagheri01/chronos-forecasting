# Final Clean Result Plots

This directory mirrors the structure of `pmds/results/plots` for the final clean benchmark:

```text
plots/
  <dataset>/
    mae.png
    rmse.png
    smape.png
    mase.png
    wql.png
    wql_macro.png
    all_metrics_normalized.png
    all_metrics_dot_comparison.png
    forecast_vs_actual/
      <model>.png
```

Rain occurrence error and positive-value MAE plots are also present for datasets where those
diagnostics were populated.

## Source-run selection

The consolidated CSVs intentionally retain overlapping runs. To avoid averaging a rerun with
its predecessor in a single plot, this combined 12-model view uses:

- `keyless` rows for the six Chronos/Monash datasets;
- `external_keyless` rows for the six external and official datasets;
- `heavier_chronos` rows for the four larger Chronos checkpoints on all 12 datasets.

The source CSVs in the parent directory were not modified or deduplicated. This selection only
controls which source rows contribute to this plot collection.

The metric and trajectory images were produced with `pmds/plot_results.py`. Repeated stochastic
forecasts are aggregated for display by the plotting code, and the trajectory images use the
latest rolling origin represented by origin index `0`.
