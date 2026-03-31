Show the latest model comparison results from the outputs/ directory.

Steps to follow:
1. Read outputs/comparison_results.csv (summary — one row per model)
2. Read outputs/per_series_metrics.csv if it exists
3. Display the summary table with these columns highlighted:
   model_name | wmape_mean | r2_mean | bias_mean | best_series_count | n_evaluations
4. Identify the best model by lowest wmape_mean and state it clearly
5. If per_series_metrics.csv exists:
   - Show the top 5 SKU×state combos where models disagree most (highest WMAPE variance across models)
   - Show the bottom 5 worst-performing series (highest wmape across all models)
6. List the forecast output files in outputs/forecasts/ and their row counts
7. Report the date the files were last modified so the user knows how fresh the results are

If outputs/ is empty or comparison_results.csv doesn't exist, say so and suggest running /run first.
