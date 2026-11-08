# Rainfall Prediction

Small learning project. The notebook `rainfall_prediction.ipynb` trains a random forest classifier.

Data: Daily weather data (Rainfall.csv), predicting whether it rains from 7 weather features. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`src/benchmark.py` compares HistGradientBoosting, random forest and logistic regression with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (1095 samples, 7 features), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python src/benchmark.py
```

## Results (5-fold CV, mean)

| Model | roc_auc | accuracy | f1 |
|---|---|---|---|
| HistGradientBoosting Forecaster | 0.9802 | 0.9352 | 0.919 |
| Grid-Tuned Random Forest (200 Trees) | 0.9791 | 0.9315 | 0.9139 |
| Regularized Logistic Regression | 0.9535 | 0.8813 | 0.8516 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `rainfall_prediction.ipynb`: notebook
- `src/benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
