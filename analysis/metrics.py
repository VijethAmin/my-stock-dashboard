import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(actual, predicted) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    a, p = np.array(actual), np.array(predicted)
    mask = a != 0
    mape = np.abs((a[mask] - p[mask]) / a[mask]).mean() * 100 if mask.any() else float("nan")
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    dir_acc = np.mean((np.diff(a) > 0) == (np.diff(p) > 0)) * 100 if len(a) > 1 else 0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2, "Direction Accuracy (%)": dir_acc}
