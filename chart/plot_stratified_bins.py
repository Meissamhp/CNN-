import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === تنظیم مسیر اصلی ===
base_dir = "/Users/meisam/Desktop/maghale/chart/"
models = ["CNN", "LSTM", "GRU", "PINN", "PINN_LWR"]

# === توابع ارزیابی ===
def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred) ** 2))
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

# === تقسیم بن‌ها ===
def stratify_density(df):
    bins = [0, 0.33, 0.66, 1.0]
    labels = ["Low", "Medium", "High"]
    df["density_bin"] = pd.cut(df["density"], bins=bins, labels=labels, include_lowest=True)
    return df

results = []

# === پردازش هر مدل ===
for model in models:
    path = os.path.join(base_dir, f"results_{model}.csv")
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        continue

    df = pd.read_csv(path)
    df = stratify_density(df)

    for bin_name in ["Low", "Medium", "High"]:
        subset = df[df["density_bin"] == bin_name]
        if subset.empty:
            continue
        y_true = subset["true_value"].values
        y_pred = subset["pred_value"].values

        results.append({
            "Model": model,
            "Bin": bin_name,
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred)
        })

# === ساخت جدول خروجی ===
df_results = pd.DataFrame(results)
csv_path = os.path.join(base_dir, "stratified_bins_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"✅ Saved results to {csv_path}")

# === رسم نمودار ===
metrics = ["MAE", "RMSE", "MAPE"]
for metric in metrics:
    plt.figure(figsize=(8,5))
    pivot = df_results.pivot(index="Bin", columns="Model", values=metric)
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title(f"Stratified {metric} by Traffic Density")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"stratified_{metric}.png"))
    plt.close()

print("📊 All stratified metric plots saved successfully!")
