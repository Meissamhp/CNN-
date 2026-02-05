# ==========================================
# 📊 Full Metrics Comparison: MSE, MAE, RMSE, MAPE for all models
# ==========================================

import matplotlib.pyplot as plt
import numpy as np

# مدل‌ها
models = ["CNN", "LSTM", "GRU", "PINN", "PINN-LWR"]

# معیارها (بر اساس خروجی واقعی تو)
mse  = [0.8510, 0.8126, 0.8056, 0.8137, 0.8105]
mae  = [0.7350, 0.6753, 0.6859, 0.6886, 0.6892]
rmse = [0.9225, 0.9015, 0.8975, 0.9020, 0.9003]
mape = [43.87, 41.87, 42.44, 42.79, 42.53]
time = [1.7566, 8.0878, 6.7461, 6.8144, 7.0757]

metrics = [mse, mae, rmse, mape]
metric_names = ["MSE", "MAE", "RMSE", "MAPE"]

x = np.arange(len(models))
width = 0.18

plt.figure(figsize=(10,6))

# 🎨 رسم میله‌ها
for i, (metric, name, color) in enumerate(zip(metrics, metric_names, 
    ["#2E86C1", "#28B463", "#8E44AD", "#E67E22"])):
    plt.bar(x + i*width - 1.5*width, metric, width, label=name, color=color)

plt.xticks(x, models, fontsize=11)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Metric Value", fontsize=12)
plt.title("Full Performance Comparison of Models", fontsize=14, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()
