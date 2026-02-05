# ==========================================
# 📊 Comparison of All Models (CNN, LSTM, GRU, PINN, PINN-LWR)
# ==========================================

import matplotlib.pyplot as plt
import numpy as np

# داده‌ها از خروجی واقعی تو:
models = ["CNN", "LSTM", "GRU", "PINN", "PINN-LWR"]

mse =  [0.8510, 0.8126, 0.8056, 0.8137, 0.8105]
mae =  [0.7350, 0.6753, 0.6859, 0.6886, 0.6892]
rmse = [0.9225, 0.9015, 0.8975, 0.9020, 0.9003]
mape = [43.87, 41.87, 42.44, 42.79, 42.53]
time = [1.7566, 8.0878, 6.7461, 6.8144, 7.0757]

# 🎨 تنظیمات نمودار
x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

# 📉 میله‌های MSE
bars1 = ax1.bar(x - width/2, mse, width, label="MSE", color="#2E86C1")

# ⚡ میله‌های زمان
bars2 = ax2.bar(x + width/2, time, width, label="Prediction Time (ms)", color="#F39C12")

# 📈 تنظیمات ظاهری
ax1.set_xlabel("Model", fontsize=12)
ax1.set_ylabel("MSE", color="#2E86C1", fontsize=12)
ax2.set_ylabel("Prediction Time (ms)", color="#F39C12", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_title("Performance Comparison of Models", fontsize=14, fontweight="bold")

# 📋 Legend
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# 📊 نمایش مقدار بالای هر ستون
def add_labels(bars, axis):
    for bar in bars:
        height = bar.get_height()
        axis.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}", 
                  ha='center', va='bottom', fontsize=9)

add_labels(bars1, ax1)
add_labels(bars2, ax2)

plt.tight_layout()
plt.show()
