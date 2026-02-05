import matplotlib.pyplot as plt
import numpy as np

# 🔹 داده‌های نهایی مدل‌ها
models = ["CNN", "LSTM", "GRU", "PINN-Simple", "PINN-LWR"]
mse = [0.8839, 0.8110, 0.8133, 0.8105, 0.8154]
time_ms = [3.21, 9.78, 6.81, 6.88, 6.89]

# تنظیمات شکل
fig, ax1 = plt.subplots(figsize=(8, 5))

ax2 = ax1.twinx()
ax1.bar(models, mse, color='skyblue', label='MSE (lower is better)', width=0.4)
ax2.plot(models, time_ms, color='red', marker='o', label='Prediction Time (ms)')

ax1.set_xlabel("Model Type", fontsize=12)
ax1.set_ylabel("MSE", color='blue', fontsize=12)
ax2.set_ylabel("Prediction Time (ms)", color='red', fontsize=12)
ax1.set_title("Model Comparison: MSE vs. Prediction Speed", fontsize=14, fontweight='bold')

# اضافه کردن legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=300)
plt.show()
