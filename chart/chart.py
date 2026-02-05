# =======================================================
# Comparative Analysis of GRU, CNN, LSTM, and Physics-GRU
# =======================================================

import matplotlib.pyplot as plt
import numpy as np

# 🔹 Experimental results
models = ["GRU", "CNN", "LSTM", "Physics-GRU"]
mse = [1.4303, 1.2298, 0.9821, 1.2710]
latency = [0.4060, 0.1730, 0.9815, 0.1698]

# 🔹 Plot configuration
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(9, 5))
plt.title("Model Performance Comparison", fontsize=16, fontweight="bold")

# 🔸 Bars
bar1 = plt.bar(x - width/2, mse, width, label="MSE", color="#5DADE2", alpha=0.9)
bar2 = plt.bar(x + width/2, latency, width, label="Prediction Time (ms)", color="#F4D03F", alpha=0.9)

# 🔹 Labels and axes
plt.xticks(x, models, fontsize=12, fontweight="bold")
plt.ylabel("Value", fontsize=12)
plt.legend(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# 🔹 Add numerical labels above each bar
def add_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", 
                 ha="center", fontsize=10, fontweight="bold")

add_labels(bar1)
add_labels(bar2)

plt.tight_layout()
plt.savefig("final_model_comparison.png", dpi=300)
plt.show()
