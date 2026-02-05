# ============================================================
# تحلیل ناحیه پرترافیک برای بررسی برتری PINN-LWR نسبت به GRU
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from step7_train_gru_pinn_LWR_fixed import GRUModel  # همان مدلی که استفاده کردی

# ------------------------------
# ۱) بارگذاری داده‌ها
# ------------------------------
data = np.load("../GTFS_Project/napoli_train_test_multi.npz")
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"][:, None], dtype=torch.float32)

# ------------------------------
# ۲) بارگذاری مدل‌های ذخیره‌شده
# ------------------------------
gru_model = GRUModel()
gru_model.load_state_dict(torch.load("../peredict/gru_torch_model.pt"))
gru_model.eval()

pinn_model = GRUModel()
pinn_model.load_state_dict(torch.load("pinn_gru_model_LWR.pt"))
pinn_model.eval()

# ------------------------------
# ۳) پیش‌بینی دو مدل
# ------------------------------
with torch.no_grad():
    rho_gru, v_gru = gru_model(X_test)
    rho_pinn, v_pinn = pinn_model(X_test)

y_true = y_test.numpy().flatten()
y_pred_gru = rho_gru.numpy().flatten()
y_pred_pinn = rho_pinn.numpy().flatten()

# ------------------------------
# ۴) تحلیل ناحیه پرترافیک (High-Density)
# ------------------------------
threshold = np.percentile(y_true, 80)  # بالاترین ۲۰٪ داده‌ها
mask = y_true > threshold

gru_high_mse = np.mean((y_true[mask] - y_pred_gru[mask])**2)
pinn_high_mse = np.mean((y_true[mask] - y_pred_pinn[mask])**2)

print("🚦 High-Density Analysis (Top 20% Traffic Flow)")
print(f"GRU  MSE (High-Density):  {gru_high_mse:.4f}")
print(f"PINN MSE (High-Density):  {pinn_high_mse:.4f}")

# ------------------------------
# ۵) رسم نمودار مقایسه‌ای
# ------------------------------
plt.figure(figsize=(7,6))
plt.scatter(y_true[mask], y_pred_gru[mask], color="orange", alpha=0.5, label=f"GRU (MSE={gru_high_mse:.3f})")
plt.scatter(y_true[mask], y_pred_pinn[mask], color="green", alpha=0.5, label=f"PINN-LWR (MSE={pinn_high_mse:.3f})")
plt.plot([0, max(y_true)], [0, max(y_true)], 'k--', lw=1)
plt.xlabel("True Traffic Flow", fontsize=12)
plt.ylabel("Predicted Traffic Flow", fontsize=12)
plt.title("High-Density Scenario Comparison", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("high_density_comparison.png", dpi=300)
plt.show()
