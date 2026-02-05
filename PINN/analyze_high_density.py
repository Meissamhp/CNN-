# ==========================================
# تحلیل عملکرد مدل‌ها در ناحیه‌ی پرترافیک (High-Density Region)
# ==========================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# افزودن مسیر فایل GRU اصلی
sys.path.append("../peredict")

from step7_train_gru_pinn_LWR import GRUModel  # مدل PINN-LWR
from step3_train_gru_torch import GRUModel as BaseGRU  # مدل GRU ساده

# خواندن داده‌ها
data = np.load("../GTFS_Project/napoli_train_test_multi.npz")
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"][:, None], dtype=torch.float32)

# ناحیه پرترافیک (تراکم بالا)
high_density_idx = y_test.squeeze() > y_test.mean()  # تراکم بالاتر از میانگین
X_hd = X_test[high_density_idx]
y_hd = y_test[high_density_idx]

# بارگذاری مدل‌ها
gru_model = BaseGRU()
gru_model.load_state_dict(torch.load("../peredict/gru_torch_model.pt"))
pinn_model = GRUModel()
pinn_model.load_state_dict(torch.load("pinn_gru_model_LWR.pt"))

gru_model.eval()
pinn_model.eval()

# پیش‌بینی
with torch.no_grad():
    y_gru_pred = gru_model(X_hd)
    y_pinn_pred, _ = pinn_model(X_hd)

# محاسبه خطاها
mse_gru = torch.mean((y_gru_pred - y_hd) ** 2).item()
mse_pinn = torch.mean((y_pinn_pred - y_hd) ** 2).item()

print(f"📊 GRU High-density MSE: {mse_gru:.4f}")
print(f"📊 PINN-LWR High-density MSE: {mse_pinn:.4f}")

# رسم نمودار مقایسه
plt.figure(figsize=(6, 6))
plt.scatter(y_hd, y_gru_pred, s=10, alpha=0.6, label=f"GRU (MSE={mse_gru:.3f})")
plt.scatter(y_hd, y_pinn_pred, s=10, alpha=0.6, label=f"PINN-LWR (MSE={mse_pinn:.3f})")
plt.plot([y_hd.min(), y_hd.max()], [y_hd.min(), y_hd.max()], "k--", lw=1)
plt.xlabel("True Traffic Density")
plt.ylabel("Predicted Density")
plt.title("High-Density Region Performance Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("high_density_comparison.png", dpi=300)
plt.show()
