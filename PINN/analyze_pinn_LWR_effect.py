# ==========================================
# تحلیل رفتار فیزیکی مدل PINN (LWR)
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
import torch

# بارگذاری داده
data = np.load("../GTFS_Project/napoli_train_test_multi.npz")
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"][:, None], dtype=torch.float32)

# بارگذاری مدل‌ها
import sys, os
sys.path.append(os.path.abspath("../peredict"))
sys.path.append(os.path.abspath("."))

from step3_train_gru_torch import GRUModel as BaseGRUModel
from step7_train_gru_pinn_LWR import GRUModel as PinnGRUModel

# مدل GRU پایه
gru_model = BaseGRUModel()
gru_model.load_state_dict(torch.load("../peredict/gru_torch_model.pt"))
gru_model.eval()

# مدل PINN (LWR)
pinn_model = PinnGRUModel()
pinn_model.load_state_dict(torch.load("pinn_gru_model_LWR.pt"))
pinn_model.eval()

# پیش‌بینی
with torch.no_grad():
    y_pred_gru = gru_model(X_test)
    rho_pred, v_pred = pinn_model(X_test)
    flow_pred = rho_pred * v_pred  # جریان در PINN (q = ρv)

# نمونه‌برداری از چند ایستگاه
plt.figure(figsize=(7,5))
plt.scatter(y_test[:300], y_pred_gru[:300], alpha=0.6, label="GRU", color="#E74C3C")
plt.scatter(y_test[:300], rho_pred[:300], alpha=0.6, label="PINN-LWR", color="#2E86C1")
# محاسبه‌ی مقدار بیشینه به صورت عددی (نه Tensor)
max_val = float(y_test.max())
plt.plot([0, max_val], [0, max_val], 'k--', lw=1)

plt.xlabel("True Density (ρ)")
plt.ylabel("Predicted Density (ρ̂)")
plt.title("Comparison of Physical Realism in Predictions")
plt.legend()
plt.grid(True)
plt.show()

# نمودار جریان–چگالی
plt.figure(figsize=(7,5))
plt.scatter(rho_pred[:300], flow_pred[:300], color="#27AE60", alpha=0.6)
plt.title("Fundamental Diagram Learned by PINN-LWR")
plt.xlabel("Density (ρ)")
plt.ylabel("Flow (q = ρv)")
plt.grid(True)
plt.show()
