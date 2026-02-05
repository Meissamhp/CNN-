# ==========================================
# Visualizing LWR Physics Effect in PINN-GRU
# ==========================================

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.append(os.path.abspath("../PINN"))

# ---- GRU معمولی (یک خروجی) ----
class SimpleGRU(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out_last = out[:, -1, :]
        return self.fc(out_last)

# ---- GRU PINN با قید فیزیکی (دو خروجی) ----
class GRUModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_rho = torch.nn.Linear(hidden_size, 1)
        self.fc_v = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out_last = out[:, -1, :]
        rho = self.fc_rho(out_last)
        v = torch.sigmoid(self.fc_v(out_last)) * 30.0
        return rho, v

# ---- داده‌ها ----
data = np.load("../GTFS_Project/napoli_train_test_multi.npz")
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"][:, None], dtype=torch.float32)

# ---- مدل‌ها ----
gru_model = SimpleGRU()
gru_model.load_state_dict(torch.load("../peredict/gru_torch_model.pt"))
gru_model.eval()

pinn_model = GRUModel()
pinn_model.load_state_dict(torch.load("../PINN/pinn_gru_model_LWR.pt"))
pinn_model.eval()

# ---- پیش‌بینی ----
with torch.no_grad():
    rho_gru = gru_model(X_test)
    rho_pinn, v_pinn = pinn_model(X_test)

# ---- جریان‌ها ----
v_gru_approx = torch.ones_like(rho_gru) * 15  # سرعت ثابت فرضی برای مدل GRU ساده
flow_gru = (rho_gru * v_gru_approx).numpy().flatten()
flow_pinn = (rho_pinn * v_pinn).numpy().flatten()
rho_true = y_test.numpy().flatten()

# ---- رسم نمودار ----
plt.figure(figsize=(7, 5))
plt.scatter(rho_true, flow_gru, color="gray", alpha=0.5, label="GRU (No Physics)")
plt.scatter(rho_true, flow_pinn, color="blue", alpha=0.6, label="PINN-GRU (LWR)")
plt.xlabel("Traffic Density ρ (veh/km)")
plt.ylabel("Traffic Flow q = ρv")
plt.title("Effect of Physics-Informed Constraint (LWR Equation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("physics_effect.png", dpi=300)
plt.show()
