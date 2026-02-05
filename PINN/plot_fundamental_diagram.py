import torch
import numpy as np
import matplotlib.pyplot as plt

# === تعریف مدل GRU ساده ===
class GRUModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# === تعریف مدل PINN-LWR ===
class GRUModel_LWR(torch.nn.Module):
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

# === بارگذاری مدل‌ها ===
gru_model = GRUModel()
pinn_model = GRUModel_LWR()

gru_model.load_state_dict(torch.load("../peredict/gru_torch_model.pt"))
pinn_model.load_state_dict(torch.load("pinn_gru_model_LWR.pt"))

gru_model.eval()
pinn_model.eval()

# === داده تست ===
data = np.load("../GTFS_Project/napoli_train_test_multi.npz")
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"][:, None], dtype=torch.float32)

# === پیش‌بینی ===
with torch.no_grad():
    y_pred_gru = gru_model(X_test)
    rho_pred, v_pred = pinn_model(X_test)
    y_pred_pinn = rho_pred

# === تبدیل به NumPy برای رسم ===
y_test_np = y_test.numpy().flatten()
y_pred_gru_np = y_pred_gru.numpy().flatten()
y_pred_pinn_np = y_pred_pinn.numpy().flatten()

# === رسم نمودار Fundamental Diagram ===
plt.figure(figsize=(6,5))
plt.scatter(y_test_np, y_pred_gru_np, color='blue', alpha=0.5, label='GRU')
plt.scatter(y_test_np, y_pred_pinn_np, color='red', alpha=0.5, label='PINN-LWR')
plt.plot([0, max(y_test_np)], [0, max(y_test_np)], 'k--', lw=1)
plt.xlabel("True Flow")
plt.ylabel("Predicted Flow")
plt.title("Fundamental Diagram Comparison: GRU vs PINN-LWR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fundamental_diagram_comparison.png")
print("✅ Diagram saved as fundamental_diagram_comparison.png")
