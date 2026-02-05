# ==========================================
# مرحله ۷: Physics-Informed GRU (LWR-based)
# ==========================================

import torch
import torch.nn as nn
import numpy as np
import time

# 1️⃣ خواندن داده‌ها
data = np.load("../GTFS_Project/napoli_train_test_multi.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# تبدیل به Tensor برای PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train[:, None], dtype=torch.float32)
y_test  = torch.tensor(y_test[:, None], dtype=torch.float32)

seq_len = X_train.shape[1]
print(f"📏 Sequence length: {seq_len}")

# 2️⃣ تعریف مدل GRU
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_rho = nn.Linear(hidden_size, 1)  # خروجی چگالی
        self.fc_v   = nn.Linear(hidden_size, 1)  # خروجی سرعت (برای قید LWR)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out_last = out[:, -1, :]
        rho = self.fc_rho(out_last)
        v = torch.sigmoid(self.fc_v(out_last)) * 30.0  # محدود به 0–30 m/s
        return rho, v

# 3️⃣ تابع فیزیکی بر اساس معادله LWR
def physics_loss_lwr(rho_pred, v_pred, X):
    """
    تقریب ساده معادله‌ی LWR:
    dρ/dt + d(ρv)/dx ≈ 0
    چون داده‌ ما ایستگاه-زمان است، از تفاوت زمانی بین نمونه‌ها استفاده می‌کنیم.
    """
    drho_dt = rho_pred[1:] - rho_pred[:-1]
    flow = rho_pred * v_pred
    dflow_dx = flow[1:] - flow[:-1]
    return torch.mean((drho_dt + dflow_dx)**2)

# 4️⃣ تنظیم مدل، معیار و optimizer
model = GRUModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lambda_phys = 0.01  # وزن قید فیزیکی (λ)

# 5️⃣ آموزش مدل
epochs = 200
print("🚦 Physics-Informed GRU (LWR) Training...\n")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    rho_pred, v_pred = model(X_train)
    data_loss = criterion(rho_pred, y_train)
    phys_loss = physics_loss_lwr(rho_pred, v_pred, X_train)

    total_loss = data_loss + lambda_phys * phys_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"📉 Epoch {epoch}/{epochs}: DataLoss={data_loss.item():.4f} | PhysLoss={phys_loss.item():.4f}")

torch.save(model.state_dict(), "pinn_gru_model_LWR.pt")
print("\n💾 مدل Physics-Informed GRU (LWR) ذخیره شد.")

# 6️⃣ ارزیابی مدل روی داده تست
model.eval()
with torch.no_grad():
    rho_pred_test, v_pred_test = model(X_test)
    mse = criterion(rho_pred_test, y_test).item()

    mae = torch.mean(torch.abs(rho_pred_test - y_test)).item()
    rmse = torch.sqrt(torch.mean((rho_pred_test - y_test) ** 2)).item()
    mape = (torch.mean(torch.abs((rho_pred_test - y_test) / (y_test + 1e-8))) * 100).item()

print(f"📊 MSE روی داده‌ی تست: {mse:.4f}")
print(f"📊 MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")

# 7️⃣ محاسبه سرعت پیش‌بینی
start = time.time()
for _ in range(100):
    _ = model(X_test)
end = time.time()
print(f"⚡ میانگین زمان پیش‌بینی: {(end - start)/100*1000:.4f} ms")
