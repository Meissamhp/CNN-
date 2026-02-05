# ==========================================
# مرحله ۳: آموزش GRU پایه با داده‌های چندمسیره
# ==========================================

import torch
import torch.nn as nn
import numpy as np
import time

# 1️⃣ خواندن داده‌های جدید
data = np.load("../GTFS_Project/napoli_train_test_multi.npz")

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# تبدیل به Tensor
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
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out_last = out[:, -1, :]
        return self.fc(out_last)

# 3️⃣ تنظیمات آموزش
model = GRUModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4️⃣ آموزش مدل
epochs = 200
print("🚀 Training GRU...\n")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"📉 Epoch {epoch}/{epochs}: Loss = {loss.item():.4f}")

torch.save(model.state_dict(), "gru_torch_model.pt")
print("💾 مدل GRU ذخیره شد.")

# 5️⃣ ارزیابی
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    mse = criterion(y_pred, y_test).item()
    mae = torch.mean(torch.abs(y_pred - y_test)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2)).item()
    mape = (torch.mean(torch.abs((y_pred - y_test) / (y_test + 1e-8))) * 100).item()

print(f"📊 MSE روی داده‌ی تست: {mse:.4f}")
print(f"📊 MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")

# 6️⃣ زمان پیش‌بینی
start = time.time()
for _ in range(100):
    _ = model(X_test)
end = time.time()
print(f"⚡ میانگین زمان پیش‌بینی: {(end - start)/100*1000:.4f} ms")
