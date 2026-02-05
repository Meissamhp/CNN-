# ==========================================
# مرحله ۶: مدل Physics-Informed GRU (نسخه ساده)
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
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# 3️⃣ تابع قید فیزیکی (smoothness loss)
def physics_loss(y_pred, X):
    """
    هدف: هموارسازی تغییرات خروجی مدل در طول زمان
    یعنی مدل نباید تغییرات ناگهانی بین گام‌ها داشته باشد
    """
    diff = y_pred[1:] - y_pred[:-1]
    return torch.mean(diff**2)

# 4️⃣ ایجاد مدل، معیارها و بهینه‌ساز
model = GRUModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5️⃣ آموزش مدل با دو نوع Loss
epochs = 200
lambda_phys = 0.01  # وزن قید فیزیکی

print("🚀 Starting Physics-Informed Training...\n")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train)
    data_loss = criterion(y_pred, y_train)

    phys_loss = physics_loss(y_pred, X_train)
    total_loss = data_loss + lambda_phys * phys_loss

    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"📉 Epoch {epoch}/{epochs}: DataLoss={data_loss.item():.4f} | PhysLoss={phys_loss.item():.4f}")

torch.save(model.state_dict(), "pinn_gru_model_simple.pt")
print("\n💾 Physics-Informed GRU model trained successfully!")

# 6️⃣ ارزیابی مدل روی داده تست
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    mse = criterion(y_pred, y_test).item()
    print(f"📊 MSE on test data: {mse:.4f}")
    # معیارهای ارزیابی تکمیلی
mae = torch.mean(torch.abs(y_pred - y_test)).item()
rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2)).item()
mape = (torch.mean(torch.abs((y_pred - y_test) / (y_test + 1e-8))) * 100).item()

print(f"📊 MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")


# 7️⃣ سرعت پیش‌بینی
start = time.time()
for _ in range(100):
    _ = model(X_test)
end = time.time()
print(f"⚡ Avg prediction time: {(end - start)/100*1000:.4f} ms")
