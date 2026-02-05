# ==========================================
# مرحله ۴: ساخت و آموزش مدل CNN 1D در PyTorch (نسخه چندمسیره)
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# 1️⃣ خواندن داده‌های مرحله قبل
data = np.load("../GTFS_Project/napoli_train_test_multi.npz")

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# تبدیل به Tensor برای PyTorch
X_train = torch.tensor(X_train[:, :, None], dtype=torch.float32)  # (batch, seq_len, 1)
X_test  = torch.tensor(X_test[:, :, None], dtype=torch.float32)
y_train = torch.tensor(y_train[:, None], dtype=torch.float32)
y_test  = torch.tensor(y_test[:, None], dtype=torch.float32)

# گرفتن طول دنباله از داده
seq_len = X_train.shape[1]
print(f"📏 Sequence length: {seq_len}")

# 2️⃣ تعریف مدل CNN به صورت پویا
class CNN1DModel(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # محاسبه خودکار اندازه‌ی خروجی پس از کانولوشن‌ها و pooling
        with torch.no_grad():
            x = torch.zeros(1, 1, seq_len)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            flattened_size = x.numel()
        
        self.fc = nn.Linear(flattened_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channel, seq_len)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

# 3️⃣ ایجاد مدل، معیار خطا و بهینه‌ساز
model = CNN1DModel(seq_len=seq_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4️⃣ آموزش مدل
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"📉 Epoch {epoch}/{epochs}: Loss = {loss.item():.4f}")

torch.save(model.state_dict(), "cnn_torch_model_multi.pt")
print("💾 مدل CNN ذخیره شد.")

# 5️⃣ ارزیابی روی داده تست
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    mse = criterion(y_pred, y_test).item()
    print(f"📊 MSE روی داده‌ی تست: {mse:.4f}")
    # معیارهای ارزیابی تکمیلی
mae = torch.mean(torch.abs(y_pred - y_test)).item()
rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2)).item()
mape = (torch.mean(torch.abs((y_pred - y_test) / (y_test + 1e-8))) * 100).item()

print(f"📊 MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")


# 6️⃣ محاسبه سرعت پیش‌بینی
start = time.time()
for _ in range(100):
    _ = model(X_test)
end = time.time()
print(f"⚡ میانگین زمان پیش‌بینی: {(end - start)/100*1000:.4f} ms")
