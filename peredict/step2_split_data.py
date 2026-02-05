# ===============================================
# مرحله ۲: تقسیم داده‌ها به آموزش و آزمون (Train/Test)
# ===============================================

import numpy as np
from sklearn.model_selection import train_test_split

# 1️⃣ خواندن فایل آماده‌شده از مرحله‌ی ۱
data = np.load("napoli_sequences_c16.npz")
X = data["X"]
y = data["y"]

print("🔹 X shape:", X.shape)
print("🔹 y shape:", y.shape)

# 2️⃣ تقسیم داده‌ها به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("✅ Train shape:", X_train.shape)
print("✅ Test shape:", X_test.shape)

# 3️⃣ ذخیره‌ی خروجی برای مراحل بعد
np.savez("napoli_train_test.npz", 
         X_train=X_train, X_test=X_test, 
         y_train=y_train, y_test=y_test)

print("💾 فایل napoli_train_test.npz ساخته شد!")
