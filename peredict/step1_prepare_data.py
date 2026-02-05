# ============================================
# مرحله ۱: آماده‌سازی داده‌های حمل‌ونقل ناپل
# ============================================

import pandas as pd
import numpy as np

# 1️⃣ خواندن فایل داده
df = pd.read_csv("napoli_c16_timeseries.csv")

# نمایش چند ردیف برای بررسی
print("🔹 چند ردیف اول داده:")
print(df.head())

# 2️⃣ ساخت ستون datetime از minute_bin
# فرض می‌کنیم داده‌ها مربوط به یک روز هستند و هر minute_bin یعنی چندمین بازه‌ی ۱۵ دقیقه‌ای در روز
# (یعنی minute_bin * 15 دقیقه)
df["datetime"] = pd.to_timedelta(df["minute_bin"] * 15, unit="m")

# حالا داده‌ها رو بر اساس زمان مرتب می‌کنیم
df = df.sort_values("datetime")

# 3️⃣ انتخاب فقط ستون تعداد وسایل نقلیه
flow_data = df["vehicle_count"].values

# 4️⃣ تابع برای ساخت دنباله‌های زمانی
def create_sequences(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(flow_data, window_size=5)

# 5️⃣ چاپ شکل داده‌ها برای اطمینان
print("🔹 X shape:", X.shape)
print("🔹 y shape:", y.shape)

# 6️⃣ ذخیره برای مراحل بعد
np.savez("napoli_sequences_c16.npz", X=X, y=y)
print("✅ فایل napoli_sequences_c16.npz ساخته شد!")
