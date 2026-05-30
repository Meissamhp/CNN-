from utils.data_loader_unified import get_unified_data

X_train, y_train, X_val, y_val, X_test, y_test, meta, test_meta = get_unified_data(
    seq_len=12,
    return_test_meta=True
)

print("Train samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Test samples:", X_test.shape[0])
print("Total samples:", X_train.shape[0] + X_val.shape[0] + X_test.shape[0])

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

print("Meta:", meta)
print("Test meta shape:", test_meta.shape)
print(test_meta.head())
from utils.data_loader_unified import get_unified_data

X_train, y_train, X_val, y_val, X_test, y_test, meta, test_meta = get_unified_data(
    seq_len=12,
    return_test_meta=True
)

print("Unique route_short_name in test:")
print(sorted(test_meta["route_short_name"].astype(str).unique().tolist()))

print("Number of unique routes in test:")
print(test_meta["route_short_name"].nunique())

print("\nRoute counts in test:")
print(test_meta["route_short_name"].value_counts().sort_index())