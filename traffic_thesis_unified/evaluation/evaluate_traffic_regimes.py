import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.architectures import TrafficModel
from utils.data_loader_unified import get_unified_data
from utils.metrics import calculate_metrics


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def load_model(model_name, checkpoint_path, num_features, seq_len):
    if model_name in ["LSTM", "PINN", "PINN-LWR"]:
        base_type = "LSTM"
    elif model_name == "GRU":
        base_type = "GRU"
    elif model_name == "CNN":
        base_type = "CNN"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = TrafficModel(
        model_type=base_type,
        input_dim=num_features,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2,
        seq_len=seq_len
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        pred = model(X.to(device)).detach().cpu().numpy().reshape(-1)
    return pred


def scenario_to_regime(series):
    s = series.astype(str).str.lower()

    heavy_keywords = ["heavy", "congested", "rush", "peak"]
    stable_keywords = ["stable", "light", "free", "normal", "smooth"]

    regime = []
    for val in s:
        if any(k in val for k in heavy_keywords):
            regime.append("heavy")
        elif any(k in val for k in stable_keywords):
            regime.append("stable")
        else:
            regime.append("unknown")
    return pd.Series(regime)


def save_regime_chart(result_df):
    plot_df = result_df[result_df["Regime"].isin(["heavy", "stable"])].copy()

    if plot_df.empty:
        print("No heavy/stable rows found for chart.")
        return

    pivot_df = plot_df.pivot(index="Model", columns="Regime", values="RMSE").fillna(np.nan)
    pivot_df = pivot_df.reindex(["LSTM", "GRU", "CNN", "PINN", "PINN-LWR"]).dropna(how="all")

    ax = pivot_df.plot(kind="bar", figsize=(10, 6), width=0.8)
    ax.set_title("Traffic Regime Comparison (RMSE)")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.legend(title="Regime")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/traffic_regime_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, meta, test_meta = get_unified_data(
        seq_len=12,
        return_test_meta=True
    )

    y_true = y_test.detach().cpu().numpy().reshape(-1)
    test_meta = test_meta.copy()

    if "scenario" not in test_meta.columns:
        raise ValueError("Column 'scenario' not found in test metadata.")

    test_meta["regime"] = scenario_to_regime(test_meta["scenario"])

    models = {
        "LSTM": "checkpoints/lstm_model.pth",
        "GRU": "checkpoints/gru_model.pth",
        "CNN": "checkpoints/cnn_model.pth",
        "PINN": "checkpoints/pinn_model.pth",
        "PINN-LWR": "checkpoints/pinn_lwr_model.pth",
    }

    all_rows = []

    for model_name, ckpt in models.items():
        if not os.path.exists(ckpt):
            print(f"Missing checkpoint for {model_name}: {ckpt}")
            continue

        model = load_model(model_name, ckpt, meta["num_features"], meta["seq_len"])
        y_pred = predict(model, X_test)

        for regime_name in ["heavy", "stable", "unknown"]:
            idx = test_meta["regime"] == regime_name

            if idx.sum() == 0:
                continue

            metrics = calculate_metrics(y_true[idx.values], y_pred[idx.values])

            all_rows.append({
                "Model": model_name,
                "Regime": regime_name,
                "Count": int(idx.sum()),
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "STD_ERROR": metrics["STD_ERROR"]
            })

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv("results/traffic_regime_comparison.csv", index=False)
    save_regime_chart(result_df)

    print("\nSaved: results/traffic_regime_comparison.csv")
    print("Saved: results/traffic_regime_comparison.png")
    print(result_df.sort_values(["Regime", "MSE"]))


if __name__ == "__main__":
    main()