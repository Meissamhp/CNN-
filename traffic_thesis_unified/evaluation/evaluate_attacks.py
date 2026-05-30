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


NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]
SEED = 42


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


def add_gaussian_noise(X, noise_std, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    noise = torch.randn_like(X) * noise_std
    return X + noise


def save_rmse_chart(result_df):
    pivot_df = result_df.pivot(index="Noise_STD", columns="Model", values="RMSE")
    ordered_cols = [c for c in ["LSTM", "GRU", "CNN", "PINN", "PINN-LWR"] if c in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]

    ax = pivot_df.plot(marker="o", figsize=(10, 6))
    ax.set_title("Robustness Under Gaussian Noise (RMSE)")
    ax.set_xlabel("Noise STD")
    ax.set_ylabel("RMSE")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/attack_robustness_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_degradation_chart(result_df):
    pivot_df = result_df.pivot(index="Noise_STD", columns="Model", values="RMSE_Degradation_Pct")
    ordered_cols = [c for c in ["LSTM", "GRU", "CNN", "PINN", "PINN-LWR"] if c in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]

    ax = pivot_df.plot(marker="o", figsize=(10, 6))
    ax.set_title("Relative RMSE Degradation Under Gaussian Noise")
    ax.set_xlabel("Noise STD")
    ax.set_ylabel("RMSE Degradation (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/attack_robustness_degradation.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, meta = get_unified_data(seq_len=12)
    y_true = y_test.detach().cpu().numpy().reshape(-1)

    models = {
        "LSTM": "checkpoints/lstm_model.pth",
        "GRU": "checkpoints/gru_model.pth",
        "CNN": "checkpoints/cnn_model.pth",
        "PINN": "checkpoints/pinn_model.pth",
        "PINN-LWR": "checkpoints/pinn_lwr_model.pth",
    }

    all_rows = []
    clean_rmse = {}

    for model_name, ckpt in models.items():
        if not os.path.exists(ckpt):
            print(f"Missing checkpoint for {model_name}: {ckpt}")
            continue

        model = load_model(model_name, ckpt, meta["num_features"], meta["seq_len"])

        for noise_std in NOISE_LEVELS:
            X_attack = add_gaussian_noise(X_test.clone(), noise_std, seed=SEED)
            y_pred = predict(model, X_attack)
            metrics = calculate_metrics(y_true, y_pred)

            if noise_std == 0.0:
                clean_rmse[model_name] = metrics["RMSE"]

            base_rmse = clean_rmse[model_name]
            degradation_pct = ((metrics["RMSE"] - base_rmse) / base_rmse) * 100.0 if base_rmse != 0 else 0.0

            all_rows.append({
                "Model": model_name,
                "Attack_Type": "Gaussian_Noise",
                "Noise_STD": noise_std,
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "STD_ERROR": metrics["STD_ERROR"],
                "Clean_RMSE": base_rmse,
                "RMSE_Degradation_Pct": degradation_pct
            })

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv("results/attack_robustness_comparison.csv", index=False)

    save_rmse_chart(result_df)
    save_degradation_chart(result_df)

    print("\nSaved: results/attack_robustness_comparison.csv")
    print("Saved: results/attack_robustness_rmse.png")
    print("Saved: results/attack_robustness_degradation.png")
    print(result_df.sort_values(["Noise_STD", "MSE"]))


if __name__ == "__main__":
    main()