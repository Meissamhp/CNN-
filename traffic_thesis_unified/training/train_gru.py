import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

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


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        loss = nn.MSELoss()(pred, y).item()
        metrics = calculate_metrics(y, pred)
    return loss, metrics


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, meta = get_unified_data(seq_len=12)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model = TrafficModel(
        model_type="GRU",
        input_dim=meta["num_features"],
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2,
        seq_len=meta["seq_len"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_epoch = -1
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()

        pred = model(X_train)
        loss = criterion(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_loss, val_metrics = evaluate(model, X_val, y_val)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {loss.item():.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSE: {val_metrics['RMSE']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "checkpoints/gru_model.pth")

    model.load_state_dict(torch.load("checkpoints/gru_model.pth", map_location=device))
    test_loss, test_metrics = evaluate(model, X_test, y_test)

    results = pd.DataFrame([{
        "Model": "GRU",
        "Best_Epoch": best_epoch,
        "Test_MSE": test_metrics["MSE"],
        "Test_MAE": test_metrics["MAE"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_STD_ERROR": test_metrics["STD_ERROR"]
    }])

    results.to_csv("results/gru_results.csv", index=False)

    print("\nBest epoch:", best_epoch)
    print("Saved checkpoint: checkpoints/gru_model.pth")
    print("Saved results: results/gru_results.csv")
    print(results)


if __name__ == "__main__":
    main()