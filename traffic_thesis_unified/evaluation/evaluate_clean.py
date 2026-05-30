import os
import pandas as pd
import matplotlib.pyplot as plt


RESULT_FILES = [
    "results/lstm_results.csv",
    "results/gru_results.csv",
    "results/cnn_results.csv",
    "results/pinn_results.csv",   # اگر بعداً ساخته شد، خودکار اضافه می‌شود
    "results/pinn_lwr_results.csv"  # اگر بعداً ساخته شد، خودکار اضافه می‌شود
]


def save_bar_chart(df, metric, output_path, title, color):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["Model"], df[metric], color=color, edgecolor="black")

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_grouped_chart(df, metrics, output_path, title):
    plot_df = df[["Model"] + metrics].copy()
    plot_df = plot_df.set_index("Model")

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
        edgecolor="black"
    )

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Metrics")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    all_results = []

    for file_path in RESULT_FILES:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_results.append(df)
        else:
            print(f"Missing result file: {file_path}")

    if len(all_results) == 0:
        print("No result files found.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values(by="Test_MSE", ascending=True)

    os.makedirs("results", exist_ok=True)

    csv_path = "results/clean_baseline_comparison.csv"
    final_df.to_csv(csv_path, index=False)

    mse_chart_path = "results/chart_mse_comparison.png"
    grouped_chart_path = "results/chart_metrics_comparison.png"

    save_bar_chart(
        final_df,
        metric="Test_MSE",
        output_path=mse_chart_path,
        title="Model Comparison - Test MSE",
        color="skyblue"
    )

    save_grouped_chart(
        final_df,
        metrics=["Test_MAE", "Test_RMSE", "Test_STD_ERROR"],
        output_path=grouped_chart_path,
        title="Model Comparison - MAE, RMSE, STD_ERROR"
    )

    print("\nSaved:", csv_path)
    print("Saved:", mse_chart_path)
    print("Saved:", grouped_chart_path)
    print(final_df)


if __name__ == "__main__":
    main()