import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


def main():
    clean_path = os.path.join(RESULTS_DIR, "clean_baseline_comparison.csv")
    regime_path = os.path.join(RESULTS_DIR, "traffic_regime_comparison.csv")
    attack_path = os.path.join(RESULTS_DIR, "attack_robustness_comparison.csv")

    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Missing file: {clean_path}")
    if not os.path.exists(regime_path):
        raise FileNotFoundError(f"Missing file: {regime_path}")
    if not os.path.exists(attack_path):
        raise FileNotFoundError(f"Missing file: {attack_path}")

    clean_df = pd.read_csv(clean_path)
    regime_df = pd.read_csv(regime_path)
    attack_df = pd.read_csv(attack_path)

    clean_keep = clean_df[["Model", "Test_MSE", "Test_MAE", "Test_RMSE"]].copy()
    clean_keep = clean_keep.rename(columns={
        "Test_MSE": "Clean_MSE",
        "Test_MAE": "Clean_MAE",
        "Test_RMSE": "Clean_RMSE"
    })

    heavy_df = regime_df[regime_df["Regime"] == "heavy"][["Model", "RMSE", "MAE"]].copy()
    heavy_df = heavy_df.rename(columns={
        "RMSE": "Heavy_RMSE",
        "MAE": "Heavy_MAE"
    })

    stable_df = regime_df[regime_df["Regime"] == "stable"][["Model", "RMSE", "MAE"]].copy()
    stable_df = stable_df.rename(columns={
        "RMSE": "Stable_RMSE",
        "MAE": "Stable_MAE"
    })

    attack_050 = attack_df[attack_df["Noise_STD"] == 0.50][
        ["Model", "RMSE", "RMSE_Degradation_Pct"]
    ].copy()
    attack_050 = attack_050.rename(columns={
        "RMSE": "Attack_RMSE_0.50",
        "RMSE_Degradation_Pct": "Attack_DegradationPct_0.50"
    })

    final_df = clean_keep.merge(heavy_df, on="Model", how="left")
    final_df = final_df.merge(stable_df, on="Model", how="left")
    final_df = final_df.merge(attack_050, on="Model", how="left")

    final_df["Rank_Clean_RMSE"] = final_df["Clean_RMSE"].rank(method="min")
    final_df["Rank_Heavy_RMSE"] = final_df["Heavy_RMSE"].rank(method="min")
    final_df["Rank_Stable_RMSE"] = final_df["Stable_RMSE"].rank(method="min")
    final_df["Rank_Attack_RMSE"] = final_df["Attack_RMSE_0.50"].rank(method="min")
    final_df["Rank_Robustness"] = final_df["Attack_DegradationPct_0.50"].rank(method="min")

    final_df = final_df.sort_values(by="Clean_RMSE", ascending=True)

    output_csv = os.path.join(RESULTS_DIR, "final_model_comparison.csv")
    final_df.to_csv(output_csv, index=False)

    plot_df = final_df.set_index("Model")[[
        "Clean_RMSE", "Heavy_RMSE", "Stable_RMSE", "Attack_RMSE_0.50"
    ]]

    ax = plot_df.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title("Final Model Comparison Across Evaluation Settings")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_png = os.path.join(RESULTS_DIR, "final_model_comparison.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_csv}")
    print(f"Saved: {output_png}")
    print(final_df)


if __name__ == "__main__":
    main()