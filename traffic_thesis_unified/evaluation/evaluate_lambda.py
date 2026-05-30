import os
import subprocess
import pandas as pd


LAMBDAS = [0.001, 0.01, 0.1, 0.5, 1.0]


def main():
    all_results = []

    for lam in LAMBDAS:
        print(f"\nRunning PINN with lambda = {lam}")

        env = os.environ.copy()
        env["LAMBDA_PHYSICS"] = str(lam)

        subprocess.run(
            ["/usr/local/bin/python3", "training/train_pinn.py"],
            env=env,
            check=True
        )

        df = pd.read_csv("results/pinn_results.csv")
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values(by="Test_MSE", ascending=True)

    final_df.to_csv("results/pinn_lambda_comparison.csv", index=False)

    print("\nSaved: results/pinn_lambda_comparison.csv")
    print(final_df)


if __name__ == "__main__":
    main()