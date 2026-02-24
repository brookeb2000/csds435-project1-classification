import argparse
import os
import joblib
import numpy as np
import pandas as pd


def load_test_csv(path: str) -> np.ndarray:
    """
    loads testing.cv for validation
    """
    df = pd.read_csv(path)
    return df.to_numpy(dtype=np.float64)


def write_predictions(preds: np.ndarray, out_path: str) -> None:
    """
    Writes one prediction per line for outputting to file at out_path
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(f"{int(p)}\n")

def main():

    # pares CLI arguments for running
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True, help="Path to testing.csv")
    parser.add_argument("--models_dir", default="models", help="Directory containing saved .joblib models")
    parser.add_argument("--pred_dir", default="predictions", help="Directory to write prediction files")
    parser.add_argument("--summary_in", default="models/training_summary.joblib",
                        help="Path to training summary joblib to identify best model")
    args = parser.parse_args()

    X_test = load_test_csv(args.test_csv)

    # These filenames must match what train_all.py saves
    model_files = {
        "naive_bayes": os.path.join(args.models_dir, "naive_bayes.joblib"),
        "svm": os.path.join(args.models_dir, "svm.joblib"),
        "random_forest": os.path.join(args.models_dir, "random_forest.joblib"),
    }

    # Predict with each model and write outputs
    print("\n=== Running test predictions ===")
    for key, path in model_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {path} (run train_all.py first)")

        bundle = joblib.load(path)
        model = bundle["model"]

        preds = model.predict(X_test)
        out_path = os.path.join(args.pred_dir, f"pred_{key}.txt")
        write_predictions(preds, out_path)
        print(f"{key:13s} -> wrote {out_path}")

    # Determine best model
    if os.path.exists(args.summary_in):
        summary_bundle = joblib.load(args.summary_in)
        info = summary_bundle.get("info", summary_bundle)

        best_name = max(info.keys(), key=lambda name: info[name]["cv_mean_acc"])
        print(f"\nBest model from CV summary: {best_name} ({info[best_name]['cv_mean_acc']:.4f})")

        # Map that display name to model filename keys
        display_to_key = {
            "Naive Bayes": "naive_bayes",
            "SVM (RBF)": "svm",
            "Random Forest": "random_forest",
        }
        best_key = display_to_key.get(best_name, None)

        if best_key is not None:
            best_model_path = model_files[best_key]
            best_bundle = joblib.load(best_model_path)
            best_model = best_bundle["model"]
            best_preds = best_model.predict(X_test)

            best_out = os.path.join(args.pred_dir, "pred_best.txt")
            write_predictions(best_preds, best_out)
            print(f"Wrote best-model predictions to: {best_out}")
        else:
            print("Warning: Could not map best model name to a saved file key; pred_best.txt not written.")
    else:
        print(f"\nWarning: summary file not found at {args.summary_in}. ")

if __name__ == "__main__":
    main()