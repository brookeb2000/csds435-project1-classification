import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# region Data Loading

def load_training_csv(path: str):
    """
    loads training.csv file to y = labels x = rest of matrix
    Params:
        path            Path to training.csv
    """
    df = pd.read_csv(path)
    y = df.iloc[:, 0].astype(int).to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    return X, y

# endregion

# region Training

def train_naive_bayes(X, y, cv):
    """
    Traines multinomial naive bayes model by tuning alpah, selects alpha with highest mean CV accuracy
    Params:
        X               data matrix
        y               label array
        cv              cross-validation object (splits set into 5 and shuffles training/testing)
    """
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]

    best_alpha = None
    best_mean = -1.0
    best_std = None
    all_results = []

    for a in alphas:
        model = MultinomialNB(alpha=a)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        mean_acc = float(scores.mean())
        std_acc = float(scores.std())
        all_results.append({"alpha": a, "mean_acc": mean_acc, "std_acc": std_acc})

        if mean_acc > best_mean:
            best_mean = mean_acc
            best_std = std_acc
            best_alpha = a

    final_model = MultinomialNB(alpha=best_alpha)
    final_model.fit(X, y)

    info = {
        "library": "sklearn.naive_bayes.MultinomialNB",
        "best_params": {"alpha": best_alpha},
        "cv_results": all_results,
        "cv_mean_acc": best_mean,
        "cv_std_acc": best_std,
    }
    return final_model, info

def train_svm(X, y, cv):
    """
    Trains SVM model by tuning C and gamma and selects using GridSearch CV on our cv training object
    Params:
        X               data matrix
        y               label array
        cv              cross-validation object (splits set into 5 and shuffles training/testing)
    """
    # RBF SVM commonly needs C and gamma tuning
    param_grid = {
        "C": [0.5, 1, 2, 5, 10],
        "gamma": ["scale", 0.01, 0.03, 0.1],
        "kernel": ["rbf"],
    }

    base = SVC()
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    # Use the CV performance reported by GridSearchCV for the best setting
    best_mean = float(grid.best_score_)

    # Estimate std across folds for the chosen hyperparameters
    # (GridSearchCV stores split scores in cv_results_)
    best_idx = grid.best_index_
    split_keys = [k for k in grid.cv_results_.keys() if k.startswith("split") and k.endswith("_test_score")]
    split_scores = np.array([grid.cv_results_[k][best_idx] for k in split_keys], dtype=float)
    best_std = float(split_scores.std())

    info = {
        "library": "sklearn.svm.SVC",
        "best_params": grid.best_params_,
        "cv_mean_acc": best_mean,
        "cv_std_acc": best_std,
        "cv_results_note": "CV mean/std derived from GridSearchCV best setting.",
    }
    return best_model, info

def train_random_forest(X, y, cv):
    """
    Trains a random forest by tuning n_estimators, max_depth, max_featuers, min_samples_split and selects based on RandomizedSearchCV of the cv space
    Params:
        X               data matrix
        y               label array
        cv              cross-validation object (splits set into 5 and shuffles training/testing)
    """
    base = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 20, 40],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=12,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)

    best_model = search.best_estimator_
    best_mean = float(search.best_score_)

    best_idx = search.best_index_
    split_keys = [k for k in search.cv_results_.keys() if k.startswith("split") and k.endswith("_test_score")]
    split_scores = np.array([search.cv_results_[k][best_idx] for k in split_keys], dtype=float)
    best_std = float(split_scores.std())

    info = {
        "library": "sklearn.ensemble.RandomForestClassifier",
        "best_params": search.best_params_,
        "cv_mean_acc": best_mean,
        "cv_std_acc": best_std,
        "cv_results_note": "CV mean/std derived from RandomizedSearchCV best setting.",
    }
    return best_model, info

# endregion

# region Utils

def save_bundle(path, model, info):
    """
    saves a model to joblib file
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"model": model, "info": info}, path)

def plot_accuracies(results_dict, out_path):
    """
    plots all accuracies for the models included in this file
    Params:
        results_dict            Key = model name value = dict of cv_mean_acc and cv_std_acc
        out_path                path to where you need to save the model
    """
    names = list(results_dict.keys())
    means = [results_dict[n]["cv_mean_acc"] for n in names]
    stds = [results_dict[n]["cv_std_acc"] for n in names]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure()
    plt.bar(names, means, yerr=stds)
    plt.ylabel("Mean CV Accuracy")
    plt.title("Model Comparison (Training via Cross-Validation)")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# endregion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="Path to training.csv")
    parser.add_argument("--models_dir", default="models", help="Directory to save trained models")
    parser.add_argument("--figures_dir", default="figures", help="Directory to save figures")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of stratified CV folds")
    args = parser.parse_args()

    X, y = load_training_csv(args.train_csv)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    # Train each model and store results
    summary = {}

    nb_model, nb_info = train_naive_bayes(X, y, cv)
    save_bundle(os.path.join(args.models_dir, "naive_bayes.joblib"), nb_model, nb_info)
    summary["Naive Bayes"] = nb_info

    svm_model, svm_info = train_svm(X, y, cv)
    save_bundle(os.path.join(args.models_dir, "svm.joblib"), svm_model, svm_info)
    summary["SVM (RBF)"] = svm_info

    rf_model, rf_info = train_random_forest(X, y, cv)
    save_bundle(os.path.join(args.models_dir, "random_forest.joblib"), rf_model, rf_info)
    summary["Random Forest"] = rf_info

    # Print summary table
    print("\n=== CV Accuracy Summary (mean ± std) ===")
    for name, info in summary.items():
        print(f"{name:15s}: {info['cv_mean_acc']:.4f} ± {info['cv_std_acc']:.4f}")
        print(f"  Library: {info['library']}")
        print(f"  Best Params: {info['best_params']}\n")

    # Determine best model (by mean CV accuracy)
    best_name = max(summary.keys(), key=lambda n: summary[n]["cv_mean_acc"])
    print(f"Best Model by mean CV accuracy: {best_name} "
          f"({summary[best_name]['cv_mean_acc']:.4f})")

    # Save comparison figure
    fig_path = os.path.join(args.figures_dir, "cv_accuracy.png")
    plot_accuracies(summary, fig_path)
    print(f"Saved figure to: {fig_path}")

    # Also save a single summary bundle (optional but helpful)
    save_bundle(os.path.join(args.models_dir, "training_summary.joblib"), model=None, info=summary)
    print(f"Saved training summary to: {os.path.join(args.models_dir, 'training_summary.joblib')}")

if __name__ == "__main__":
    main()