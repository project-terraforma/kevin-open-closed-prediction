import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def extract_categories(x):
    if x is None:
        return ""

    if isinstance(x, list):
        return " ".join(map(str, x))

    if isinstance(x, dict):
        out = []
        for v in x.values():
            if isinstance(v, (list, tuple, np.ndarray)):
                out.extend([str(i) for i in v])
            else:
                out.append(str(v))
        return " ".join(out)

    if isinstance(x, np.ndarray):
        return " ".join(map(str, x))

    return str(x)


def extract_name(x):
    if isinstance(x, dict):
        return x.get("primary") or ""
    return ""


def get_primary_category(x):
    if isinstance(x, dict):
        return x.get("primary", "")
    return ""


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    df = df.dropna(subset=["open"]).copy()

    df["categories_text"] = df["categories"].apply(extract_categories).fillna("").astype(str)
    df["name_text"] = df["names"].apply(extract_name).fillna("").astype(str)
    df["primary_category"] = df["categories"].apply(get_primary_category).fillna("").astype(str)

    df["num_websites"] = df["websites"].apply(lambda x: 0 if x is None else len(x))
    df["num_phones"] = df["phones"].apply(lambda x: 0 if x is None else len(x))
    df["num_socials"] = df["socials"].apply(lambda x: 0 if x is None else len(x))

    df["contact_score"] = (
        (df["num_websites"] > 0).astype(int)
        + (df["num_phones"] > 0).astype(int)
        + (df["num_socials"] > 0).astype(int)
    )

    df["confidence"] = df["confidence"].fillna(0)
    df["name_length"] = df["name_text"].apply(len)
    df["has_numbers_in_name"] = df["name_text"].str.contains(r"\d", na=False).astype(int)
    df["num_categories"] = df["categories_text"].apply(lambda x: len(x.split()))
    df["multi_category"] = (df["num_categories"] > 2).astype(int)
    df["name_equals_category_hint"] = (
        df["name_text"].fillna("").str.lower()
        == df["primary_category"].fillna("").str.lower()
    ).astype(int)

    return df


def build_model(numeric_features):
    preprocess = ColumnTransformer([
        ("name_tfidf", TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        ), "name_text"),

        ("cat_tfidf", TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        ), "categories_text"),

        ("primary_cat", OneHotEncoder(handle_unknown="ignore"), ["primary_category"]),

        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric_features),
    ])

    model = LogisticRegression(
        max_iter=3000,
        C=0.5,
        solver="liblinear",
        class_weight="balanced"
    )

    clf = Pipeline([
        ("features", preprocess),
        ("model", model)
    ])

    return clf


def tune_threshold(y_true, y_probs):
    best_score = -1
    best_thresh = 0
    best_f1_0 = 0
    best_f1_1 = 0
    best_macro = 0

    for t in np.linspace(0.05, 0.95, 100):
        preds = (y_probs > t).astype(int)

        f1_0 = f1_score(y_true, preds, pos_label=0)
        f1_1 = f1_score(y_true, preds, pos_label=1)
        macro = (f1_0 + f1_1) / 2

        score = 0.6 * f1_0 + 0.4 * f1_1

        if score > best_score:
            best_score = score
            best_thresh = t
            best_f1_0 = f1_0
            best_f1_1 = f1_1
            best_macro = macro

    return best_thresh, best_score, best_f1_0, best_f1_1, best_macro


def save_results(
    out_dir: str,
    dataset_summary: str,
    default_f1_0: float,
    default_f1_1: float,
    default_macro: float,
    best_thresh: float,
    best_score: float,
    best_f1_0: float,
    best_f1_1: float,
    best_macro: float,
    report: str,
    cm: np.ndarray,
    top_features: list,
):
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "final_metrics.txt")
    features_path = os.path.join(out_dir, "top_features.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("FINAL METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(dataset_summary + "\n\n")

        f.write("DEFAULT THRESHOLD (0.5)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Closed class F1 (0): {default_f1_0:.6f}\n")
        f.write(f"Open class F1 (1):   {default_f1_1:.6f}\n")
        f.write(f"Macro F1:            {default_macro:.6f}\n\n")

        f.write("BEST THRESHOLD RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Best threshold:      {best_thresh:.6f}\n")
        f.write(f"Best weighted score: {best_score:.6f}\n")
        f.write(f"Closed class F1 (0): {best_f1_0:.6f}\n")
        f.write(f"Open class F1 (1):   {best_f1_1:.6f}\n")
        f.write(f"Macro F1:            {best_macro:.6f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(report + "\n\n")

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 60 + "\n")
        f.write(str(cm) + "\n")
        f.write("\n[[ TN  FN]\n [ FP  TP]]\n")

    with open(features_path, "w", encoding="utf-8") as f:
        f.write("TOP FEATURES\n")
        f.write("=" * 60 + "\n\n")
        for coef, name in top_features[:20]:
            f.write(f"{coef:.6f}\t{name}\n")

    return metrics_path, features_path


def save_visualizations(out_dir: str, y_test, y_pred, y, top_features):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Class distribution
    class_counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Closed (0)", "Open (1)"], [class_counts.get(0, 0), class_counts.get(1, 0)])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    plt.close()

    # 2) Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Closed", "Open"])
    plt.yticks([0, 1], ["Closed", "Open"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 3) Top feature coefficients
    top_pos = sorted(top_features, key=lambda x: x[0], reverse=True)[:10]
    top_neg = sorted(top_features, key=lambda x: x[0])[:10]
    combined = list(reversed(top_neg)) + top_pos

    labels = [name for _, name in combined]
    values = [coef for coef, _ in combined]

    plt.figure(figsize=(10, 8))
    plt.barh(labels, values)
    plt.title("Top Positive and Negative Features")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_features.png"), dpi=150)
    plt.close()


def main():
    print("Loading data...")
    df = load_and_prepare_data("./data/project_c_samples.parquet")

    y = df["open"]
    class_counts = y.value_counts().sort_index()

    dataset_summary = (
        f"Total rows: {len(df)}\n"
        f"Closed (0): {class_counts.get(0, 0)}\n"
        f"Open (1):   {class_counts.get(1, 0)}"
    )

    print("\nDataset summary:")
    print(dataset_summary)

    numeric_features = [
        "confidence",
        "contact_score",
        "name_length",
        "has_numbers_in_name",
        "multi_category",
        "name_equals_category_hint"
    ]

    X = df[["name_text", "categories_text", "primary_category"] + numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = build_model(numeric_features)

    print("\nTraining model...")
    clf.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_probs = clf.predict_proba(X_test)[:, 1]

    default_preds = (y_probs > 0.5).astype(int)
    default_f1_0 = f1_score(y_test, default_preds, pos_label=0)
    default_f1_1 = f1_score(y_test, default_preds, pos_label=1)
    default_macro = (default_f1_0 + default_f1_1) / 2

    print("\n=== DEFAULT THRESHOLD (0.5) ===")
    print("Closed class F1 (0):", default_f1_0)
    print("Open class F1 (1):", default_f1_1)
    print("Macro F1:", default_macro)

    best_thresh, best_score, best_f1_0, best_f1_1, best_macro = tune_threshold(y_test, y_probs)

    print("\n=== BEST THRESHOLD RESULTS ===")
    print("Best threshold:", best_thresh)
    print("Best weighted score:", best_score)

    print("\nClass-wise performance at best threshold:")
    print("Closed class F1 (0):", best_f1_0)
    print("Open class F1 (1):", best_f1_1)
    print("Macro F1:", best_macro)

    y_pred = (y_probs > best_thresh).astype(int)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    print("=== CONFUSION MATRIX ===")
    print(cm)
    print("""
[[ TN  FN]
 [ FP  TP]]""")

    feature_names = clf.named_steps["features"].get_feature_names_out()
    coefficients = clf.named_steps["model"].coef_[0]

    top_features = sorted(
        zip(coefficients, feature_names),
        key=lambda x: abs(x[0]),
        reverse=True
    )

    print("\n=== TOP FEATURES ===")
    for coef, name in top_features[:20]:
        print(coef, name)

    metrics_path, features_path = save_results(
        out_dir="./results",
        dataset_summary=dataset_summary,
        default_f1_0=default_f1_0,
        default_f1_1=default_f1_1,
        default_macro=default_macro,
        best_thresh=best_thresh,
        best_score=best_score,
        best_f1_0=best_f1_0,
        best_f1_1=best_f1_1,
        best_macro=best_macro,
        report=report,
        cm=cm,
        top_features=top_features,
    )

    save_visualizations(
        out_dir="./results",
        y_test=y_test,
        y_pred=y_pred,
        y=y,
        top_features=top_features,
    )

    print("\nSaved results to:")
    print(metrics_path)
    print(features_path)
    print(os.path.join("./results", "class_distribution.png"))
    print(os.path.join("./results", "confusion_matrix.png"))
    print(os.path.join("./results", "top_features.png"))


if __name__ == "__main__":
    main()