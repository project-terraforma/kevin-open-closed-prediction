import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------
# 1. Load data
# --------------------
df = pd.read_parquet("./data/project_c_samples.parquet")

# --------------------
# 2. Basic cleanup
# --------------------
df = df.dropna(subset=["open"])

y = df["open"]

# --------------------
# 3. Feature engineering
# --------------------

# Convert categories (likely list-like or dict-like)
def extract_categories(x):
    if x is None:
        return ""

    # Case 1: already list-like
    if isinstance(x, list):
        return " ".join(map(str, x))

    # Case 2: dict-like (but values may be arrays)
    if isinstance(x, dict):
        out = []
        for v in x.values():
            if isinstance(v, (list, tuple, np.ndarray)):
                out.extend([str(i) for i in v])
            else:
                out.append(str(v))
        return " ".join(out)

    # Case 3: numpy array directly
    if isinstance(x, np.ndarray):
        return " ".join(map(str, x))

    return str(x)

df["categories_text"] = df["categories"].apply(extract_categories).fillna("").astype(str)

def extract_name(x):
    if isinstance(x, dict):
        return x.get("primary") or ""
    return ""

df["name_text"] = df["names"].apply(extract_name)

# Count-based features (simple signal strength proxies)
df["num_websites"] = df["websites"].apply(lambda x: 0 if x is None else len(x))
df["num_phones"] = df["phones"].apply(lambda x: 0 if x is None else len(x))
df["num_socials"] = df["socials"].apply(lambda x: 0 if x is None else len(x))

df["contact_score"] = (
    (df["num_websites"] > 0).astype(int) +
    (df["num_phones"] > 0).astype(int) +
    (df["num_socials"] > 0).astype(int)
)

df["low_contact"] = (df["contact_score"] <= 1).astype(int)

# Confidence (already numeric but ensure clean)
df["confidence"] = df["confidence"].fillna(0)

df["name_length"] = df["name_text"].apply(len)

def get_primary_category(x):
    if isinstance(x, dict):
        return x.get("primary", "")
    return ""
df["primary_category"] = df["categories"].apply(get_primary_category).fillna("")

df["has_numbers_in_name"] = df["name_text"].str.contains(r"\d", na=False).astype(int)

df["num_categories"] = df["categories_text"].apply(lambda x: len(x.split()))
df["multi_category"] = (df["num_categories"] > 2).astype(int)

df["name_equals_category_hint"] = (
    df["name_text"].fillna("").str.lower()
    == df["primary_category"].fillna("").str.lower()
).astype(int)

# --------------------
# 4. Feature columns
# --------------------

numeric_features = [
    "confidence",
    "contact_score",
    "name_length",
    "has_numbers_in_name",
    "multi_category",
    "name_equals_category_hint"
]

# --------------------
# 5. Preprocessing
# --------------------

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

# --------------------
# 6. Model
# --------------------

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

# --------------------
# 7. Train/test split
# --------------------
X = df[["name_text", "categories_text", "primary_category"] + numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------
# 8. Train
# --------------------
clf.fit(X_train, y_train)

# --------------------
# 9. Evaluate
# --------------------

y_probs = clf.predict_proba(X_test)[:, 1]

best_score = -1
best_thresh = 0

best_f1_0 = 0
best_f1_1 = 0
best_macro = 0

for t in np.linspace(0.05, 0.95, 100):
    preds = (y_probs > t).astype(int)

    f1_0 = f1_score(y_test, preds, pos_label=0)
    f1_1 = f1_score(y_test, preds, pos_label=1)
    macro = (f1_0 + f1_1) / 2

    # your current optimization objective (kept unchanged)
    score = 0.6 * f1_0 + 0.4 * f1_1

    if score > best_score:
        best_score = score
        best_thresh = t
        best_f1_0 = f1_0
        best_f1_1 = f1_1
        best_macro = macro

print("\n=== BEST THRESHOLD RESULTS ===")
print("Best threshold:", best_thresh)
print("Best weighted score:", best_score)

print("\nClass-wise performance at best threshold:")
print("Closed class F1 (0):", best_f1_0)
print("Open class F1 (1):", best_f1_1)
print("Macro F1:", best_macro)

# final predictions
y_pred = (y_probs > best_thresh).astype(int)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))
print("""
[[ TN  FN]
 [ FP  TP]]""")

feature_names = clf.named_steps["features"].get_feature_names_out()
coefficients = clf.named_steps["model"].coef_[0]

# Look at top features
top_features = sorted(zip(coefficients, feature_names), key=lambda x: abs(x[0]), reverse=True)

for coef, name in top_features[:20]:
    print(coef, name)