import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --------------------
# 1. Load data
# --------------------
df = pd.read_parquet("project_c_samples.parquet")

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

# Combine name + brand as text signal
df["name_text"] = (
    df["names"].fillna("").astype(str)
    + " "
    + df["brand"].fillna("").astype(str)
)

# Count-based features (simple signal strength proxies)
df["num_websites"] = df["websites"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df["num_phones"] = df["phones"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df["num_socials"] = df["socials"].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Confidence (already numeric but ensure clean)
df["confidence"] = df["confidence"].fillna(0)

# --------------------
# 4. Feature columns
# --------------------
text_features = "name_text"
category_features = "categories_text"

numeric_features = [
    "confidence",
    "num_websites",
    "num_phones",
    "num_socials"
]

text_cols = ["name_text", "categories_text"]
num_cols = numeric_features

X = df[text_cols + num_cols]

# Actually split properly:
X_text = df[["name_text", "categories_text"]]
X_num = df[numeric_features]

# --------------------
# 5. Preprocessing
# --------------------

text_transformer = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english"))
])

from sklearn.pipeline import FeatureUnion

preprocess = ColumnTransformer([
    ("name_tfidf", TfidfVectorizer(max_features=5000), "name_text"),
    ("cat_tfidf", TfidfVectorizer(max_features=2000), "categories_text"),
    ("num", SimpleImputer(strategy="median"), numeric_features)
])

# --------------------
# 6. Model
# --------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")

clf = Pipeline([
    ("features", preprocess),
    ("model", model)
])

# --------------------
# 7. Train/test split
# --------------------
X = df[["name_text", "categories_text"] + numeric_features]

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
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

print("Closed F1:",
      f1_score(y_test, y_pred, pos_label=0))

print(confusion_matrix(y_test, y_pred))