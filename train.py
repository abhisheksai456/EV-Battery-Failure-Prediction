import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# =====================================
# 1. LOAD DATASET
# =====================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "ev_battery_failure_dataset.csv")

print("Dataset shape:", df.shape)

# =====================================
# 2. REMOVE UNNECESSARY COLUMNS
# =====================================

drop_columns = [
    "vehicle_id",
    "battery_serial",
    "predicted_remaining_life_cycles"
]

df = df.drop(columns=drop_columns, errors="ignore")

X = df.drop(columns=["battery_failure"])
y = df["battery_failure"].astype(int)

# =====================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# =====================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# =====================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL
# =====================================

num = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

cat = [
    col for col in X_train.columns
    if col not in num
]

print("\nNumerical features:", len(num))
print("Categorical features:", len(cat))

# =====================================
# 5. PREPROCESSING
# =====================================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num),
    ("cat", cat_pipeline, cat)
])

# =====================================
# 6. INITIAL RANDOM FOREST MODEL
# =====================================

initial_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining initial Random Forest...")

initial_model.fit(X_train, y_train)

# =====================================
# 7. SELECT TOP 15 FEATURES
# =====================================

transformed_names = (
    initial_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    initial_model
    .named_steps["model"]
    .feature_importances_
)

importance_dict = {}

for col in X_train.columns:

    if col in num:
        mask = transformed_names == f"num__{col}"

    else:
        prefix = f"cat__{col}_"

        mask = np.array([
            name.startswith(prefix)
            for name in transformed_names
        ])

    importance_dict[col] = importances[mask].sum()

feature_importance = pd.Series(
    importance_dict
).sort_values(ascending=False)

selected_features = (
    feature_importance.head(15).index.tolist()
)

print("\nTop 15 Selected Features:")
print(feature_importance.head(15))

# =====================================
# 8. FINAL RANDOM FOREST MODEL
# =====================================

selected_num = [
    col for col in selected_features
    if col in num
]

selected_cat = [
    col for col in selected_features
    if col in cat
]

final_preprocessor = ColumnTransformer([
    ("num",
     num_pipeline,
     selected_num),

    ("cat",
     cat_pipeline,
     selected_cat)
])

rf_model = Pipeline([
    ("preprocessor", final_preprocessor),

    ("model", RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining final Random Forest with 15 features...")

rf_model.fit(
    X_train[selected_features],
    y_train
)

# =====================================
# 9. TUNE THRESHOLD USING VALIDATION DATA
# =====================================

val_probabilities = rf_model.predict_proba(
    X_val[selected_features]
)[:, list(rf_model.classes_).index(1)]

fpr, tpr, thresholds = roc_curve(
    y_val,
    val_probabilities
)

specificity = 1 - fpr

balanced_accuracy = (tpr + specificity) / 2

# Prefer thresholds with strong balanced accuracy
# and similar sensitivity and specificity.

valid = np.isfinite(thresholds)

scores = (
    balanced_accuracy[valid]
    - 0.0001 * np.abs(tpr[valid] - specificity[valid])
)

best_index = np.argmax(scores)

best_threshold = thresholds[valid][best_index]

print("\nSelected threshold:", best_threshold)

# =====================================
# 10. TEST FINAL MODEL
# =====================================

test_probabilities = rf_model.predict_proba(
    X_test[selected_features]
)[:, list(rf_model.classes_).index(1)]

y_pred = (
    test_probabilities >= best_threshold
).astype(int)

# =====================================
# 11. MODEL PERFORMANCE
# =====================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n========== RANDOM FOREST RESULTS ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# =====================================
# 12. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n========== CONFUSION MATRIX ==========")

print(cm)

print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)

print("\nDifference between TN and TP:", abs(tn - tp))

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    y_pred,
    target_names=["No Failure", "Battery Failure"],
    zero_division=0
))

# =====================================
# 13. SAVE FINAL MODEL
# =====================================

model_data = {
    "model": rf_model,
    "feature_columns": selected_features,
    "threshold": float(best_threshold)
}

backend_dir = BASE_DIR / "backend"
backend_dir.mkdir(exist_ok=True)

model_path = backend_dir / "model.pkl"

joblib.dump(
    model_data,
    model_path,
    compress=9
)

file_size_mb = model_path.stat().st_size / (1024 * 1024)
print(f"\nModel saved successfully! Size: {file_size_mb:.2f} MB")
print("Selected features:", selected_features)