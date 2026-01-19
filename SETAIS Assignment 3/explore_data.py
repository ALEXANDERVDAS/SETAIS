import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Load the CSV
# -----------------------------
csv_path = "rollouts.csv"
df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Separate features and target
# -----------------------------
target = "crashed"
X = df.drop(columns=[target, "closest_same_lane"])
y = df[target]

# Optional: handle NaN values (from closest distances etc.)
X = X.fillna(X.median())

# -----------------------------
# 3. Correlation with target
# -----------------------------
# Pearson correlation of each feature with the crash label
corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)

print("\nCorrelation of features with 'crashed':")
print(corr)

# Plot correlations
plt.figure(figsize=(10, 6))
corr.drop(target).plot(kind="barh")
plt.axvline(0, linestyle="--")
plt.title("Feature Correlation with Collision Target")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.show()


# -----------------------------
# 4. Train a simple crash predictor
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# -----------------------------
# 5. Feature importance (model-based correlation)
# -----------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nModel Feature Importances:")
print(importances)

plt.figure(figsize=(10, 6))
importances.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
