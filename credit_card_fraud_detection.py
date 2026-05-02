import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv(r"D:\archive (16)\creditcard.csv")
X = df.iloc[:, 1:-1]
y = df["Class"]

# =========================
# 3. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4. BEFORE SMOTE MODEL
# =========================
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# 5. SMOTE APPLY
# =========================
smote = SMOTE(random_state=42)
X_train_smo, y_train_smo = smote.fit_resample(X_train, y_train)

# =========================
# 6. AFTER SMOTE MODEL
# =========================
model_smote = DecisionTreeClassifier(max_depth=3, random_state=42)
model_smote.fit(X_train_smo, y_train_smo)
y_pred_smote = model_smote.predict(X_test)

# =========================
# GRAPH 1: BEFORE SMOTE DISTRIBUTION
# =========================
plt.figure()
y_train.value_counts().plot(kind="bar")
plt.title("Before SMOTE Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("before_smote_distribution.png")
plt.show()

# =========================
# GRAPH 2: AFTER SMOTE DISTRIBUTION
# =========================
plt.figure()
pd.Series(y_train_smo).value_counts().plot(kind="bar")
plt.title("After SMOTE Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("after_smote_distribution.png")
plt.show()

# =========================
# GRAPH 3: CONFUSION MATRIX (SMOTE MODEL)
# =========================
cm = confusion_matrix(y_test, y_pred_smote)
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (After SMOTE)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# GRAPH 4: FEATURE IMPORTANCE
# =========================
importances = model_smote.feature_importances_
plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# =========================
# 7. FINAL RESULTS
# =========================
print("Accuracy (Before SMOTE):", accuracy_score(y_test, y_pred))
print("Accuracy (After SMOTE):", accuracy_score(y_test, y_pred_smote))
print("\nClassification Report (After SMOTE):\n", classification_report(y_test, y_pred_smote))
