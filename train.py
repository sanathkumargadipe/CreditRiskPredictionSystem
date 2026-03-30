<<<<<<< HEAD
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

# ==============================
# 1. LOAD DATA
# ==============================
data = pd.read_csv("data/credit_data.csv")

# ==============================
# 2. SELECT FEATURES
# ==============================
features = [
    "credit_history",
    "amount",
    "duration",
    "age",
    "employment_duration",
    "savings",
    "purpose",
    "other_debtors",
    "housing",
    "job",
    "installment_rate",
    "property"
]

X = data[features]
y = data["risk"]

# ==============================
# 3. ENCODE CATEGORICAL DATA
# ==============================
le_dict = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

# ==============================
# 4. HANDLE IMBALANCE (SMOTE)
# ==============================
print("\nBefore SMOTE:\n", y.value_counts())

sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

print("\nAfter SMOTE:\n", pd.Series(y).value_counts())

# ==============================
# 5. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. MODELS
# ==============================

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=5
)

cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0
)

svm = SVC(probability=True)

# ==============================
# 7. STACKING MODEL
# ==============================
model = StackingClassifier(
    estimators=[
        ("lgbm", lgbm),
        ("cat", cat),
        ("svm", svm)
    ],
    final_estimator=LogisticRegression()
)

# ==============================
# 8. CROSS VALIDATION
# ==============================
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\n🔁 Cross-validation Accuracy: {np.mean(scores)*100:.2f} %")

# ==============================
# 9. TRAIN FINAL MODEL
# ==============================
model.fit(X_train, y_train)

# ==============================
# 9.5 TEST ACCURACY (NEW)
# ==============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Test Accuracy: {accuracy*100:.2f} %")

# ==============================
# 10. SAVE MODEL
# ==============================
joblib.dump(model, "credit_model.pkl")
joblib.dump(le_dict, "encoders.pkl")

print("\n✅ Model trained and saved successfully!")
=======
import joblib

from src.preprocessing import load_data, preprocess
from src.feature_selection import select_features
from src.train_model import build_model

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

data = load_data("data/credit_data.csv")

data = preprocess(data)

X, y = select_features(data)

sm = SMOTE()

X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

model = build_model()

model.fit(X_train, y_train)

joblib.dump(model, "models/credit_model.pkl")

print("Model trained successfully")
>>>>>>> 6c27dc6ea38ac1644b7746004d14c57db18b9a1d
