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