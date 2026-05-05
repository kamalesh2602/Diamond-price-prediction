
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib
import os
os.makedirs("../models", exist_ok=True)
X = pd.read_csv("../data/X_processed.csv")
y = pd.read_csv("../data/y.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

lr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

joblib.dump(lr, "../models/linear_model.pkl")
joblib.dump(ridge, "../models/ridge_model.pkl")
joblib.dump(lasso, "../models/lasso_model.pkl")

print("Models Trained & Saved")