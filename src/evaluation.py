
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import train_test_split

# Load data
X = pd.read_csv("../data/X_processed.csv")
y = pd.read_csv("../data/y.csv")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Load models
models = {
    "Linear Regression": joblib.load("../models/linear_model.pkl"),
    "Ridge Regression": joblib.load("../models/ridge_model.pkl"),
    "Lasso Regression": joblib.load("../models/lasso_model.pkl")
}

results = []

# Evaluate each model
for name, model in models.items():
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append([name, mse, rmse, r2])


# Convert to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "R2"])
print(results_df)
# Save results
results_df.to_csv("../outputs/results.csv", index=False)

# Also save readable text
with open("../outputs/results.txt", "w") as f:
    f.write(results_df.to_string(index=False))

print("Evaluation completed. Results saved.")