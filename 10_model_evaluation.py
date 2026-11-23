
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

# 2. Identify Columns
categorical_columns = ["Location", "Condition", "Garage"]

# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Rows:", len(X_train))
print("Testing Rows:", len(X_test))


# 5. Create Models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# 6. Cross Validation on Training Data Only
print("\n========== CROSS VALIDATION ==========")

results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    average_mae = -scores.mean()

    results[name] = average_mae

    print(name, "Average CV MAE:", round(average_mae, 2))


# 7. Select Best Model Using CV MAE
best_model_name = min(results, key=results.get)

print("\nBest Model:", best_model_name)


# 8. Train Best Model
best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[best_model_name])
])

best_model.fit(X_train, y_train)


# 9. Evaluate on Test Data
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== FINAL MODEL EVALUATION ==========")

print("Best Model:", best_model_name)
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# 10. Compare with Simple Average Price Baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

print("\n========== BASELINE COMPARISON ==========")

print("Baseline MAE:", round(baseline_mae, 2))
print("ML Model MAE:", round(mae, 2))


if mae < baseline_mae:
    print("Result: ML Model performed better than baseline.")
else:
    print("Result: ML Model did not beat the baseline.")


# 11. Save Final Model
joblib.dump(best_model, "house_price_model.pkl")

print("\nFinal Model Saved Successfully!")
print("Model Evaluation Completed!")
