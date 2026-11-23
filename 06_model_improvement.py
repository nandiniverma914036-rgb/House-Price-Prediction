
# Step 1: Import Libraries

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load Dataset

df = pd.read_csv("dataset/cleaned_house_data.csv")

# Step 3: Separate Input and Output

X = df.drop("Price", axis=1)
y = df["Price"]

# Step 4: Define Columns

categorical_features = [
    "Location", "Condition", "Garage"
]

# Step 5: Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Step 6: Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Create Algorithms

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

# Step 8: Compare Models

results = []
best_mae = float("inf")
best_model = None
best_name = ""

for name, algorithm in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test, predictions
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "R2 Score": round(r2, 4)
    })

    print("\nModel:", name)
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 4))

    if mae < best_mae:
        best_mae = mae
        best_model = pipeline
        best_name = name

# Step 9: Display Comparison

print("\n========== MODEL COMPARISON ==========")

result_df = pd.DataFrame(results)
print(result_df.sort_values("MAE"))

# Step 10: Save Best Model

joblib.dump(best_model, "house_price_model.pkl")

print("\nBest Model:", best_name)
print("Best Model Saved Successfully!")
