import joblib
# Step 1: Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load Cleaned Dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

# Step 3: Separate Input and Output
X = df.drop("Price", axis=1)
y = df["Price"]

# Step 4: Select Numerical and Categorical Columns
numerical_features = [
    "Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"
]

categorical_features = [
    "Location", "Condition", "Garage"
]

# Step 5: Convert Text Data into Numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Step 6: Split Dataset (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Rows:", len(X_train))
print("Testing Rows:", len(X_test))

# Step 7: Create Machine Learning Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Step 8: Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# Step 11: Save Trained Model

joblib.dump(model, "house_price_model.pkl")

print("\nModel Saved Successfully!")
