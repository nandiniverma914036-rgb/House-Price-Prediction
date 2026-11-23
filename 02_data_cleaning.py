
import pandas as pd

# Step 1: Load Dataset
df = pd.read_csv("dataset/House Price Prediction Dataset.csv")

# Step 2: Check Duplicate Rows
print("Duplicate Rows:")
print(df.duplicated().sum())

# Step 3: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Remove Duplicate Rows
df = df.drop_duplicates()

# Step 5: Remove Id Column
df = df.drop("Id", axis=1)

# Step 6: Check Dataset Shape
print("\nCleaned Dataset Shape:")
print(df.shape)

# Step 7: Show Cleaned Dataset
print("\nCleaned Dataset:")
print(df.head())

# Step 8: Save Cleaned Dataset
df.to_csv("dataset/cleaned_house_data.csv", index=False)

print("\nData Cleaning Completed!")
