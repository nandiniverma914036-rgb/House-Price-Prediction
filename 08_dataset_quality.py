
import pandas as pd

# Load Dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

# 1. Check Unique Values
print("========== UNIQUE VALUES ==========")

for column in df.columns:
    print(column, ":", df[column].nunique())

# 2. Check Price Distribution
print("\n========== PRICE DISTRIBUTION ==========")

print(df["Price"].describe())

# 3. Check Location Categories
print("\n========== LOCATION VALUES ==========")

print(df["Location"].value_counts())

# 4. Check Condition Categories
print("\n========== CONDITION VALUES ==========")

print(df["Condition"].value_counts())

# 5. Check Garage Categories
print("\n========== GARAGE VALUES ==========")

print(df["Garage"].value_counts())

# 6. Check Feature Relationship
print("\n========== GROUP PRICE ANALYSIS ==========")

print(df.groupby("Location")["Price"].agg(
    ["count", "mean", "median", "std"]
))

print("\nDataset Quality Check Completed!")
