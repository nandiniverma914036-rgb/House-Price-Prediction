
# Step 1: Import Pandas
import pandas as pd

# Step 2: Load Dataset
df = pd.read_csv("dataset/House Price Prediction Dataset.csv")

# Step 3: Show First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# Step 4: Show Total Rows and Columns
print("\nDataset Shape:")
print(df.shape)

# Step 5: Show Column Names
print("\nColumn Names:")
print(df.columns)

# Step 6: Dataset Information
print("\nDataset Information:")
df.info()

# Step 7: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 8: Statistical Information
print("\nStatistical Summary:")
print(df.describe())
