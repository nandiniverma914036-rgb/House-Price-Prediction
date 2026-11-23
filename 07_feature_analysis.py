
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

# 1. Price Information
print("========== PRICE ANALYSIS ==========")

print("Average Price:", df["Price"].mean())
print("Minimum Price:", df["Price"].min())
print("Maximum Price:", df["Price"].max())

# 2. Correlation Analysis
print("\n========== FEATURE CORRELATION ==========")

correlation = df.corr(numeric_only=True)

print(correlation["Price"].sort_values(ascending=False))

# 3. Correlation Heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("House Price Feature Correlation")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")

plt.show()

# 4. Average Price by Location
print("\n========== LOCATION ANALYSIS ==========")

print(
    df.groupby("Location")["Price"].mean()
)

# 5. Average Price by Condition
print("\n========== CONDITION ANALYSIS ==========")

print(
    df.groupby("Condition")["Price"].mean()
)

# 6. Average Price by Garage
print("\n========== GARAGE ANALYSIS ==========")

print(
    df.groupby("Garage")["Price"].mean()
)

print("\nFeature Analysis Completed!")
