
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

# 1. Price Distribution
print("========== PRICE DISTRIBUTION ==========")

print(df["Price"].describe())

# 2. Check Price Skewness
print("\n========== PRICE SKEWNESS ==========")

print("Skewness:", df["Price"].skew())

# 3. Check Price Quartiles
print("\n========== PRICE QUARTILES ==========")

print(df["Price"].quantile([0.25, 0.50, 0.75]))

# 4. Compare Features with Price
print("\n========== FEATURE CORRELATION ==========")

print(
    df.select_dtypes(include="number")
    .corr()["Price"]
    .sort_values(ascending=False)
)

# 5. Create Graphs
plt.figure(figsize=(10, 5))

sns.histplot(df["Price"], kde=True)

plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")

plt.savefig("price_distribution.png")
plt.show()


# 6. Area vs Price
plt.figure(figsize=(10, 5))

sns.scatterplot(data=df, x="Area", y="Price")

plt.title("Area vs House Price")
plt.savefig("area_price_pattern.png")
plt.show()


# 7. Price by Location
plt.figure(figsize=(9, 5))

sns.boxplot(data=df, x="Location", y="Price")

plt.title("House Price by Location")
plt.savefig("location_price_pattern.png")
plt.show()


print("\nPrice Pattern Analysis Completed!")
