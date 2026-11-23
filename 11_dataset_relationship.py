
import pandas as pd

# Load Cleaned Dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

print("========== NUMERIC CORRELATION ==========")

numeric_columns = [
    "Area", "Bedrooms", "Bathrooms",
    "Floors", "YearBuilt", "Price"
]

print(df[numeric_columns].corr(method="pearson")["Price"].sort_values(ascending=False))

print("\n========== SPEARMAN CORRELATION ==========")

print(df[numeric_columns].corr(method="spearman")["Price"].sort_values(ascending=False))


print("\n========== CATEGORICAL FEATURE ANALYSIS ==========")

for column in ["Location", "Condition", "Garage"]:
    print("\nFeature:", column)

    print(
        df.groupby(column)["Price"].agg(
            ["count", "mean", "median", "std"]
        ).round(2)
    )


print("\n========== AREA GROUP ANALYSIS ==========")

df["Area_Group"] = pd.qcut(
    df["Area"],
    q=5,
    duplicates="drop"
)

print(
    df.groupby("Area_Group", observed=True)["Price"].agg(
        ["count", "mean", "median"]
    ).round(2)
)


print("\nDataset Relationship Analysis Completed!")
