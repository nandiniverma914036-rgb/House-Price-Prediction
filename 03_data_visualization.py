
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Cleaned Dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

# Graph 1: House Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.show()

# Graph 2: Area vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Area", y="Price")
plt.title("Area vs House Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# Graph 3: Average Price by Location
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Location", y="Price")
plt.title("Average House Price by Location")
plt.xticks(rotation=45)
plt.show()

print("Data Visualization Completed!")
