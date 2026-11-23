
# Step 1: Import Libraries
import joblib
import pandas as pd

# Step 2: Load Trained Model
model = joblib.load("house_price_model.pkl")

print("===================================")
print("      HOUSE PRICE PREDICTION")
print("===================================")

# Step 3: Take Input From User

area = float(input("Enter House Area: "))

bedrooms = int(input("Enter Number of Bedrooms: "))

bathrooms = int(input("Enter Number of Bathrooms: "))

floors = int(input("Enter Number of Floors: "))

year_built = int(input("Enter Year Built: "))

location = input(
    "Enter Location (Urban/Rural/Downtown/Suburban): "
)

condition = input(
    "Enter Condition (Excellent/Good/Fair/Poor): "
)

garage = input("Garage Available? (Yes/No): ")

# Step 4: Create Input DataFrame

house = pd.DataFrame([{
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Floors": floors,
    "YearBuilt": year_built,
    "Location": location,
    "Condition": condition,
    "Garage": garage
}])

# Step 5: Predict House Price

predicted_price = model.predict(house)

# Step 6: Display Result

print("\n===================================")
print("       HOUSE PRICE RESULT")
print("===================================")

print(
    "Estimated House Price:",
    round(predicted_price[0], 2)
)

print("===================================")
