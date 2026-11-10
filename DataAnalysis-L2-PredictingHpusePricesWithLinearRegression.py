
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

print("\nDataset size:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

# Remove duplicate rows
df = df.drop_duplicates()

# Convert price into numbers
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Remove rows without a valid price
df = df.dropna(subset=["Price"])

# Fill missing values
num_cols = df.select_dtypes(include="number").columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

text_cols = df.select_dtypes(include=["object", "str"]).columns

for col in text_cols:
    if df[col].notna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nDataset cleaned successfully!")
print("Total houses:", len(df))

# Show price distribution
plt.figure(figsize=(7, 4))
plt.hist(df["Price"], bins=30, edgecolor="black")
plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Prices")
plt.tight_layout()
plt.show()

# Select features and target
x = df.drop(columns=["Price", "Id"], errors="ignore")
y = df["Price"]

print("\nFeatures used:")
print(x.columns.tolist())

# Check correlation before encoding
print("\nCorrelation with House Price:")
print(df.corr(numeric_only=True)["Price"].sort_values(ascending=False))

# Convert text columns into numbers
x = pd.get_dummies(x, drop_first=True, dtype=int)

# Correlation heatmap
corr_data = pd.concat([x, y], axis=1)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_data.corr(), cmap="coolwarm", annot=False)
plt.title("House Price Correlation Heatmap")
plt.tight_layout()
plt.show()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("\nTraining houses:", len(x_train))
print("Testing houses:", len(x_test))

# Train Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)

# Predict prices
y_pred = model.predict(x_test)

# Calculate results
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nHOUSE PRICE PREDICTION RESULTS")
print("\nLinear Regression Performance")
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2 * 100, 2), "%")

if r2 >= 0.75:
    print("Model Performance: Good")
elif r2 >= 0.50:
    print("Model Performance: Average")
elif r2 >= 0:
    print("Model Performance: Weak")
else:
    print("Model Performance: Needs Improvement")

# Compare actual and predicted prices
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.round(2)
})

result["Difference"] = (
    result["Actual Price"] - result["Predicted Price"]
).round(2)

result["Error Percentage"] = (
    abs(result["Difference"]) /
    result["Actual Price"].replace(0, np.nan) * 100
).round(2)

print("\nActual and Predicted Prices:")
print(result.head(10).to_string(index=False))

avg_error = result["Error Percentage"].mean()
print("\nAverage Prediction Error:", round(avg_error, 2), "%")

# Actual price vs predicted price
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Prediction Error")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

# Show important features
features = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": model.coef_
})

features = features.sort_values(
    by="Coefficient", ascending=False
)

print("\nFeatures with Highest Positive Coefficients:")
print(features.head(5).to_string(index=False))

print("\nFeatures with Highest Negative Coefficients:")
print(features.tail(5).sort_values(
    by="Coefficient"
).to_string(index=False))

# Ridge Regression comparison
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)

ridge_pred = ridge_model.predict(x_test)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\nRidge Regression Results")
print("R2 Score:", round(ridge_r2 * 100, 2), "%")

if ridge_r2 > r2:
    print("Ridge Regression performed better.")
elif ridge_r2 < r2:
    print("Linear Regression performed better.")
else:
    print("Both models performed equally.")

# Predict a new house price
print("\nPREDICT THE PRICE OF A NEW HOUSE")

area = float(input("Enter house area: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
floors = int(input("Enter number of floors: "))
year = int(input("Enter year built: "))
location = input("Enter location: ").strip()
condition = input("Enter condition: ").strip()
garage = input("Garage available? (Yes/No): ").strip()

new_house = pd.DataFrame([{
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Floors": floors,
    "YearBuilt": year,
    "Location": location,
    "Condition": condition,
    "Garage": garage
}])

# Encode the new house using the same feature columns
new_house = pd.get_dummies(new_house, dtype=int)
new_house = new_house.reindex(columns=x.columns, fill_value=0)

# Predict price
price = model.predict(new_house)[0]

print("\nHOUSE PRICE ESTIMATION")

if price < 0:
    print("The model predicted an unrealistic negative price.")
    print("Please check the dataset and model performance.")
else:
    print("Predicted House Price:", round(price, 2))

print("This is an estimated price, not a guaranteed market value.")
