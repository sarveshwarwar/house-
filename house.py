import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load your dataset
# Replace 'houses.csv' with your CSV file path
df = pd.read_csv('houses.csv')

# 2. Select features and target variable
# Example feature selection – customize as per your data columns
features = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'garage', 'floors']
target = 'price'

X = df[features]
y = df[target]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 7. Predict price for a new house (example)
new_house = np.array([[2000, 4, 3, 2015, 6000, 1, 2]])  # Example input
predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: ${predicted_price[0]:,.0f}")
