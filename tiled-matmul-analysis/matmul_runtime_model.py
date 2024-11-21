import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the data from CSV
data_file = "profiling/execution_times_all.csv"  # Update with your actual CSV file path
data = pd.read_csv(data_file)

# Separate features (X) and target (y)
X = data[['M', 'N', 'K', 'Tile Size']]
y = data['Execution Time']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Performance for Runtime Prediction:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Display coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nModel Coefficients:")
print(coefficients)

# Test the model with a sample input
sample_input = np.array([[1024, 1024, 1024, 16]])  # Replace with your values for M, N, K, Tile Size
predicted_runtime = model.predict(sample_input)
print(f"\nPredicted Execution Time for {sample_input[0]}: {predicted_runtime[0]:.2f} ms")
