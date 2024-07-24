import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the egg price dataset
df = pd.read_csv('egg_price.csv')

# Convert the 'date' column to a datetime object
df['date'] = pd.to_datetime(df['date'])

# Extract the month and year from the date
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Visualize the data
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='price', data=df, label='Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Egg Price Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Prepare the data for modeling
X = df[['month', 'year']]
y = df['price']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot the actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.show()

# Create a list of months and years for future predictions
start_date = pd.to_datetime('2023-11-01')
end_date = pd.to_datetime('2028-10-01')
date_range = pd.date_range(start_date, end_date, freq='MS')

# Make predictions for each month
future_predictions = []
for date in date_range:
    month = date.month
    year = date.year
    prediction = model.predict([[month, year]])
    future_predictions.append((date, prediction[0]))
    print(f"Year: {year}, Month: {month}, Prediction: {prediction[0]}")

# Prepare the data for plotting
future_dates = [date for date, _ in future_predictions]
future_prices = [price for _, price in future_predictions]

# Combine the original dataset with prediction results
combined_dates = df['date'].tolist() + future_dates
combined_prices = df['price'].tolist() + future_prices

# Create a time series plot with future predictions
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['price'], label='Actual Prices')
plt.plot(future_dates, future_prices, 'r--', label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Egg Price Time Series with Future Predictions')
plt.legend()
plt.grid(True)
plt.show()

# Save the predictions to a CSV file
future_df = pd.DataFrame(future_predictions, columns=['date', 'predicted_price'])
future_df.to_csv('egg_price_predictions.csv', index=False)
