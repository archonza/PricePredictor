# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load the dataset
# bitcoin_data = pd.read_csv('./resources/bitcoin_csv.csv')  # Replace with your actual dataset file

# # Perform data preprocessing, feature engineering, and labeling
# # Remove rows with missing data
# bitcoin_data = bitcoin_data.dropna()

# bitcoin_data['date'] = pd.to_datetime(bitcoin_data['date'])
# bitcoin_data['year'] = bitcoin_data['date'].dt.year
# bitcoin_data['month'] = bitcoin_data['date'].dt.month
# bitcoin_data['day'] = bitcoin_data['date'].dt.day

# # Drop the original date column
# bitcoin_data = bitcoin_data.drop('date', axis=1)

# # Split the data into features (X) and target variable (y)
# X = bitcoin_data[['year', 'month', 'day', 'price(USD)', 'generatedCoins']]
# y = bitcoin_data['price(USD)']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Choose and train a model (Random Forest as an example)
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Make predictions
# predictions = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f'Model Accuracy: {accuracy}')

# # Make predictions on new, unseen data
# new_data = pd.read_csv('./resources/predicted_bitcoin_csv.csv')  # Replace with your actual new data file
# new_predictions = model.predict(new_data)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
bitcoin_data = pd.read_csv('./resources/bitcoin_csv.csv')  # Replace with your actual dataset file

# Perform data preprocessing, feature engineering, and labeling
# Remove rows with missing data
bitcoin_data = bitcoin_data.dropna()

bitcoin_data['date'] = pd.to_datetime(bitcoin_data['date'])
bitcoin_data['year'] = bitcoin_data['date'].dt.year
bitcoin_data['month'] = bitcoin_data['date'].dt.month
bitcoin_data['day'] = bitcoin_data['date'].dt.day

# Drop the original date column
bitcoin_data = bitcoin_data.drop('date', axis=1)

# Split the data into features (X) and target variable (y)
X = bitcoin_data[['year', 'month', 'day', 'generatedCoins', 'marketcap(USD)', 'txVolume(USD)', 'adjustedTxVolume(USD)', 'txCount','fees', 'activeAddresses', 'averageDifficulty', 'paymentCount', 'medianTxValue(USD)', 'medianFee', 'blockSize', 'blockCount']]
y = bitcoin_data['activeAddresses']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train a model (Random Forest Regressor as an example)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Make predictions on new, unseen data
#new_data = pd.read_csv('./resources/predicted_bitcoin_csv.csv')  # Replace with your actual new data file
#new_predictions = model.predict(new_data)

# Feature Importance
feature_importance = model.feature_importances_
print('Feature Importance:', feature_importance)

# # Visualize Predicted vs. Actual Values with Different Colors
# import matplotlib.pyplot as plt

# plt.scatter(y_test, predictions, color='blue', label='Actual vs. Predicted')
# plt.plot(y_test, y_test, color='red', linestyle='--', label='Perfect Prediction')

# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs. Predicted Values on Test Set')
# plt.legend()
# plt.grid(True)  # Add gridlines
# plt.show()



# # Visualize Predicted vs. Actual Values with Line of Best Fit
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.regplot(x=y_test, y=predictions, scatter_kws={'color':'blue'}, line_kws={'color':'red', 'linestyle':'--'})

# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs. Predicted Values on Test Set with Line of Best Fit')
# plt.grid(True)
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# # Assuming actual and predicted values are in y_test and predictions

# # Calculate the range of values
# value_range = max(max(y_test), max(predictions)) - min(min(y_test), min(predictions))

# # Set the interval for gridlines
# grid_interval = 500  # Adjust this value based on your preference for gridline density

# # Calculate the positions of gridlines
# x_gridlines = np.arange(0, value_range + grid_interval, grid_interval)
# y_gridlines = np.arange(0, value_range + grid_interval, grid_interval)

# # Visualize Predicted vs. Actual Values with Gridlines
# plt.scatter(y_test, predictions, color='blue', label='Actual vs. Predicted')
# plt.plot(y_test, y_test, color='red', linestyle='--', label='Perfect Prediction')

# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs. Predicted Values on Test Set with Gridlines')
# plt.legend()
# plt.grid(True)  # Add gridlines

# # Set the positions of tick marks on the x-axis and y-axis
# plt.xticks(x_gridlines)
# plt.yticks(y_gridlines)

# plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Assuming actual and predicted values are in y_test and predictions

# Calculate the range of values
value_range = max(max(y_test), max(predictions)) - min(min(y_test), min(predictions))

# Set the interval for gridlines
grid_interval = 10000 # Adjust this value based on your preference for gridline density

# Calculate the positions of gridlines
x_gridlines = np.arange(0, value_range + grid_interval, grid_interval)
y_gridlines = np.arange(0, value_range + grid_interval, grid_interval)

# Set the figure size
fig, ax = plt.subplots(figsize=(10, 8))

# Visualize Predicted vs. Actual Values with Gridlines
ax.scatter(y_test, predictions, color='blue', label='Actual vs. Predicted')
ax.plot(y_test, y_test, color='red', linestyle='--', label='Perfect Prediction')

ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Actual vs. Predicted Values on Test Set with Gridlines')
ax.legend()
ax.grid(True)  # Add gridlines

# Set the positions of tick marks on the x-axis and y-axis
ax.set_xticks(x_gridlines)
ax.set_yticks(y_gridlines)

plt.show()
