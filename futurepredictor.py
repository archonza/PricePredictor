import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Assuming your dataset has a 'dt' (datetime) and 'y' (target variable) column
historical_data = pd.read_csv('./resources/bitcoin_csv.csv')

# Perform data preprocessing, feature engineering, and labeling
# Remove rows with missing data
historical_data = historical_data.dropna()

historical_data['ds'] = pd.to_datetime(historical_data['date'])
historical_data['y'] = historical_data['price(USD)']  # Replace 'price' with your actual target variable column

# Additional features
#historical_data['txVolume(USD)'] = historical_data['txVolume(USD)']
#historical_data['marketcap(USD)'] = historical_data['marketcap(USD)']
#historical_data['generatedCoins'] = historical_data['generatedCoins']

# Drop unnecessary columns
historical_data = historical_data.drop(['date', 'price(USD)'], axis=1)

# Create a Prophet model
model = Prophet()

# Add additional regressors (features) to the model
#model.add_regressor(historical_data['txVolume(USD)'])
#model.add_regressor(historical_data['marketcap(USD)'])
#model.add_regressor('generatedCoins')

# Fit the model to historical data
model.fit(historical_data)

# Create a dataframe with future dates for prediction
future = model.make_future_dataframe(periods=365, include_history=True)  # Adjust the number of future periods as needed

# Add future values of additional regressors
# For example, if you have future values for 'txVolume(USD)', 'marketcap(USD)', and 'generatedCoins'
# future['txVolume(USD)'] = ...
# future['marketcap(USD)'] = ...
# future['generatedCoins'] = ...

# Make predictions for future dates
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)

# Customize the color of historical data points
fig.get_axes()[0].get_lines()[0].set_color('blue')  # Change 'blue' to your desired color

# Customize the color of forecasted values
fig.get_axes()[0].get_lines()[1].set_color('green')  # Change 'green' to your desired color

# Customize x-axis ticks to display months and years
ax = fig.get_axes()[0]
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Display the plot
plt.show()
