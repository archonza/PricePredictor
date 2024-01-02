import pandas as pd
from fbprophet import Prophet

# Assuming your dataset has a 'ds' (datetime) and 'y' (target variable) column
historical_data = pd.read_csv('./resources/bitcoin_csv.csv')

# Create a Prophet model
model = Prophet()

# Fit the model to historical data
model.fit(historical_data)

# Create a dataframe with future dates for prediction
future = model.make_future_dataframe(periods=365)  # Adjust the number of future periods as needed

# Make predictions for future dates
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
