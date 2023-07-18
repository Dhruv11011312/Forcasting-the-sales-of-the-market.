# Forcasting-the-sales-of-the-market.
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Sample data (replace this with your actual supermarket sales data)
data = {
    'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'Sales': [100, 110, 120, 115, 125, 130, 140, 135, 145, 150, 160, 155]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Plot the original sales data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Sales'], marker='o')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Supermarket Sales Data')
plt.show()

# ARIMA model parameters 
order = (1, 1, 1)  # (p, d, q)

# Fit ARIMA model
model = ARIMA(df['Sales'], order=order)
results = model.fit()

# Forecasting future sales
forecast_steps = 6
forecast, stderr, conf_int = results.forecast(steps=forecast_steps)


forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, closed='right')

# Createing a DataFrame for the forecasted sales
forecast_df = pd.DataFrame({'Forecasted Sales': forecast}, index=forecast_index[1:])

# Ploting the sales data along with the forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Sales'], marker='o', label='Actual Sales')
plt.plot(forecast_df.index, forecast_df['Forecasted Sales'], marker='o', color='red', label='Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Supermarket Sales Forecast')
plt.legend()
plt.show()
