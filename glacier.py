import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'glacier.csv'  # Replace this with the correct file path
glacier_data = pd.read_csv(file_path)

# Fill missing values with interpolation method to handle time series data
glacier_data_interpolated = glacier_data.interpolate(method='linear', limit_direction='forward', axis=0)

# Define the mapping of regions to continents
region_to_continent = {
    'ISL': 'Europe',
    'CEU': 'Europe',
    'ALA': 'North America',
    'ASW': 'Asia',
    'SA2': 'South America',
    'WNA': 'North America',
    'ACN': 'Oceania',
    'ACS': 'Oceania',
    'GRL': 'North America',
    'SJM': 'Europe',
    'SCA': 'Europe',
    'RUA': 'Asia',
    'ASN': 'Asia',
    'CAU': 'Asia',
    'ASC': 'Asia',
    'ASE': 'Asia',
    'TRP': 'Oceania',
    'SA1': 'South America',
    'NZL': 'Oceania',
    'ANT': 'Antarctica'
}

# Create a new column for continent and map the regions to their respective continents
continent_data = glacier_data_interpolated.set_index('YEAR')
continent_data.columns = continent_data.columns.map(region_to_continent)

# Aggregate the data by continent
continent_data_aggregated = continent_data.groupby(axis=1, level=0).mean()

# Group data by 5-year intervals
continent_data_aggregated_5year = continent_data_aggregated.groupby((continent_data_aggregated.index // 5) * 5).sum()

# Preparing data for modeling
X = continent_data_aggregated.index.values.reshape(-1, 1)  # Years as the independent variable
predictions = {}

# Create a linear regression model for each continent and predict future values
future_years = np.arange(2024, 2045).reshape(-1, 1)

for continent in continent_data_aggregated.columns:
    y = continent_data_aggregated[continent].dropna().values  # Values for the continent as dependent variable
    
    # Train the model
    model = LinearRegression()
    model.fit(X[:len(y)], y)  # Fit only on available data
    
    # Make predictions
    predictions[continent] = model.predict(future_years)

# Convert predictions to a DataFrame for better visualization
predictions_df = pd.DataFrame(predictions, index=future_years.flatten())

# Group predicted data by 5-year intervals
predictions_df_5year = predictions_df.groupby((predictions_df.index // 5) * 5).sum()

# Combine historical and predicted data
combined_data_5year = pd.concat([continent_data_aggregated_5year, predictions_df_5year])

# Sum historical data and predicted data by 5-year intervals
historical_sum_5year = continent_data_aggregated_5year.sum(axis=1)
predicted_sum_5year = predictions_df_5year.sum(axis=1)

# Plot the data as a line graph
fig, ax = plt.subplots(figsize=(14, 8))

# Plot historical data
ax.plot(historical_sum_5year.index, historical_sum_5year, label='Historical Data', marker='o')

# Plot predicted data
ax.plot(predicted_sum_5year.index, predicted_sum_5year, label='Predicted Data', marker='o', linestyle='--')

plt.title("Aggregated Glacier Retreat (Historical vs Predicted) by 5-Year Intervals (1915-2040)")
plt.xlabel("5-Year Interval")
plt.ylabel("Aggregated Glacier Retreat Index")
plt.legend()
plt.grid(True)
plt.show()
