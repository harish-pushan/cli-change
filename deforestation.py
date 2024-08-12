import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pycountry_convert as pc

# Load the data from the CSV file
file_path = 'deforestation.csv'
df = pd.read_csv(file_path)

# Remove rows with missing values
df_clean = df.dropna()

# Function to convert ISO3 country codes to continents
def iso3_to_continent(iso3c):
    try:
        country_alpha2 = pc.country_alpha3_to_country_alpha2(iso3c)
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except:
        return 'Unknown'

# Apply the conversion function to the dataset
df_clean['continent'] = df_clean['iso3c'].apply(iso3_to_continent)

# Feature Engineering: Calculate annual forest cover change rate (2000-2020)
df_clean['annual_change_rate'] = (df_clean['forests_2020'] - df_clean['forests_2000']) / 20

# Prepare data for modeling: Create a DataFrame for historical and future predictions
years = np.arange(2000, 2045)
historical_years = np.arange(2000, 2021)
predicted_years = np.arange(2021, 2045)
historical_predictions = []
future_predictions = []

for _, row in df_clean.iterrows():
    forest_2000 = row['forests_2000']
    annual_change_rate = row['annual_change_rate']
    
    historical_forest_cover = [forest_2000 + annual_change_rate * (year - 2000) for year in historical_years]
    future_forest_cover = [historical_forest_cover[-1] + annual_change_rate * (year - 2020) for year in predicted_years]
    
    historical_predictions.append(historical_forest_cover)
    future_predictions.append(future_forest_cover)

# Convert predictions into DataFrames
historical_df = pd.DataFrame(historical_predictions, columns=historical_years)
historical_df['continent'] = df_clean['continent']

future_df = pd.DataFrame(future_predictions, columns=predicted_years)
future_df['continent'] = df_clean['continent']

# Group by continent and calculate the mean deforestation trends
historical_means = historical_df.groupby('continent').mean()
future_means = future_df.groupby('continent').mean()

# Plot the historical and predicted data by continent
plt.figure(figsize=(10, 6))
for continent in historical_means.index:
    plt.plot(historical_years, historical_means.loc[continent], label=f'{continent} (Historical)', linestyle='-')
    plt.plot(predicted_years, future_means.loc[continent], label=f'{continent} (Predicted)', linestyle='--')

plt.title('Historical and Predicted Deforestation Trends by Continent (2000-2044)')
plt.xlabel('Year')
plt.ylabel('Average Forest Cover (%)')
plt.legend(title='Continent', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
