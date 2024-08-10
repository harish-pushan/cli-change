import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('sea_tide_level.pkl')

# Load the data
df = pd.read_csv('sea.csv')
df.columns = ['Year', 'GMSL', 'Uncertainty']

# Predict the next 20 years
average_uncertainty = df['Uncertainty'].mean()
future_years = np.array([[year, average_uncertainty] for year in range(2004, 2044)])
future_gmsl = model.predict(future_years)

# Display predictions
predictions = pd.DataFrame({'Year': future_years[:, 0], 'Predicted GMSL': future_gmsl})
print(predictions)

# Visualize the predictions
plt.plot(df['Year'], df['GMSL'], label='Historical GMSL', color='blue')
plt.plot(future_years[:, 0], future_gmsl, label='Predicted GMSL', color='red')
plt.fill_between(future_years[:, 0], future_gmsl - average_uncertainty, future_gmsl + average_uncertainty, color='red', alpha=0.2, label='Uncertainty')
plt.xlabel('Year')
plt.ylabel('GMSL')
plt.title('GMSL Prediction with Uncertainty for the Next 20 Years')
plt.legend()
plt.show()
