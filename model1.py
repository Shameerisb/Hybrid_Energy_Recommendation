import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error,r2_score

from lightgbm import LGBMRegressor


dt = pd.read_csv(r"combined_data(copied).csv", low_memory=False)



dt['ghi_rsi'] = dt['ghi_rsi'].fillna(dt['ghi_rsi'].mean())
dt['dhi'] = dt['dhi'].fillna(dt['dhi'].mean())
dt['dni'] = dt['dni'].fillna(dt['dni'].mean())

# Define constants
eta_rated = 0.18  # Rated efficiency at STC (18%)
beta = -0.004     # Temperature coefficient
T_stc = 25        # Standard temperature (°C)
delta_humidity = 0.002  # Humidity impact coefficient
gamma = 0.01      # Wind cooling coefficient

# Effective irradiance
dt['Effective_Irradiance'] = dt['ghi_pyr'] + dt['dni'] + dt['dhi']  # You can add RSI if needed

# Adjusted efficiency based on weather factors
dt['Adjusted_Efficiency'] = (
    eta_rated *
    (1 + beta * (dt['air_temperature'] - T_stc)) *
    (1 - delta_humidity * dt['relative_humidity']) *
    (1 + gamma * dt['wind_speed'])
)

# Calculate solar power
dt['P_solar'] = dt['Adjusted_Efficiency']* dt['Effective_Irradiance']

# View results
print(dt[['ghi_pyr', 'dni', 'dhi', 'air_temperature', 'relative_humidity', 'wind_speed', 'P_solar']].head())


# Save the updated DataFrame with two-decimal formatting for float columns
dt.to_csv('updated_dataset.csv', index=False, float_format='%.2f')




