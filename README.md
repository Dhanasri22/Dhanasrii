import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\leona\Downloads\Basel_Weather (1).csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop rows with missing values
df = df.dropna()

# Shift target column to predict the next day

df['target_temp'] = df['temp_mean'].shift(-1)
df['target_rain'] = df['precipitation'].shift(-1)

# Drop the last row (because of shift)
df = df.dropna()

# Select features (you can customize this)
features = ['temp_mean', 'temp_min', 'temp_max', 'humidity', 'sunshine', 'precipitation', 'cloud_cover', 'global_radiation']
X = df[features]
y_temp = df['target_temp']
y_rain = df['target_rain']

# Split data
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_rain_train, y_rain_test = train_test_split(X, y_rain, test_size=0.2, random_state=42)

# Train models
model_temp = RandomForestRegressor()
model_rain = RandomForestRegressor()

model_temp.fit(X_train, y_temp_train)
model_rain.fit(X_train, y_rain_train)

# Predict next day based on the last row
latest_data = X.iloc[[-1]]
predicted_temp = model_temp.predict(latest_data)[0]
predicted_rain = model_rain.predict(latest_data)[0]

print(f"🌡️ Predicted temperature for the next day: {predicted_temp:.2f} °C")
print(f"🌧️ Predicted rainfall for the next day: {predicted_rain:.2f} mm")
