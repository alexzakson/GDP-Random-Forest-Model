import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Load and explore data
data = pd.read_csv('gdp_data.csv')
print(data.info())
print(data.describe())

sns.pairplot(data)
plt.show()

# Prepare data
X = data.drop(['GDP'], axis=1)
y = data['GDP']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

# Manual inputs to make predictions with new data
new_data = pd.DataFrame({
    #'VAR_1': [data],
    #'VAR_2': [data],
    #'VAR_3': [data]
    # ... 
})

new_data_scaled = scaler.transform(new_data)
predictions = best_model.predict(new_data_scaled)
print("GDP Predictions:", predictions)


# Generate predictions for the entire dataset
y_all_pred = best_model.predict(X_scaled)

# Plot the model's performance
plt.figure(figsize=(10, 6))

# Plot actual GDP values
plt.plot(y, color='green', label='Actual GDP', linestyle='-', linewidth=4, marker=',')

# Plot predicted GDP values
plt.plot(y_all_pred, color='red', label='Predicted GDP', linestyle='-', linewidth=2, marker=',')

# Add legend and labels
plt.xlabel('')
plt.ylabel('GDP')
plt.title('Actual vs. Predicted GDP')
plt.legend()

plt.show()
