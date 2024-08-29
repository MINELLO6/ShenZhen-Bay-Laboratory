import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Step 1: Data Preparation
# Load the data

x_data = pd.read_csv('coordinate_31/'+'Cst3.txt', sep='\t', header=0) #随机抽到最好的一组：Camk2n1 vs Dclk1
y_data = pd.read_csv('coordinate_31/'+'Apoe.txt', sep='\t', header=0)

# Rename columns for clarity
x_data.columns = ['x', 'y', 'x_expression']
y_data.columns = ['x', 'y', 'y_expression']


# Step 2: Data Preprocessing
# Merge the datasets on 'x' and 'y'
merged_data = pd.merge(y_data, x_data, on=['x', 'y'])

# Ensure all data is numeric
merged_data = merged_data.apply(pd.to_numeric)

# Prepare for 10-fold cross-validation
coords = merged_data[['x', 'y']].values
X = merged_data[['x_expression']].values
y = merged_data[['y_expression']].values

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Arrays to store results
predictions = np.zeros_like(y)
intercepts = np.zeros(y.shape[0])
slopes = np.zeros(y.shape[0])

# 10-fold cross-validation loop
for train_index, test_index in kf.split(X):
    coords_train, coords_test = coords[train_index], coords[test_index]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Select optimal bandwidth
    bw = Sel_BW(coords_train, y_train, X_train).search()

    # Fit the GWR model
    gwr_model = GWR(coords_train, y_train, X_train, bw)
    gwr_results = gwr_model.fit()

    # Predict on test set
    gwr_pred = gwr_model.predict(coords_test, X_test)

    # Store predictions and parameters
    predictions[test_index] = gwr_pred.predictions
    intercepts[test_index] = gwr_pred.params[:, 0]
    slopes[test_index] = gwr_pred.params[:, 1]

# Calculate residuals
residuals = y.flatten() - predictions.flatten()
residuals = abs(residuals)

# Create results dataframe
results_df = pd.DataFrame({
    'x': merged_data['x'].values,
    'y': merged_data['y'].values,
    'intercept': intercepts.flatten(),
    'slope': slopes.flatten(),
    'predicted': predictions.flatten(),
    'residual': residuals.flatten()
})

# Save the results to a CSV file
#results_df.to_csv('gwr_kfold_results.csv', index=False)

# Calculate Pearson Correlation Coefficient (PCC)
pcc, p_value = pearsonr(results_df['predicted'], merged_data['y_expression'])
print(f'Pearson Correlation Coefficient: {pcc}')
print(f'p-value: {p_value}')

# Calculate additional evaluation metrics
mae = mean_absolute_error(merged_data['y_expression'], results_df['predicted'])
mse = mean_squared_error(merged_data['y_expression'], results_df['predicted'])
rmse = np.sqrt(mse)
r2 = r2_score(merged_data['y_expression'], results_df['predicted'])
n = len(merged_data)
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R²: {r2}')
print(f'Adjusted R²: {adj_r2}')

# Results Interpretation and Visualization
gdf = gpd.GeoDataFrame(results_df, geometry=gpd.points_from_xy(results_df.x, results_df.y))
gdf['x_expression'] = merged_data['x_expression'].values
gdf['y_expression'] = merged_data['y_expression'].values
gdf['predicted'] = results_df['predicted'].values

# Calculate the global vmin and vmax for consistent color scaling
vmin = min(gdf['x_expression'].min(), gdf['y_expression'].min(), gdf['predicted'].min())
vmax = max(gdf['x_expression'].max(), gdf['y_expression'].max(), gdf['predicted'].max())

# Define a threshold for clipping
lower_percentile = 5
upper_percentile = 95

vmin = np.percentile(gdf[['x_expression', 'y_expression', 'predicted']], lower_percentile)
vmax = np.percentile(gdf[['x_expression', 'y_expression', 'predicted']], upper_percentile)


# Plot the Y expression
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
gdf['y_expression'] = merged_data['y_expression']
gdf.plot(column='y_expression', cmap='coolwarm', legend=True, ax=ax, vmin=vmin, vmax=vmax)
plt.title('Spatial Distribution of Y Expression')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()


# Plot the X expression
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
gdf['x_expression'] = merged_data['x_expression']
gdf.plot(column='x_expression', cmap='coolwarm', legend=True, ax=ax)
plt.title('Spatial Distribution of X Expression')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Plot the intercept
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
gdf.plot(column='intercept', cmap='coolwarm', legend=True, ax=ax)
plt.title('Spatial Distribution of Intercept')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Plot the slope
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
gdf.plot(column='slope', cmap='coolwarm', legend=True, ax=ax)
plt.title('Spatial Distribution of Slope')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Plot the predicted values
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
gdf.plot(column='predicted', cmap='coolwarm', legend=True, ax=ax, vmin=vmin, vmax=vmax)
plt.title('Spatial Distribution of Predicted Y Expression')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Plot the residuals
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
gdf.plot(column='residual', cmap='coolwarm', legend=True, ax=ax)
plt.title('Spatial Distribution of Residuals')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
