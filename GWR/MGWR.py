import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os
from tqdm import tqdm


def load_data_from_folder(folder_path):
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, sep='\t', header=0)
            var_name = os.path.splitext(file_name)[0]
            data.columns = ['x', 'y', var_name]
            data_list.append(data)

    merged_data = data_list[0]
    for data in data_list[1:]:
        merged_data = merged_data.merge(data, on=['x', 'y'])

    merged_data = merged_data.apply(pd.to_numeric)
    return merged_data


def convert_coords(df):
    coords = list(zip(df['x'], df['y']))
    return coords


def select_features_random_forest(X, y, num_features=3):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y.ravel())
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(num_features).index.tolist()

    # Create a DataFrame to show feature importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return top_features, feature_importance_df


def fit_mgwr(coords, X, y):
    print("Fitting MGWR model...")
    print(f"Coordinates shape: {len(coords)}, X shape: {X.shape}, y shape: {y.shape}")
    selector = Sel_BW(coords, y, X, multi=True, constant=True)
    bw = selector.search(multi_bw_min=[2])
    print(f"Bandwidth: {bw}")

    mgwr_model = MGWR(coords, y, X, selector, constant=True)
    mgwr_results = mgwr_model.fit()
    return mgwr_results


def calculate_metrics(y_true, y_pred, X):
    pcc, p_value = pearsonr(y_pred, y_true)
    print(f'Pearson Correlation Coefficient: {pcc}')
    print(f'p-value: {p_value}')

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R²: {r2}')
    print(f'Adjusted R²: {adj_r2}')

    return {
        'pcc': pcc,
        'p_value': p_value,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'adj_r2': adj_r2
    }


def visualize_results(merged_data, results_df):
    gdf = gpd.GeoDataFrame(results_df, geometry=gpd.points_from_xy(results_df.x, results_df.y))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf['Nrgn'] = merged_data['Nrgn']
    gdf.plot(column='Nrgn', cmap='coolwarm', legend=True, ax=ax)
    plt.title('Spatial Distribution of Nrgn')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf.plot(column='intercept', cmap='coolwarm', legend=True, ax=ax)
    plt.title('Spatial Distribution of Intercept')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    for i in range(X.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        gdf.plot(column=f'slope_{i + 1}', cmap='coolwarm', legend=True, ax=ax)
        plt.title(f'Spatial Distribution of Slope {i + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf.plot(column='predicted', cmap='coolwarm', legend=True, ax=ax)
    plt.title('Spatial Distribution of Predicted Nrgn')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf.plot(column='residual', cmap='coolwarm', legend=True, ax=ax)
    plt.title('Spatial Distribution of Residuals')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


if __name__ == "__main__":
    folder_path = 'coordinate_31'
    data = load_data_from_folder(folder_path)
    coords = convert_coords(data)

    # Ensure X and y are formatted correctly
    X = data.drop(columns=['x', 'y', 'Nrgn'])
    y = data[['Nrgn']].values  # Ensure y is 2D
    print(X)
    print(y)

    selected_features, feature_importance_df = select_features_random_forest(X, y, num_features=3)
    print(f"Selected features: {selected_features}")
    print("Feature Importance Scores:")
    print(feature_importance_df)

    X_selected = data[selected_features].values  # Select features based on importance

    mgwr_results = fit_mgwr(coords, X_selected, y)
    calculate_metrics(data['Nrgn'].values.flatten(), mgwr_results.predy.flatten(), X_selected)

    results_df = pd.DataFrame({
        'x': data['x'],
        'y': data['y'],
        'Nrgn': data['Nrgn'],
        'intercept': mgwr_results.params[:, 0],
        'predicted': mgwr_results.predy.flatten(),
        'residual': (data['Nrgn'] - mgwr_results.predy.flatten()).values
    })

    for i, feature in enumerate(selected_features, 1):
        results_df[f'slope_{i}'] = mgwr_results.params[:, i]

    results_df.to_csv('mgwr_results.csv', index=False)
    visualize_results(data, results_df)
