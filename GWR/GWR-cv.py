import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

def load_data():
    abat_data = pd.read_csv('Abat.txt', sep='\t', header=0)
    aak1_data = pd.read_csv('Aak1.txt', sep='\t', header=0)
    gene3_data = pd.read_csv('Aco2.txt', sep='\t', header=0)
    gene4_data = pd.read_csv('Acot7.txt', sep='\t', header=0)

    abat_data.columns = ['x', 'y', 'abat_expression']
    aak1_data.columns = ['x', 'y', 'aak1_expression']
    gene3_data.columns = ['x', 'y', 'Aco2_expression']
    gene4_data.columns = ['x', 'y', 'Acot7_expression']

    merged_data = abat_data.merge(aak1_data, on=['x', 'y']).merge(gene3_data, on=['x', 'y']).merge(gene4_data, on=['x', 'y'])
    merged_data = merged_data.apply(pd.to_numeric)
    print(merged_data)

    return merged_data

def convert_coords(df):
    coords = list(zip(df['x'], df['y']))
    return coords

def fit_mgwr(coords, X, y):
    print("Fitting MGWR model...")
    print(f"Coordinates shape: {len(coords)}, X shape: {X.shape}, y shape: {y.shape}")
    selector = Sel_BW(coords, y, X, multi=True, constant=True)
    max_bw = len(coords) - 1  # Set maximum bandwidth to one less than the number of data points
    bw = selector.search(multi_bw_min=[2], multi_bw_max=[max_bw])
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

def perform_kfold_cv(merged_data, coords, X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}:")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coords_train, coords_test = [coords[i] for i in train_index], [coords[i] for i in test_index]

        print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        print(f"Training coordinates: {len(coords_train)}, Test coordinates: {len(coords_test)}")

        try:
            mgwr_results = fit_mgwr(coords_train, X_train, y_train)
            y_pred_train = mgwr_results.predy.flatten()

            # Predict for the test set
            mgwr_test = MGWR(coords_test, y_test, X_test, mgwr_results.selector, constant=True)
            mgwr_test_results = mgwr_test.fit()
            y_pred_test = mgwr_test_results.predy.flatten()

            print(metrics)
            print(len(metrics))

            metrics.append(calculate_metrics(y_test.flatten(), y_pred_test, X_train))
        except Exception as e:
            print(f"Exception: {e}, skipping this fold.")
            continue

    if metrics:
        avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
        print("Average metrics over 10-fold cross-validation:")
        for key, value in avg_metrics.items():
            print(f'{key}: {value}')

def visualize_results(merged_data, results_df):
    gdf = gpd.GeoDataFrame(results_df, geometry=gpd.points_from_xy(results_df.x, results_df.y))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf['abat_expression'] = merged_data['abat_expression']
    gdf.plot(column='abat_expression', cmap='coolwarm', legend=True, ax=ax)
    plt.title('Spatial Distribution of ABAT Expression')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf['aak1_expression'] = merged_data['aak1_expression']
    gdf.plot(column='aak1_expression', cmap='coolwarm', legend=True, ax=ax)
    plt.title('Spatial Distribution of AAK1 Expression')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    for i, gene in enumerate(['Aco2_expression', 'Acot7_expression']):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        gdf[gene] = merged_data[gene]
        gdf.plot(column=gene, cmap='coolwarm', legend=True, ax=ax)
        plt.title(f'Spatial Distribution of {gene}')
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
    plt.title('Spatial Distribution of Predicted Abat Expression')
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
    merged_data = load_data()
    coords = convert_coords(merged_data)
    X = merged_data[['aak1_expression', 'Aco2_expression', 'Acot7_expression']].values
    y = merged_data[['abat_expression']].values  # Ensure y is 2D
    perform_kfold_cv(merged_data, coords, X, y)
    mgwr_results = fit_mgwr(coords, X, y)
    calculate_metrics(merged_data['abat_expression'].values.flatten(), mgwr_results.predy.flatten(), X)
    visualize_results(merged_data, pd.read_csv('mgwr_results.csv'))
