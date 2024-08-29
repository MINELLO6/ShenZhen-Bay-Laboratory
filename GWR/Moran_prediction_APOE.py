import pandas as pd
import numpy as np
import itertools
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import warnings

# Ignore warnings that may not affect results
warnings.filterwarnings("ignore")

# Read Moran Index data
file_path = 'gene_moran_values_31_2.csv'
data = pd.read_csv(file_path)

# Filter genes based on Moran Index and exclude APOE
filtered_genes = data[(data['Moran Index'] > -1) & (data['File Name'] != 'Apoe.txt')]

# Get list of gene files (excluding APOE)
gene_files = filtered_genes['File Name'].tolist()

# Load APOE gene data
apoe_data = pd.read_csv('coordinate_31/APOE.txt', sep='\t', header=0)
apoe_data.columns = ['x', 'y', 'apoe_expression']

# Initialize results list
results = []

# For each gene, use it to predict APOE expression
for gene in tqdm(gene_files, total=len(gene_files)):
    try:
        # Load the gene's expression data
        gene_data = pd.read_csv(f'coordinate_31/{gene}', sep='\t', header=0)
        gene_data.columns = ['x', 'y', 'gene_expression']
        
        # Merge with APOE data on coordinates
        merged_data = pd.merge(gene_data, apoe_data, on=['x', 'y'])
        merged_data = merged_data.apply(pd.to_numeric)
        
        # Extract coordinates and expression data
        coords = merged_data[['x', 'y']].values
        X = merged_data[['gene_expression']].values
        y = merged_data[['apoe_expression']].values
        
        # Check for multicollinearity using the condition number
        condition_number = np.linalg.cond(X)
        
        # Set a threshold for the condition number to detect near-singular matrices
        if condition_number > 1e8:
            print(f"Skipping gene: {gene} due to high condition number (possible singular matrix).")
            continue
        
        # Perform additional checks on matrix properties
        eigvals = np.linalg.eigvals(np.dot(X.T, X))
        if np.any(eigvals < 1e-10):
            print(f"Skipping gene: {gene} due to near-zero eigenvalues (possible singular matrix).")
            continue
        
        # 10-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        predictions = np.zeros_like(y)
        
        for train_index, test_index in kf.split(X):
            coords_train, coords_test = coords[train_index], coords[test_index]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Select best bandwidth
            bw = Sel_BW(coords_train, y_train, X_train).search()
            
            # Fit GWR model
            gwr_model = GWR(coords_train, y_train, X_train, bw)
            gwr_results = gwr_model.fit()
            
            # Predict
            gwr_pred = gwr_model.predict(coords_test, X_test)
            predictions[test_index] = gwr_pred.predictions
            
        # Compute evaluation metrics
        pcc, _ = pearsonr(predictions.flatten(), y.flatten())
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mse)
        n = len(merged_data)
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        moran_gene = filtered_genes.loc[filtered_genes['File Name'] == gene, 'Moran Index'].values[0]
        
        # Save result
        results.append({
            'Predictor Gene': gene,
            'Moran Index': moran_gene,
            'PCC': pcc,
            'R^2': r2,
            'Adjusted R²': adj_r2,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        })
    
    except np.linalg.LinAlgError:
        # Handle singular matrix error and skip the current gene
        print(f"Skipping gene: {gene} due to singular matrix error.")
        continue
    except Exception as e:
        # Handle other errors and continue processing other genes
        print(f"Skipping gene: {gene} due to an unexpected error: {e}")
        continue    

# Save all results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('gene_gwr_apoe_results.csv', index=False)

print("GWR results for predicting APOE expression have been saved to gene_gwr_apoe_results.csv")
