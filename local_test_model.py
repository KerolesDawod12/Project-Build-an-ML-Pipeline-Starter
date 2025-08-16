#!/usr/bin/env python
"""
Local model testing script
"""
import pandas as pd
import numpy as np
import wandb
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def main():
    # Initialize W&B
    run = wandb.init(project="nyc_airbnb", group="test_model", job_type="test")
    
    # Load test data
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
    else:
        # Fallback: create test split locally
        df = pd.read_csv('components/get_data/data/sample1.csv')
        
        # Apply cleaning
        idx = df['price'].between(10, 350)
        df = df[idx].copy()
        df['last_review'] = pd.to_datetime(df['last_review'])
        idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
        df = df[idx].copy()
        
        # Split to get test data
        _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['neighbourhood_group'])
    
    # Separate features and target
    X_test = test_df.drop('price', axis=1)
    y_test = test_df['price']
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test price range: ${y_test.min():.2f} - ${y_test.max():.2f}")
    
    # Load the trained model
    if os.path.exists("random_forest_dir"):
        model = mlflow.sklearn.load_model("random_forest_dir")
        print("Model loaded successfully!")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate R² manually
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        print(f"Test R² Score: {r2_score:.4f}")
        print(f"Test MAE: ${mae:.2f}")
        
        # Log test metrics
        run.summary['test_r2'] = r2_score
        run.summary['test_mae'] = mae
        
        # Create some analysis
        residuals = y_test - y_pred
        print(f"Mean residual: ${np.mean(residuals):.2f}")
        print(f"Std residuals: ${np.std(residuals):.2f}")
        
        # Log residual stats
        run.summary['mean_residual'] = np.mean(residuals)
        run.summary['std_residual'] = np.std(residuals)
        
        print("Model testing completed successfully!")
        
    else:
        print("Model not found. Please train the model first.")
        return
    
    run.finish()

if __name__ == "__main__":
    import os
    main()
