#!/usr/bin/env python
"""
Local data splitting script without MLflow complications
"""
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

def main():
    # Initialize W&B
    run = wandb.init(project="nyc_airbnb", group="data_split", job_type="split_data")
    
    # Load the original data and apply cleaning locally
    df = pd.read_csv('components/get_data/data/sample1.csv')
    
    # Apply same cleaning as basic_cleaning
    min_price = 10
    max_price = 350
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    # Filter geolocation outliers
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    print(f"Cleaned data shape: {df.shape}")
    
    # Split into train+val and test
    stratify_col = df['neighbourhood_group'] if 'neighbourhood_group' in df.columns else None
    trainval, test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=stratify_col
    )
    
    print(f"Train+Val shape: {trainval.shape}")
    print(f"Test shape: {test.shape}")
    
    # Save trainval data
    trainval.to_csv("trainval_data.csv", index=False)
    trainval_artifact = wandb.Artifact(
        "trainval_data.csv",
        type="segregated_data",
        description="Train and validation data"
    )
    trainval_artifact.add_file("trainval_data.csv")
    run.log_artifact(trainval_artifact)
    
    # Save test data  
    test.to_csv("test_data.csv", index=False)
    test_artifact = wandb.Artifact(
        "test_data.csv", 
        type="segregated_data",
        description="Test data"
    )
    test_artifact.add_file("test_data.csv")
    run.log_artifact(test_artifact)
    
    print("Data splitting completed successfully!")
    run.finish()

if __name__ == "__main__":
    main()
