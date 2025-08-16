#!/usr/bin/env python
"""
Local model training script
"""
import pandas as pd
import numpy as np
import wandb
import json
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import mlflow.sklearn

def delta_date_feature(dates):
    """
    Given a 2d array containing dates, returns delta in days between each date and the most recent date
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()

def get_inference_pipeline(rf_config, max_tfidf_features):
    # Categorical features
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    
    ordinal_categorical_preproc = OrdinalEncoder()
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )
    
    # Numerical features
    zero_imputed = [
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365", "longitude", "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    
    # Date feature engineering
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )
    
    # Text feature (name)
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(binary=False, max_features=max_tfidf_features, stop_words='english'),
    )
    
    # Combine all preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",
    )
    
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]
    
    # Create Random Forest
    random_forest = RandomForestRegressor(**rf_config)
    
    # Create pipeline
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_forest)
        ]
    )
    
    return sk_pipe, processed_features

def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest"].feature_importances_[:len(feat_names)-1]
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp

def main():
    # Initialize W&B
    run = wandb.init(project="nyc_airbnb", group="train_model", job_type="train")
    
    # Load trainval data (use local split)
    if os.path.exists("trainval_data.csv"):
        X = pd.read_csv("trainval_data.csv")
    else:
        # Fallback: create splits locally
        df = pd.read_csv('components/get_data/data/sample1.csv')
        
        # Apply cleaning
        idx = df['price'].between(10, 350)
        df = df[idx].copy()
        df['last_review'] = pd.to_datetime(df['last_review'])
        idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
        df = df[idx].copy()
        
        # Split
        trainval, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df['neighbourhood_group'])
        X = trainval
    
    y = X.pop("price")
    
    print(f"Training data shape: {X.shape}")
    print(f"Price range: ${y.min():.2f} - ${y.max():.2f}")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=X['neighbourhood_group'], random_state=42
    )
    
    # RF config
    rf_config = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 4,
        'min_samples_leaf': 3,
        'n_jobs': -1,
        'criterion': 'squared_error',
        'max_features': 0.5,
        'oob_score': True,
        'random_state': 42
    }
    
    # Create pipeline
    print("Building pipeline...")
    sk_pipe, processed_features = get_inference_pipeline(rf_config, max_tfidf_features=5)
    
    # Train model
    print("Training model...")
    sk_pipe.fit(X_train, y_train)
    
    # Evaluate
    r_squared = sk_pipe.score(X_val, y_val)
    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"R² Score: {r_squared:.4f}")
    print(f"MAE: ${mae:.2f}")
    
    # Save model
    print("Saving model...")
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")
    
    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        input_example=X_train.iloc[:5]
    )
    
    # Create W&B artifact
    artifact = wandb.Artifact(
        "random_forest_export",
        type='model_export',
        description='Trained random forest model',
        metadata=rf_config
    )
    artifact.add_dir('random_forest_dir')
    run.log_artifact(artifact)
    
    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)
    
    # Log metrics
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae
    
    # Log visualization
    run.log({"feature_importance": wandb.Image(fig_feat_imp)})
    
    print("Training completed successfully!")
    run.finish()

if __name__ == "__main__":
    main()
