#!/usr/bin/env python
"""
Quick test for basic cleaning without W&B
"""
import pandas as pd

# Load the local sample data
df = pd.read_csv('../../components/get_data/data/sample1.csv')

print(f"Original data shape: {df.shape}")

# Apply basic cleaning
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

# Save the cleaned file
df.to_csv('clean_sample.csv', index=False)
print("Cleaned data saved to clean_sample.csv")
