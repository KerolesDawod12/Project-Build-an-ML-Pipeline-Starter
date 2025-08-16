#!/usr/bin/env python
"""
Local data validation script
"""
import pandas as pd
import wandb

def test_row_count(data):
    """Test that the dataset has reasonable number of rows"""
    assert 15000 <= len(data) <= 1000000, f"Dataset has {len(data)} rows, expected between 15,000 and 1,000,000"
    print(f"✅ Row count test passed: {len(data)} rows")

def test_price_range(data):
    """Test that prices are within reasonable range"""
    assert data['price'].min() >= 10, f"Minimum price is ${data['price'].min()}, expected >= $10"
    assert data['price'].max() <= 350, f"Maximum price is ${data['price'].max()}, expected <= $350"
    print(f"✅ Price range test passed: ${data['price'].min():.2f} - ${data['price'].max():.2f}")

def test_location_range(data):
    """Test that coordinates are within NYC area"""
    assert data['longitude'].between(-74.25, -73.50).all(), "Some longitude values are outside NYC area"
    assert data['latitude'].between(40.5, 41.2).all(), "Some latitude values are outside NYC area"
    print(f"✅ Location range test passed")

def test_required_columns(data):
    """Test that all required columns exist"""
    required_columns = ['price', 'longitude', 'latitude', 'neighbourhood_group', 'room_type']
    missing_columns = [col for col in required_columns if col not in data.columns]
    assert len(missing_columns) == 0, f"Missing required columns: {missing_columns}"
    print(f"✅ Required columns test passed")

def test_no_null_prices(data):
    """Test that there are no null prices"""
    null_prices = data['price'].isnull().sum()
    assert null_prices == 0, f"Found {null_prices} null prices"
    print(f"✅ No null prices test passed")

def main():
    # Initialize W&B
    run = wandb.init(project="nyc_airbnb", group="data_check", job_type="data_validation")
    
    # Load the data
    if os.path.exists("trainval_data.csv"):
        data = pd.read_csv("trainval_data.csv")
        data_source = "trainval_data.csv"
    elif os.path.exists("test_data.csv"):
        data = pd.read_csv("test_data.csv")
        data_source = "test_data.csv"
    else:
        print("No processed data found, loading raw data...")
        data = pd.read_csv('components/get_data/data/sample1.csv')
        data_source = "sample1.csv"
    
    print(f"Running data validation on: {data_source}")
    print(f"Data shape: {data.shape}")
    
    # Run all tests
    try:
        test_required_columns(data)
        test_row_count(data)
        test_price_range(data)
        test_location_range(data)
        test_no_null_prices(data)
        
        print("\n🎉 All data validation tests passed!")
        
        # Log validation results
        run.summary['data_validation'] = 'passed'
        run.summary['total_rows'] = len(data)
        run.summary['min_price'] = float(data['price'].min())
        run.summary['max_price'] = float(data['price'].max())
        run.summary['avg_price'] = float(data['price'].mean())
        
        print(f"Summary stats logged to W&B:")
        print(f"  - Total rows: {len(data)}")
        print(f"  - Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
        print(f"  - Average price: ${data['price'].mean():.2f}")
        
    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        run.summary['data_validation'] = 'failed'
        run.summary['error_message'] = str(e)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        run.summary['data_validation'] = 'error'
        run.summary['error_message'] = str(e)
    
    run.finish()

if __name__ == "__main__":
    import os
    main()
