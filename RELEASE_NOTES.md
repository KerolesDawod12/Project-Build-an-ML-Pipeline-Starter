# ML Pipeline Release v1.0.0

## 🎯 Project Summary
Complete NYC Airbnb Price Prediction ML Pipeline

## 📊 Model Performance
- **Training R² Score:** 0.5519
- **Test R² Score:** 0.5641  
- **Training MAE:** $34.13
- **Test MAE:** $33.85

## 🔧 Components Completed
✅ Data Download (`get_data`)
✅ Exploratory Data Analysis (`eda`) 
✅ Basic Data Cleaning (`basic_cleaning`)
✅ Data Validation (`data_check`)
✅ Train/Val/Test Split (`train_val_test_split`)
✅ Random Forest Training (`train_random_forest`)
✅ Model Testing (`test_regression_model`)

## 🚀 Local Scripts (MLflow Alternative)
- `local_data_split.py` - Data splitting with W&B integration
- `local_train_model.py` - Model training with artifact logging  
- `local_test_model.py` - Model evaluation and testing
- `local_data_check.py` - Data validation and quality checks

## 📦 W&B Artifacts
- `sample.csv` - Raw data
- `trainval_data.csv` - Training/validation split (15,200 rows)
- `test_data.csv` - Test split (3,801 rows)
- `random_forest_export` - Trained model with preprocessing pipeline

## 🛠 Technologies Used
- **MLflow 3.2.0** - Pipeline orchestration
- **Weights & Biases** - Experiment tracking
- **scikit-learn 1.7.1** - Machine learning
- **pandas 2.1.3** - Data manipulation
- **Hydra 1.3.2** - Configuration management

## 🎓 Repository
https://github.com/KerolesDawod12/Project-Build-an-ML-Pipeline-Starter

## 👨‍💻 Author
Keroles Dawod - Western Governors University

## 📝 Notes
This pipeline successfully predicts NYC Airbnb prices with reasonable accuracy. 
The local script implementation bypasses MLflow permission issues while maintaining 
full W&B integration and reproducibility.
