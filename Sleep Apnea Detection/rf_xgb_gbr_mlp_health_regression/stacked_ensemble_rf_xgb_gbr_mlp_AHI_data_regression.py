# Stacked Ensemble Model for Health Data Regression: 
# Combining XGBoost, Random Forest, Gradient Boosting, and MLP for Accurate AHI Prediction
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning
import sys

# ===========================
# 1. Configuration and Setup
# ===========================


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ===========================
# 2. Data Loading and Preprocessing
# ===========================

try:
    logging.info("Starting the script...")

    
    root = Tk()
    root.withdraw()  
    logging.info("Tkinter initialized and root hidden.")
    file_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx")])
    root.destroy()  
    logging.info(f"File selected: {file_path}")

    if not file_path:
        raise FileNotFoundError("No file selected.")

    # Load data from the selected Excel file
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip()  
    logging.info("Excel file loaded successfully.")

    # Define the base features and target variable
    base_features = [
        'Age', 'BMI', 'Neckcircum', 'Mean_spO2', 'Min_SpO2', 'Min_HR',
        'Max_HR', 'Arousals_hr', '%< 90', 'STOPBANG_total'
    ]
    target = 'AHI'

    
    required_columns = base_features + [target]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the dataset: {missing_columns}")
    logging.info("All required columns are present in the dataset.")

    
    data = data.dropna(subset=required_columns)
    logging.info("Missing values handled by dropping incomplete rows.")

    
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=required_columns)
    logging.info("Converted required columns to numeric types and dropped non-convertible rows.")

    # Winsorize outliers at the 5th and 95th percentiles to reduce the impact of extreme values
    for col in required_columns:
        data[col] = winsorize(data[col], limits=[0.05, 0.05]).data
    logging.info("Winsorization of outliers completed.")

   
    X = data[base_features].copy()
    y = data[target]
    logging.info("Separated features and target variable.")

    #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Feature scaling using StandardScaler completed.")

   
    X_selected = X_scaled
    selected_feature_names = base_features.copy()
    logging.info("Using all specified base features without feature selection.")

    # ===========================
    # 4. Train-Test Split
    # ===========================

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    logging.info("Performed train-test split (80% train, 20% test).")

    # ===========================
    # 5. Model Definition
    # ===========================

    
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=3,  
        subsample=0.7,  
        colsample_bytree=0.7,  
        n_estimators=100,  
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.2,  
        reg_lambda=1.2,  
        verbosity=0  
    )
    logging.info("Defined XGBoost regressor with enhanced regularization.")

    rf_model = RandomForestRegressor(
        n_estimators=100,  
        max_depth=5,  
        min_samples_split=8,  
        min_samples_leaf=4,  
        random_state=42,
        n_jobs=-1
    )
    logging.info("Defined Random Forest regressor with enhanced regularization.")

    gbr_model = GradientBoostingRegressor(
        learning_rate=0.05,
        n_estimators=150,  
        max_depth=3,  
        subsample=0.8,  
        min_samples_split=6,  
        min_samples_leaf=3,  
        random_state=42
    )
    logging.info("Defined Gradient Boosting regressor with enhanced regularization.")

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(15, 5),  
        activation='relu',
        solver='adam',
        learning_rate_init=0.0005,  
        max_iter=3000, 
        alpha=0.001,  
        random_state=42,
        early_stopping=True,  
        validation_fraction=0.1,  
        n_iter_no_change=50  
    )
    logging.info("Defined MLP regressor with enhanced regularization and reduced complexity.")

   
    final_estimator = ElasticNet(
        alpha=0.5,  
        l1_ratio=0.2,  
        random_state=42
    )
    logging.info("Defined ElasticNet as the final estimator with increased regularization.")

    
    stacking_model = StackingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('gbr', gbr_model),
            ('mlp', mlp_model)
        ],
        final_estimator=final_estimator,
        cv=5,  
        n_jobs=-1  
    )
    logging.info("Defined Stacking Regressor with base models and final estimator.")

    

    # ===========================
    # 6. Model Evaluation with Cross-Validation
    # ===========================

    logging.info("Starting cross-validation...")
    cv_scores = cross_val_score(
        stacking_model,
        X_train,
        y_train,
        cv=5,  
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    average_cv_mse = -np.mean(cv_scores)
    logging.info(f"Average Cross-Validation MSE: {average_cv_mse:.2f}")
    print(f"Average Cross-Validation MSE: {average_cv_mse:.2f}")

    # ===========================
    # 7. Training the Final Model
    # ===========================

    logging.info("Training the final Stacking Regressor model...")
    stacking_model.fit(X_train, y_train)
    logging.info("Final model training completed.")

    # ===========================
    # 8. Predictions and Performance Metrics
    # ===========================

    
    y_train_pred = stacking_model.predict(X_train)
    y_test_pred = stacking_model.predict(X_test)

    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    
    logging.info(f"Training MSE: {train_mse:.2f}")
    logging.info(f"Training R²: {train_r2:.2f}")
    logging.info(f"Test MSE: {test_mse:.2f}")
    logging.info(f"Test R²: {test_r2:.2f}")

    print(f"\nTraining MSE: {train_mse:.2f}")
    print(f"Training R²: {train_r2:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Test R²: {test_r2:.2f}")

    # ===========================
    # 9. Feature Importance Extraction and Plotting
    # ===========================

    
    xgb_importances = stacking_model.named_estimators_['xgb'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': selected_feature_names,
        'Importance': xgb_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=True)  

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.title("Feature Importance (XGBoost - Optimized)")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
    logging.info("Feature importance plot displayed successfully.")

    # ===========================
    # 10. Residual Analysis
    # ===========================

    # Calculate residuals for test set
    residuals = y_test - y_test_pred

    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted AHI")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted AHI")
    plt.tight_layout()
    plt.show()
    logging.info("Residuals vs. Predicted plot displayed successfully.")

    
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, color='purple', edgecolor='k', alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.tight_layout()
    plt.show()
    logging.info("Distribution of Residuals plot displayed successfully.")

    # ===========================
    # 11. Approximate Accuracy Interpretation
    # ===========================


    variance_y = np.var(y_test)
    accuracy = 1 - (test_mse / variance_y)
    logging.info(f"Approximate Accuracy: {accuracy:.2f}")
    print(f"\nApproximate Accuracy: {accuracy:.2f}")

    
except Exception as e:
    logging.error(f"An error occurred: {e}")
    sys.exit(1)




