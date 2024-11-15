import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

class GPUPerformancePredictor:
    def __init__(self):
        print("\nInitializing GPU Performance Predictor...")
        self.numerical_features = [
            'm', 'n', 'k',
            'blocksize1', 'blocksize2', 'blocksize3',
            'alpha', 'beta',
            'arithmetic_intensity',
            'problem_size_m', 'problem_size_n', 'problem_size_k',
            'temp', 'gpu_util', 'mem_util',
            'clock_sm'
        ]
        self.categorical_features = [
            'Layout',
            'stage',
            'combination_type',
            'kernel_name',
            'computation_pattern',
            'uses_shared_memory'
        ]
        self.runtime_model = None
        self.power_model = None
        self.preprocessor = None
        print("Initialization complete.")
    
    def preprocess_data(self, df):
        print("\nPreprocessing data...")
        
        # Create a copy of the dataframe
        df_processed = df.copy()
        
        # Replace '[N/A]' with np.nan
        df_processed = df_processed.replace('[N/A]', np.nan)
        
        # Convert columns to numeric
        numeric_columns = self.numerical_features + ['runtime', 'power']
        for col in numeric_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Fill missing values with mean for numeric columns
        for col in numeric_columns:
            mean_value = df_processed[col].mean()
            df_processed[col] = df_processed[col].fillna(mean_value)
            print(f"Filled {df_processed[col].isna().sum()} missing values in {col}")
        
        # Fill missing values with mode for categorical columns
        for col in self.categorical_features:
            mode_value = df_processed[col].mode()[0]
            df_processed[col] = df_processed[col].fillna(mode_value)
            print(f"Filled {df_processed[col].isna().sum()} missing values in {col}")
        
        # Convert boolean columns to int
        bool_columns = df_processed.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            df_processed[col] = df_processed[col].astype(int)
        
        print("Data preprocessing completed.")
        return df_processed
        
    def create_preprocessor(self):
        print("\nCreating data preprocessor...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(sparse_output=False), self.categorical_features)
            ])
        print("Preprocessor creation complete.")
        return preprocessor

    def create_model(self):
        print("\nCreating model architecture...")
        base_models = [
            ('xgb1', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )),
            ('xgb2', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=43
            )),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ]
        
        final_estimator = LassoCV()
        
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=5
        )
        
        model = Pipeline([
            ('preprocessor', self.create_preprocessor()),
            ('regressor', stacking_model)
        ])
        print("Model architecture creation complete.")
        return model

    def train_models(self, X, y_runtime, y_power):
        print("\nStarting model training process...")
        start_time = time.time()
        
        # Split the data
        print("Splitting data into train and test sets...")
        X_train, X_test, y_runtime_train, y_runtime_test = train_test_split(
            X, y_runtime, test_size=0.2, random_state=42
        )
        _, _, y_power_train, y_power_test = train_test_split(
            X, y_power, test_size=0.2, random_state=42
        )
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Create and train models
        print("\nInitializing models...")
        self.runtime_model = self.create_model()
        self.power_model = self.create_model()
        
        print("\nTraining Runtime Model...")
        runtime_start = time.time()
        self.runtime_model.fit(X_train, y_runtime_train)
        runtime_training_time = time.time() - runtime_start
        print(f"Runtime model training completed in {runtime_training_time:.2f} seconds")
        
        print("\nTraining Power Model...")
        power_start = time.time()
        self.power_model.fit(X_train, y_power_train)
        power_training_time = time.time() - power_start
        print(f"Power model training completed in {power_training_time:.2f} seconds")
        
        # Evaluate models
        print("\nEvaluating models...")
        runtime_pred = self.runtime_model.predict(X_test)
        power_pred = self.power_model.predict(X_test)
        
        metrics = {
            'runtime': {
                'r2': r2_score(y_runtime_test, runtime_pred),
                'mse': mean_squared_error(y_runtime_test, runtime_pred),
                'mae': mean_absolute_error(y_runtime_test, runtime_pred)
            },
            'power': {
                'r2': r2_score(y_power_test, power_pred),
                'mse': mean_squared_error(y_power_test, power_pred),
                'mae': mean_absolute_error(y_power_test, power_pred)
            }
        }
        
        total_time = time.time() - start_time
        print(f"\nTotal training process completed in {total_time:.2f} seconds")
        
        return metrics, (X_test, y_runtime_test, y_power_test, runtime_pred, power_pred)

    def plot_predictions(self, actual_runtime, pred_runtime, actual_power, pred_power):
        print("\nGenerating prediction plots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Runtime predictions
        ax1.scatter(actual_runtime, pred_runtime, alpha=0.5)
        ax1.plot([actual_runtime.min(), actual_runtime.max()], 
                 [actual_runtime.min(), actual_runtime.max()], 
                 'r--', lw=2)
        ax1.set_xlabel('Actual Runtime')
        ax1.set_ylabel('Predicted Runtime')
        ax1.set_title('Runtime Predictions vs Actual')
        
        # Power predictions
        ax2.scatter(actual_power, pred_power, alpha=0.5)
        ax2.plot([actual_power.min(), actual_power.max()], 
                 [actual_power.min(), actual_power.max()], 
                 'r--', lw=2)
        ax2.set_xlabel('Actual Power')
        ax2.set_ylabel('Predicted Power')
        ax2.set_title('Power Predictions vs Actual')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'predictions_plot_{timestamp}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Plots saved as {filename}")

    def save_models(self):
        print("\nSaving trained models...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runtime_filename = f'runtime_model_{timestamp}.joblib'
        power_filename = f'power_model_{timestamp}.joblib'
        
        joblib.dump(self.runtime_model, runtime_filename)
        joblib.dump(self.power_model, power_filename)
        print(f"Models saved as {runtime_filename} and {power_filename}")

def print_column_info(df):
    print("\nAvailable columns in the dataset:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    print(f"\nTotal number of columns: {len(df.columns)}")


def main():
    print("\n" + "="*50)
    print("GPU Performance Prediction System")
    print("="*50)
    
    # Load the data
    print("\nLoading data...")
    start_time = time.time()
    df = pd.read_csv('data.csv')
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    print(f"Dataset shape: {df.shape}")
    
    # Print column information
    print_column_info(df)
    
    # Initialize predictor
    predictor = GPUPerformancePredictor()
    
    # Print selected features
    print("\nSelected numerical features:")
    print(predictor.numerical_features)
    print("\nSelected categorical features:")
    print(predictor.categorical_features)
    
    # Verify features exist in dataset
    missing_num_features = [col for col in predictor.numerical_features if col not in df.columns]
    missing_cat_features = [col for col in predictor.categorical_features if col not in df.columns]
    
    if missing_num_features or missing_cat_features:
        print("\nWARNING: Missing features detected!")
        if missing_num_features:
            print("Missing numerical features:", missing_num_features)
        if missing_cat_features:
            print("Missing categorical features:", missing_cat_features)
        return
    
    # Preprocess the data
    df_processed = predictor.preprocess_data(df)
    
    # Prepare features and targets
    print("\nPreparing features and targets...")
    X = df_processed[predictor.numerical_features + predictor.categorical_features]
    y_runtime = df_processed['runtime']
    y_power = df_processed['power']
    
    # Train models and get metrics
    metrics, prediction_data = predictor.train_models(X, y_runtime, y_power)
    
    # Print metrics
    print("\n" + "="*30)
    print("Model Performance Metrics:")
    print("="*30)
    
    print("\nRuntime Model:")
    print(f"R² Score: {metrics['runtime']['r2']:.4f}")
    print(f"MSE: {metrics['runtime']['mse']:.4f}")
    print(f"MAE: {metrics['runtime']['mae']:.4f}")
    
    print("\nPower Model:")
    print(f"R² Score: {metrics['power']['r2']:.4f}")
    print(f"MSE: {metrics['power']['mse']:.4f}")
    print(f"MAE: {metrics['power']['mae']:.4f}")
    
    # Plot predictions
    X_test, y_runtime_test, y_power_test, runtime_pred, power_pred = prediction_data
    predictor.plot_predictions(y_runtime_test, runtime_pred, y_power_test, power_pred)
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*50)
    print("Process completed successfully!")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: An error occurred during execution:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        raise

