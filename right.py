import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.multioutput import MultiOutputRegressor
warnings.filterwarnings('ignore')

class RTX4070GEMMPredictor:
    def __init__(self):
        self.initialize_features()
        self.initialize_models()
        
    def initialize_features(self):
        """Initialize only the essential GEMM features"""
        # Core matrix features
        self.core_features = [
            'm', 'n', 'k',
            'blocksize1', 'blocksize2', 'blocksize3'
        ]
        
        # Performance characteristics
        self.derived_features = [
            'arithmetic_intensity',
            'bytes_accessed',
            'total_flops'
        ]
        
        # Layout feature
        self.categorical_features = ['Layout']
        
        # Target features
        self.target_features = [
            'runtime',
            'power',  
            'Energy',
            'TFlops'
        ]
        
        self.numerical_features = self.core_features + self.derived_features
        
    def initialize_models(self):
        """Initialize the stacked model"""
        self.stacked_model = None

    def sanitize_numeric_columns(self, df, columns):
        """Safely convert columns to numeric type with error handling"""
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns:
                try:
                    #clean any non-numeric characters and handle commas
                    df_clean[col] = df_clean[col].astype(str).str.replace(',', '')
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception as e:
                    print(f"Error converting column {col}: {str(e)}")
                    print(f"Sample values: {df_clean[col].head()}")
        return df_clean
            
    def calculate_gemm_characteristics(self, df):
        """Calculate GEMM-specific characteristics"""
        df = df.copy()
        
        try:
            # First convert core features to numeric
            df = self.sanitize_numeric_columns(df, self.core_features + self.target_features)
            
            # Calculate FLOPS and bytes accessed for GEMM
            df['total_flops'] = 2 * df['m'] * df['n'] * df['k']  # 2 operations per FMA
            df['bytes_accessed'] = (df['m'] * df['k'] + df['k'] * df['n'] + df['m'] * df['n']) * 4  # Single precision
            
            # Calculate arithmetic intensity
            df['arithmetic_intensity'] = df['total_flops'] / df['bytes_accessed']
            
            # Determine if compute or memory bound (RTX 4070 ridge point = 59)
            df['bound_type'] = np.where(df['arithmetic_intensity'] > 59, 'compute', 'memory')
            
            # Handle any potential NaN or infinite values
            for col in self.numerical_features:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
                
            return df
            
        except Exception as e:
            print(f"Error in calculate_gemm_characteristics: {str(e)}")
            raise
            
    def preprocess_data(self, df):
        """Preprocess data focusing on GEMM characteristics"""
        print("\nPreprocessing data...")
        
        try:
            df_processed = df.copy()
            df_processed = df_processed.replace('[N/A]', np.nan)
            df_processed = df_processed.replace('', np.nan)
            
            # Calculate GEMM characteristics
            df_processed = self.calculate_gemm_characteristics(df_processed)
            df_processed['Layout'] = df_processed['Layout'].astype(str)
            
            # Handle outliers for numerical columns
            for col in self.numerical_features + self.target_features:
                if col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    
                    # Remove outliers using percentiles
                    Q1 = df_processed[col].quantile(0.01)
                    Q3 = df_processed[col].quantile(0.99)
                    df_processed[col] = df_processed[col].clip(Q1, Q3)
                    
                    if col in self.target_features:
                        df_processed = df_processed.dropna(subset=[col])
                    else:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    # Handle NaN values
                    if col in ['power', 'energy']:
                        # Drop rows with NaN in 'power' or 'energy'
                        df_processed = df_processed.dropna(subset=[col])
                    else:
                        # Fill NaN with median
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            print("Data preprocessing completed successfully")
            print(f"Features summary:")
            print(df_processed[self.numerical_features].describe())
            
            return df_processed
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            raise

    def create_model(self):
        """Create a stacked model for multi-output regression"""
        numeric_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features)
            ])
        
        # Base estimator for stacking
        base_estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            n_jobs=-1
        )
        
        # Multi-output regressor wrapping the base estimator
        model = MultiOutputRegressor(base_estimator)
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    def train_model(self, X, y):
        """Train the stacked model"""
        print("\nStarting model training process...")
        start_time = datetime.now()
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2, random_state=49
            )
            
            print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Train stacked model
            print("\nTraining Stacked Model...")
            self.stacked_model = self.create_model()
            self.stacked_model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            training_time = datetime.now() - start_time
            print(f"\nTotal training process completed in {training_time}")
            
            return metrics, (X_test, y_test)
            
        except Exception as e:
            print(f"\nError in train_model: {str(e)}")
            raise

    def evaluate_model(self, X_test, y_test):
        """Evaluate the stacked model"""
        predictions = self.stacked_model.predict(X_test)
    
        metrics = {}
        
        for i, target in enumerate(self.target_features):
            y_true = y_test[target]
            y_pred = predictions[:, i]
            percentage_errors = np.abs(y_pred - y_true) / y_true * 100
            
            metrics[target] = {
                'r2': r2_score(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'median_percentage_error': np.median(percentage_errors),
                'mean_percentage_error': np.mean(percentage_errors)
            }
        
        return metrics

    def find_best_config(self, df_processed):
        """Find best configurations for different metrics"""
        best_configs = {
            'runtime': df_processed.loc[df_processed['runtime'].idxmin()],
            'power': df_processed.loc[df_processed['power'].idxmin()],
            'Energy': df_processed.loc[df_processed['Energy'].idxmin()]
        }
        
        print("\nBest Configurations:")
        for metric, config in best_configs.items():
            print(f"\n{metric.upper()} Optimal Configuration:")
            print(f"M={int(config['m'])}, N={int(config['n'])}, K={int(config['k'])}")
            print(f"Block Sizes: {int(config['blocksize1'])}x{int(config['blocksize2'])}x{int(config['blocksize3'])}")
            print(f"Layout: {config['Layout']}")
            print(f"Arithmetic Intensity: {config['arithmetic_intensity']:.2f}")
            print(f"Performance: {metric}={config[metric]:.4f}")
            if metric == 'power':
                print(f"Power: {config['power']:.4f}")
            
        return best_configs

    def save_model(self):
        """Save the stacked model to disk in joblib format"""
        joblib.dump(self.stacked_model, 'stacked_model.joblib')

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("RTX 4070 GEMM Performance Predictor")
    print("="*60)
    
    try:
        #Load data
        print("\nLoading dataset...")
        df = pd.read_csv('data.csv')
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        predictor = RTX4070GEMMPredictor()
        
        print("\nPreprocessing data...")
        df_processed = predictor.preprocess_data(df)
        best_configs = predictor.find_best_config(df_processed)

        X = df_processed[predictor.numerical_features + predictor.categorical_features]
        y = df_processed[predictor.target_features]

        print("\nTraining stacked model...")
        metrics, test_data = predictor.train_model(X, y)
        
        predictor.save_model()

        print("\n" + "="*40)
        print("Stacked Model Performance Metrics:")
        print("="*40)

        for target, perf in metrics.items():
            print(f"\n{target.upper()} Prediction:")
            print(f"R² Score: {perf['r2']:.4f}")
            print(f"MSE: {perf['mse']:.4f}")
            print(f"MAE: {perf['mae']:.4f}")
            print(f"Median Percentage Error: {perf['median_percentage_error']:.2f}%")
            print(f"Mean Percentage Error: {perf['mean_percentage_error']:.2f}%")
        
        print("\n" + "="*60)
        print("GEMM Performance Prediction System completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()