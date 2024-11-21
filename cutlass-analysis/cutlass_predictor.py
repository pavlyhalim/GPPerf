import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.multioutput import MultiOutputRegressor
import joblib
warnings.filterwarnings('ignore')

class EnhancedVisualizer:
    def __init__(self):
        plt.style.use('default') 
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
    def plot_performance_metrics(self, metrics):
        """performance metrics visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        r2_scores = {target: metrics[target]['r2'] for target in metrics}
        ax1.bar(r2_scores.keys(), r2_scores.values(), color=self.color_palette)
        ax1.set_title('R² Scores by Target Variable', pad=20)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('white')
        
        error_metrics = pd.DataFrame({
            target: {
                'MAE': metrics[target]['mae'],
                'MSE': metrics[target]['mse'],
                'Mean % Error': metrics[target]['mean_percentage_error']
            } for target in metrics
        }).T
        
        error_metrics.plot(kind='bar', ax=ax2, width=0.8, color=self.color_palette[:3])
        ax2.set_title('Error Metrics by Target', pad=20)
        ax2.set_yscale('log')
        ax2.set_facecolor('white')
        
        plt.tight_layout()
        return fig

    def plot_predictions_vs_actual(self, X_test, y_test, predictions, target_features):
        plt.style.use('default')
        for i, target in enumerate(target_features):
            fig, ax1 = plt.subplots(figsize=(20, 8), dpi=300)
            fig.patch.set_facecolor('white')
            
            y_true = y_test[target].values
            y_pred = predictions[:, i]
            
            # Create DataFrame with matrix dimensions and sort by m*n
            matrix_dims = pd.DataFrame({
                'size_str': [f"{m}×{n}" for m, n in zip(X_test['m'], X_test['n'])],
                'mn_size': X_test['m'] * X_test['n'],
                'k_size': X_test['k'],
                'y_true': y_true,
                'y_pred': y_pred
            })
            
            grouped = matrix_dims.groupby(['size_str', 'k_size']).agg({
                'mn_size': 'first',
                'y_true': 'mean',
                'y_pred': 'mean'
            }).reset_index()
            
            grouped = grouped.sort_values(['mn_size', 'k_size'])
            
            grouped['combined_label'] = grouped.apply(
                lambda row: f"{row['size_str']}×{row['k_size']}", axis=1
            )
            
            # Plot mean lines
            ax1.plot(range(len(grouped)), grouped['y_true'], 'k--', 
                    linewidth=2, label='Mean Actual')
            ax1.plot(range(len(grouped)), grouped['y_pred'], 'r--', 
                    linewidth=2, label='Mean Predicted')
            ax1.plot(range(len(grouped)), grouped['y_pred'], 'bx', 
                    markersize=8, label='Mean Predicted Points')
            
            if len(grouped) > 100:
                step = len(grouped) // 10 + 1
                xticks = range(0, len(grouped), step)
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(grouped['combined_label'].iloc[xticks], 
                                rotation=45, ha='right')
            else:
                ax1.set_xticks(range(len(grouped)))
                ax1.set_xticklabels(grouped['combined_label'], 
                                rotation=45, ha='right')
            
            ax1.set_xlabel('Matrix Dimensions (m×n×k)', fontsize=12, fontweight='bold')
            ax1.set_ylabel(target, fontsize=12, fontweight='bold')
            ax1.set_title(f'{target}: Mean Values vs Matrix Size (Sorted by m×n)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax1.set_yscale('log', base=2)
            ax1.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none',
                    bbox_to_anchor=(1.02, 1), loc='upper left')
            ax1.grid(True, which="both", ls="-", alpha=0.2)
            ax1.set_axisbelow(True)

            for idx in range(len(grouped)):
                if idx % (len(grouped) // 10 + 1) == 0:
                    mn_size = grouped['mn_size'].iloc[idx]
                    ax1.annotate(f'size: {mn_size:,}', 
                            xy=(idx, ax1.get_ylim()[0]),
                            xytext=(0, -40), textcoords='offset points',
                            ha='right', va='top', rotation=45,
                            fontsize=8, color='gray')

            plt.tight_layout()
            yield fig

class RTX4070GEMMPredictor:
    def __init__(self):
        self.initialize_features()
        self.initialize_models()
        self.visualizer = EnhancedVisualizer()
        
    def initialize_features(self):
        """Initialize only the essential GEMM features"""
        self.core_features = [
            'm', 'n', 'k',
            'blocksize1', 'blocksize2', 'blocksize3'
        ]
        
        self.derived_features = [
            'arithmetic_intensity',
            'bytes_accessed',
            'total_flops'
        ]
        
        self.categorical_features = ['Layout']
        
        self.target_features = [
            'runtime',
            'power',
            'Energy',
            'TFlops'
        ]
        
        self.numerical_features = self.core_features + self.derived_features
        
    def initialize_models(self):
        self.stacked_model = None

    def sanitize_numeric_columns(self, df, columns):
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = df_clean[col].astype(str).str.replace(',', '')
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception as e:
                    print(f"Error converting column {col}: {str(e)}")
                    print(f"Sample values: {df_clean[col].head()}")
        return df_clean
            
    def calculate_gemm_characteristics(self, df):
        df = df.copy()
        
        try:
            df = self.sanitize_numeric_columns(df, self.core_features + self.target_features)
            
            df['total_flops'] = 2 * df['m'] * df['n'] * df['k']
            df['bytes_accessed'] = (df['m'] * df['k'] + df['k'] * df['n'] + df['m'] * df['n']) * 4
            df['arithmetic_intensity'] = df['total_flops'] / df['bytes_accessed']
            df['bound_type'] = np.where(df['arithmetic_intensity'] > 59, 'compute', 'memory')
            
            for col in self.numerical_features:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
                
            return df
            
        except Exception as e:
            print(f"Error in calculate_gemm_characteristics: {str(e)}")
            raise
            
    def preprocess_data(self, df):
        print("\nPreprocessing data...")
        
        try:
            df_processed = df.copy()
            df_processed = df_processed.replace('[N/A]', np.nan)
            df_processed = df_processed.replace('', np.nan)
            
            df_processed = self.calculate_gemm_characteristics(df_processed)
            df_processed['Layout'] = df_processed['Layout'].astype(str)
            
            for col in self.numerical_features + self.target_features:
                if col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    
                    Q1 = df_processed[col].quantile(0.01)
                    Q3 = df_processed[col].quantile(0.99)
                    df_processed[col] = df_processed[col].clip(Q1, Q3)
                    
                    if col in self.target_features:
                        df_processed = df_processed.dropna(subset=[col])
                    else:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            print("Data preprocessing completed successfully")
            return df_processed
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            raise

    def create_model(self):
        numeric_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features)
            ])
        
        base_estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            n_jobs=-1
        )
        
        model = MultiOutputRegressor(base_estimator)
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    def train_model(self, X, y):
        print("\nStarting model training process...")
        start_time = datetime.now()
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )
            
            print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            self.stacked_model = self.create_model()
            self.stacked_model.fit(X_train, y_train)
            
            predictions = self.stacked_model.predict(X_test)
            metrics = self.evaluate_model(X_test, y_test)
            
            self.create_visualization_suite(X_test, y_test, predictions, metrics)
            
            training_time = datetime.now() - start_time
            print(f"\nTotal training process completed in {training_time}")
            
            return metrics, (X_test, y_test, predictions)
            
        except Exception as e:
            print(f"\nError in train_model: {str(e)}")
            raise

    def evaluate_model(self, X_test, y_test):
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
    
    def export_model(self, filepath):
        """Export the trained model to a Joblib file."""
        joblib.dump(self.stacked_model, filepath)

    def calculate_and_verify_energy(self, df):
        """Verify and recalculate energy values"""
        print("\nVerifying energy calculations...")
        
        df_check = df.copy()
        
        df_check['Expected_Energy'] = df_check['runtime'] * df_check['power'] / 1000
        
        print("\nEnergy Statistics:")
        print("Existing Energy mean:", df_check['Energy'].mean())
        print("Calculated Energy mean:", df_check['Expected_Energy'].mean())
        print("\nCorrelation between existing and calculated Energy:", 
            df_check['Energy'].corr(df_check['Expected_Energy']))
        
        existing_corr = df_check['Energy'].corr(df_check['M×N×K'])
        calculated_corr = df_check['Expected_Energy'].corr(df_check['M×N×K'])
        print(f"\nCorrelation with M×N×K:")
        print(f"Existing Energy: {existing_corr:.3f}")
        print(f"Calculated Energy: {calculated_corr:.3f}")
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df_check['M×N×K'], df_check['Energy'], alpha=0.5, label='Existing')
        plt.xlabel('M×N×K')
        plt.ylabel('Energy (J)')
        plt.title('Existing Energy vs Matrix Size')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(df_check['M×N×K'], df_check['Expected_Energy'], alpha=0.5, label='Calculated')
        plt.xlabel('M×N×K')
        plt.ylabel('Energy (J)')
        plt.title('Calculated Energy vs Matrix Size')
        plt.legend()
        
        plt.tight_layout()
        
        return plt.gcf(), df_check


    def plot_correlation_heatmap(self, df):
        """Create correlation heatmap between matrix dimensions and performance metrics"""
        dimensions_df = pd.DataFrame({
            'M': df['m'],
            'N': df['n'],
            'K': df['k'],
            'M×N': df['m'] * df['n'],
            'M×K': df['m'] * df['k'],
            'N×K': df['n'] * df['k'],
            'M×N×K': df['m'] * df['n'] * df['k'],
            'Runtime': df['runtime'],
            'Energy': df['Energy'],
            'Power': df['power'],
            'TFLOPS': df['TFlops']
        })

        correlation_matrix = dimensions_df.corr()
        plt.figure(figsize=(12, 10), dpi=300)
        
        mask = np.triu(np.ones_like(correlation_matrix), k=0)
        
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True,           
                    fmt='.2f',            
                    cmap='RdBu_r',       
                    vmin=-1, vmax=1,     
                    center=0,            
                    square=True,         
                    linewidths=0.5,       
                    cbar_kws={"shrink": .8})

        plt.title('Correlation between Matrix Dimensions and Performance Metrics', 
                pad=20, fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        return plt.gcf()

    def create_visualization_suite(self, X_test, y_test, predictions, metrics):
        """visualization suite"""
        try:
            perf_fig = self.visualizer.plot_performance_metrics(metrics)
            perf_fig.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
            
            for i, pred_fig in enumerate(self.visualizer.plot_predictions_vs_actual(
                    X_test, y_test, predictions, self.target_features)):
                pred_fig.savefig(f'predictions_{self.target_features[i]}.png', 
                            dpi=300, bbox_inches='tight')
        
            analysis_df = pd.concat([
                pd.DataFrame(X_test, columns=self.numerical_features + self.categorical_features),
                pd.DataFrame(y_test, columns=self.target_features)
            ], axis=1)
            
            corr_fig = self.plot_correlation_heatmap(analysis_df)
            corr_fig.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            
            plt.close('all')
            print("\nVisualization suite generated successfully")
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            raise
    
def main():
    print("\n" + "="*60)
    print("RTX 4070 GEMM Performance Predictor")
    print("="*60)
    
    try:
        print("\nLoading dataset...")
        df = pd.read_csv('data.csv')
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        predictor = RTX4070GEMMPredictor()
        df_processed = predictor.preprocess_data(df)

        X = df_processed[predictor.numerical_features + predictor.categorical_features]
        y = df_processed[predictor.target_features]

        print("\nTraining model and generating visualizations...")
        metrics, (X_test, y_test, predictions) = predictor.train_model(X, y)
        predictor.export_model('trained_model.joblib')

        print("\n" + "="*40)
        print("Model Performance Metrics:")
        print("="*40)

        for target, perf in metrics.items():
            print(f"\n{target.upper()}:")
            print(f"R² Score: {perf['r2']:.4f}")
            print(f"MSE: {perf['mse']:.4f}")
            print(f"MAE: {perf['mae']:.4f}")
            print(f"Median % Error: {perf['median_percentage_error']:.2f}%")
            print(f"Mean % Error: {perf['mean_percentage_error']:.2f}%")
        
    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()