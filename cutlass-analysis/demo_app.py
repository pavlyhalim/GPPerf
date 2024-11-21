import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

class GEMMPredictor:
    def __init__(self, model_path='/Users/pavly/Downloads/stacked_model.joblib'):
        self.stacked_model = joblib.load(model_path)
        self.initialize_features()

    def initialize_features(self):
        """Initialize features used by the model"""
        # Core matrix features
        self.core_features = [
            'm', 'n', 'k',
            'blocksize1', 'blocksize2', 'blocksize3'
        ]
        # Derived features
        self.derived_features = [
            'arithmetic_intensity',
            'bytes_accessed',
            'total_flops'
        ]
        # Categorical features
        self.categorical_features = ['Layout']
        # Target features
        self.target_features = [
            'runtime',
            'power',
            'Energy',
            'TFlops'
        ]
        self.numerical_features = self.core_features + self.derived_features
        
    def calculate_gemm_characteristics(self, m, n, k, blocksize1, blocksize2, blocksize3):
        """Calculate GEMM-specific characteristics"""
        total_flops = 2 * m * n * k  # 2 operations per FMA
        bytes_accessed = (m * k + k * n + m * n) * 4  # Single precision
        arithmetic_intensity = total_flops / bytes_accessed
        bound_type = 'compute' if arithmetic_intensity > 59 else 'memory'
        
        return {
            'total_flops': total_flops,
            'bytes_accessed': bytes_accessed,
            'arithmetic_intensity': arithmetic_intensity,
            'bound_type': bound_type
        }
        
    def get_default_numeric_values(self):
        """Return default values for missing numeric features"""
        return {
            # Memory-related defaults
            'total_memory': 12288,  # 12GB for RTX 4070
            'free_memory': 10240,   # Assuming 80% free
            'used_memory': 2048,    # Assuming 20% used
            'mem_util': 20.0,       # 20% utilization
            'mem_util2': 20.0,      # Secondary memory utilization
            
            # GPU state defaults
            'temp': 65.0,           # Default temperature
            'gpu_util': 80.0,       # Default GPU utilization
            'gpu_util1': 80.0,      # Secondary GPU utilization
            'clock_sm': 2475,       # Default SM clock for RTX 4070
            'power_limit': 200.0,   # Default power limit
            'clocks.meme': 2000,    # Memory clock speed
            
            'alpha': 1.0,           # Default scaling factor
            'beta': 0.0,            # Default scaling factor
            'problem_size_m': 1024,
            'problem_size_n': 1024,
            'problem_size_k': 1024
        }

    def get_default_categorical_values(self):
        """Return default values for missing categorical features"""
        return {
            'stage': 'main',
            'kernel_name': 'cutlass_simt_sgemm_128x128_8x2_nn_align1',
            'computation_pattern': 'GEMM',
            'combination_type': 'standard',
            'state': 'active',
            'uses_shared_memory': 'true',
            'gpu_name': 'RTX4070'
        }
        
    def prepare_input_data(self, input_dict):
        """Prepare input data for prediction with default values for missing features"""
        numeric_defaults = self.get_default_numeric_values()
        categorical_defaults = self.get_default_categorical_values()
        
        complete_input = {**numeric_defaults, **categorical_defaults}
        
        complete_input.update(input_dict)
        
        df = pd.DataFrame([complete_input])
        
        characteristics = self.calculate_gemm_characteristics(
            df['m'].iloc[0], df['n'].iloc[0], df['k'].iloc[0],
            df['blocksize1'].iloc[0], df['blocksize2'].iloc[0], df['blocksize3'].iloc[0]
        )
        
        df['total_flops'] = characteristics['total_flops']
        df['bytes_accessed'] = characteristics['bytes_accessed']
        df['arithmetic_intensity'] = characteristics['arithmetic_intensity']
        
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def estimate_power(df):
        BASE_POWER = 30  
        MAX_POWER = 200 
        MAX_TFLOPS = 40
        
        df['estimated_power'] = BASE_POWER + (
            (MAX_POWER - BASE_POWER) * 
            (df['total_flops'] / (MAX_TFLOPS * 1e12))
        )
        
        df['power'] = df['power'].fillna(df['estimated_power'])
        
        return df

    def filter_power_bounds(df):
        MIN_POWER = 25  # Minimum idle power
        MAX_POWER = 200 # Maximum TDP
        
        df = df[
            (df['power'].between(MIN_POWER, MAX_POWER)) | 
            (df['power'].isna())
        ]
        
        return df
    
    def impute_power(df):
        df['total_elements'] = df['m'] * df['n'] * df['k']
        valid_power = df[df['power'].notna()]
        
        features = ['total_elements', 'total_flops', 'arithmetic_intensity']
        X = valid_power[features]
        y = valid_power['power']
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        missing_power = df[df['power'].isna()]
        imputed_values = model.predict(missing_power[features])
        df.loc[df['power'].isna(), 'power'] = imputed_values
        
        return df

    def preprocess_data(self, df):
        """Preprocess data focusing on GEMM characteristics with improved power handling"""
        print("\nPreprocessing data...")
        
        try:
            df_processed = df.copy()
            df_processed = df_processed.replace('[N/A]', np.nan)
            df_processed = df_processed.replace('', np.nan)
            df_processed = self.calculate_gemm_characteristics(df_processed)

            df_processed['Layout'] = df_processed['Layout'].astype(str)
            
            df_processed = self.estimate_power(df_processed)
            df_processed = self.impute_power(df_processed)
            df_processed = self.filter_power_bounds(df_processed)
            
            for col in self.numerical_features:
                if col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    Q1 = df_processed[col].quantile(0.01)
                    Q3 = df_processed[col].quantile(0.99)
                    df_processed[col] = df_processed[col].clip(Q1, Q3)
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            print("Data preprocessing completed successfully")
            print(f"Features summary:")
            print(df_processed[self.numerical_features].describe())
            
            return df_processed
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            raise

    def predict(self, input_data):
        """Make predictions using the stacked model"""
        df = self.prepare_input_data(input_data)
        predictions = self.stacked_model.predict(df)

        # Map predictions to target features
        prediction_dict = {target: predictions[0][i] for i, target in enumerate(self.target_features)}

        prediction_dict['characteristics'] = self.calculate_gemm_characteristics(
            input_data['m'], input_data['n'], input_data['k'],
            input_data['blocksize1'], input_data['blocksize2'], input_data['blocksize3']
        )

        return prediction_dict

def create_comparison_chart(current_metrics, optimal_metrics):
    """Create a comparison chart using plotly"""
    metrics = ['Runtime (ms)', 'Power (W)', 'Energy (J)', 'TFLOPS']
    current_values = [
        current_metrics['runtime'],
        current_metrics['power'],
        current_metrics['Energy'],  
        current_metrics['TFlops']  
    ]
    optimal_values = [
        optimal_metrics['runtime'],
        optimal_metrics['power'],
        optimal_metrics['Energy'],  
        optimal_metrics['TFlops']
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Current', x=metrics, y=current_values, marker_color='#ff7c43'),
        go.Bar(name='Optimal', x=metrics, y=optimal_values, marker_color='#00ba38')
    ])
    
    fig.update_layout(
        barmode='group',
        title='Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Values',
        height=400
    )
    
    return fig

def create_heatmap(m, n, k, block_m, block_n):
    """Create a heatmap visualization of the matrix blocking"""
    grid_m = int(np.ceil(m / block_m))
    grid_n = int(np.ceil(n / block_n))
    
    grid = np.random.uniform(0.5, 1.0, (grid_m, grid_n))
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale='Viridis',
        showscale=False
    ))
    
    fig.update_layout(
        title='Matrix Blocking Visualization',
        xaxis_title='N dimension (columns)',
        yaxis_title='M dimension (rows)',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_performance_metrics_chart(predictions):
    """Create a gauge chart for TFLOPS and other metrics"""
    max_tflops = 40  # RTX 4070 theoretical max
    tflops_percentage = (predictions['TFlops'] / max_tflops) * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predictions['TFlops'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "TFLOPS Performance"},
        gauge = {
            'axis': {'range': [None, max_tflops]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_tflops/3], 'color': "red"},
                {'range': [max_tflops/3, 2*max_tflops/3], 'color': "yellow"},
                {'range': [2*max_tflops/3, max_tflops], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': predictions['TFlops']
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_efficiency_chart(arithmetic_intensity, mem_bandwidth_utilization, compute_utilization):
    """Create a spider chart showing various efficiency metrics"""
    fig = go.Figure()
    
    categories = ['Arithmetic Intensity', 'Memory BW Utilization', 'Compute Utilization']
    
    fig.add_trace(go.Scatterpolar(
        r=[arithmetic_intensity/200*100, mem_bandwidth_utilization, compute_utilization],
        theta=categories,
        fill='toself',
        name='Current Configuration'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=300
    )
    
    return fig

def main():
    st.set_page_config(page_title="GEMM Performance Predictor", layout="wide")
    st.markdown("""
        <style>
        .main {
            padding: 2rem 1rem;
            max-width: 100%;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("GEMM Performance Predictor for RTX 4070")
    
    try:
        predictor = GEMMPredictor()
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.subheader("Matrix Dimensions")
            with st.expander("Set Matrix Dimensions", expanded=True):
                m = st.number_input("M", min_value=1, value=512)
                n = st.number_input("N", min_value=1, value=512)
                k = st.number_input("K", min_value=1, value=1024)
        
        with col2:
            st.subheader("Block Sizes")
            with st.expander("Set Block Dimensions", expanded=True):
                blocksize1 = st.number_input("Block Size 1", min_value=1, value=512)
                blocksize2 = st.number_input("Block Size 2", min_value=1, value=128)
                blocksize3 = st.number_input("Block Size 3", min_value=1, value=512)
        
        with col3:
            st.subheader("Configuration")
            with st.expander("Additional Settings", expanded=True):
                layout = st.selectbox("Matrix Layout", ['nn', 'nt', 'tn', 'tt'])
                kernel_name = st.selectbox(
                    "CUTLASS Kernel",
                    [
                        'cutlass_simt_sgemm_128x128_8x2_nn_align1',
                        'cutlass_simt_sgemm_128x128_8x2_nt_align1',
                        'cutlass_simt_sgemm_128x128_8x2_tn_align1',
                        'cutlass_simt_sgemm_128x128_8x2_tt_align1'
                    ]
                )
                alpha = st.number_input("Alpha Scalar", value=1.00, step=0.25)
                beta = st.number_input("Beta Scalar", value=0.50, step=0.25)
        
        if st.button("Analyze Performance", use_container_width=True):
            with st.spinner("Analyzing performance..."):
                input_data = {
                    'm': m, 'n': n, 'k': k,
                    'blocksize1': blocksize1,
                    'blocksize2': blocksize2,
                    'blocksize3': blocksize3,
                    'Layout': layout,
                    'kernel_name': kernel_name,
                    'alpha': alpha,
                    'beta': beta
                }
                predictions = predictor.predict(input_data)
                
                tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Detailed Analysis", "Visualizations"])
                
                with tab1:
                    st.subheader("GEMM Characteristics")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric(
                            "Arithmetic Intensity",
                            f"{predictions['characteristics']['arithmetic_intensity']:.2f}",
                            f"{predictions['characteristics']['bound_type'].upper()} bound"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Total FLOPS",
                            f"{predictions['characteristics']['total_flops']/1e9:.2f}G",
                            "Operations"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Memory Accessed",
                            f"{predictions['characteristics']['bytes_accessed']/1e6:.2f}MB",
                            "Total Data Movement"
                        )
                    
                    with metric_col4:
                        memory_efficiency = min(100, predictions['characteristics']['bytes_accessed'] / (504 * 1e9) * 100)
                        st.metric(
                            "Memory Efficiency",
                            f"{memory_efficiency:.1f}%",
                            "vs Peak Bandwidth"
                        )
                    
                    st.markdown("---")
                    
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                    
                    with perf_col1:
                        st.metric(
                            "Runtime",
                            f"{max(0.01, predictions['runtime']):.2f} ms",
                            "Execution Time"
                        )
                    
                    with perf_col2:
                        st.metric(
                            "Power",
                            f"{max(1.0, predictions['power']):.2f} W",
                            "Power Consumption"
                        )
                    
                    with perf_col3:
                        st.metric(
                            "Energy",
                            f"{max(0.01, predictions['Energy']):.2f} J", 
                            "Total Energy"
                        )
                    
                    with perf_col4:
                        efficiency = (predictions['TFlops'] / 40) * 100
                        st.metric(
                            "TFLOPS",
                            f"{predictions['TFlops']:.2f}", 
                            f"{efficiency:.1f}% of Peak"
                        )
                
                with tab2:
                    st.subheader("Detailed Performance Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Matrix Configuration")
                        st.markdown(f"""
                        - Total Matrix Elements: {m*n:,}
                        - Memory Footprint: {predictions['characteristics']['bytes_accessed']/1e6:.2f} MB
                        - Block Dimensions: {blocksize1}x{blocksize2}x{blocksize3}
                        - Grid Size: {m//blocksize1}x{n//blocksize2} blocks
                        """)
                    
                    with col2:
                        st.markdown("#### Performance Bottlenecks")
                        ai = predictions['characteristics']['arithmetic_intensity']
                        if ai > 59:
                            st.success("✅ Compute Bound - Optimal for GPU")
                        else:
                            st.warning("⚠️ Memory Bound - Consider Optimization")
                        
                        efficiency = (predictions['TFlops'] / 40) * 100
                        if efficiency < 30:
                            st.error("🔴 Low Compute Efficiency - Check Configuration")
                        elif efficiency < 60:
                            st.warning("🟡 Moderate Efficiency - Room for Improvement")
                        else:
                            st.success("🟢 Good Efficiency")
                
                with tab3:
                    st.subheader("Performance Visualizations")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.plotly_chart(create_performance_metrics_chart(predictions), use_container_width=True)
                    
                    with viz_col2:
                        mem_bw_util = min(100, predictions['characteristics']['bytes_accessed'] / (504 * 1e9) * 100)
                        compute_util = min(100, (predictions['TFlops'] / 40) * 100)
                        st.plotly_chart(
                            create_efficiency_chart(
                                predictions['characteristics']['arithmetic_intensity'],
                                mem_bw_util,
                                compute_util
                            ),
                            use_container_width=True
                        )
                    
                    st.plotly_chart(create_heatmap(m, n, k, blocksize1, blocksize2), use_container_width=True)
                    
                    st.markdown("### Recommendations")
                    
                    recommendations = []
                    if blocksize1 * blocksize2 > 1024:
                        recommendations.append("⚠️ Block size might be too large for optimal occupancy")
                    if predictions['characteristics']['arithmetic_intensity'] < 30:
                        recommendations.append("Consider increasing arithmetic intensity through blocking")
                    if efficiency < 50:
                        recommendations.append("Performance is below 50% of peak - try different block sizes")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    else:
                        st.success("Current configuration appears optimal!")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure the model file 'rtx4070_performance_models.joblib' is in the correct directory.")
        st.write("If the error persists, check the input parameters and model compatibility.")

if __name__ == "__main__":
    main()