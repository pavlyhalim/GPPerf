import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
file_path = "profiling/execution_times_all.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Add a new column for the matrix dimensions as a string "M*N*K"
data['Matrix Dimensions'] = data.apply(lambda row: f"{row['M']}*{row['N']}*{row['K']}", axis=1)

# Add a new column for matrix size (M * N * K) to use for sorting
data['Matrix Size'] = data['M'] * data['N'] * data['K']

# Group the data by 'Tile Size'
tile_size_groups = data.groupby('Tile Size')

# Generate plots
for tile_size, group in tile_size_groups:
    # Sort the group by 'Matrix Size' to ensure x-axis is ordered
    group = group.sort_values('Matrix Size')

    plt.figure(figsize=(12, 6))
    
    # Plot runtime against matrix dimensions
    plt.plot(
        group['Matrix Dimensions'], 
        group['Execution Time'], 
        marker='o', 
        linestyle='-'
    )
    
    plt.title(f"Execution Times(ms) vs Matrix Size for Tile Size {tile_size}")
    plt.xlabel("Matrix Size (M*N*K)")
    plt.ylabel("Execution Time(ms)")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.show()
