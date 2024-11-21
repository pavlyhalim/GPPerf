import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
file_path = "profiling/execution_times_all.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Add a new column for matrix dimensions as integers in the format "M*N*K"
data['Matrix Dimensions'] = data.apply(
    lambda row: f"{int(row['M'])}*{int(row['N'])}*{int(row['K'])}", axis=1
)

# Add a new column for matrix size (M * N * K)
data['Matrix Size'] = data['M'] * data['N'] * data['K']

# Sort by 'Matrix Size', then 'M', 'N', and 'K' to break ties
data = data.sort_values(by=['Matrix Size', 'M', 'N', 'K'])

# Group the data by 'Tile Size'
tile_size_groups = data.groupby('Tile Size')

# Create the plot
plt.figure(figsize=(12, 6))

for tile_size, group in tile_size_groups:
    if tile_size < 8:
        continue
    plt.plot(
        group['Matrix Dimensions'], 
        group['Execution Time'], 
        marker='o', 
        linestyle='-', 
        label=f"Tile Size: {tile_size}"
    )

plt.title("Runtime vs Matrix Size")
plt.xlabel("Matrix Size (M*N*K)")
plt.ylabel("Execution Time")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend(title="Tile Sizes")
plt.grid(True)
plt.tight_layout()
plt.show()
