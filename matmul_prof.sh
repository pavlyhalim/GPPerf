#!/bin/bash

# Output CSV file
OUTPUT_FILE="execution_times.csv"

# Write header to CSV file
echo "M,N,K,Tile Size,Execution Time" > $OUTPUT_FILE

# Define different values for M, N, K, and tile sizes
M_SIZES=(512 1024 2048)
N_SIZES=(512 1024 2048)
K_SIZES=(512 1024 2048)
TILE_SIZES=(8 16 32)

# Iterate over all combinations of M, N, K, and Tile Size
for M in "${M_SIZES[@]}"; do
    for N in "${N_SIZES[@]}"; do
        for K in "${K_SIZES[@]}"; do
            for TILE_SIZE in "${TILE_SIZES[@]}"; do
                # Run the program and capture the execution time
                EXECUTION_TIME=$(./matmul.o $M $N $K $TILE_SIZE)
                
                # Extract the execution time (assumes the program prints it at the end)
                TIME=$(echo "$EXECUTION_TIME" | grep -oP "Execution time: \K[0-9\.]+")

                # Write the result to the CSV file
                echo "$M,$N,$K,$TILE_SIZE,$TIME" >> $OUTPUT_FILE
            done
        done
    done
done

echo "Execution times saved to $OUTPUT_FILE"

