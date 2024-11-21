#!/bin/bash

# Output CSV file
OUTPUT_FILE="power_usage_results.csv"

# Write header to CSV file
echo "M,N,K,Tile Size,Average Power Usage (W)" > $OUTPUT_FILE

# Define different values for M, N, K, and tile sizes
M_SIZES=(512 1024 2048)
N_SIZES=(512 1024 2048)
K_SIZES=(512 1024 2048)
TILE_SIZES=(1 4 8 16 32)

# Function to measure power usage while the program runs
measure_power_usage() {
    local PID=$1
    local POWER_SUM=0
    local COUNT=0

    # Poll power usage while the process is running
    while ps -p $PID > /dev/null; do
        POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
        POWER_SUM=$(echo "$POWER_SUM + $POWER" | bc)
        COUNT=$((COUNT + 1))
        sleep 0.1  # Poll every 100 ms
    done

    # Calculate average power usage
    if [ $COUNT -gt 0 ]; then
        echo "scale=3; $POWER_SUM / $COUNT" | bc
    else
        echo "0"
    fi
}

# Iterate over all combinations of M, N, K, and Tile Size
for M in "${M_SIZES[@]}"; do
    for N in "${N_SIZES[@]}"; do
        for K in "${K_SIZES[@]}"; do
            for TILE_SIZE in "${TILE_SIZES[@]}"; do
                total_power=0

                # Run the configuration three times
                for i in {1..3}; do
                    # Start the program in the background
                    ./matmul.o $M $N $K $TILE_SIZE &
                    PROGRAM_PID=$!

                    # Measure power usage while the program is running
                    AVG_POWER=$(measure_power_usage $PROGRAM_PID)

                    # Wait for the program to finish
                    wait $PROGRAM_PID

                    # Accumulate power usage
                    total_power=$(echo "$total_power + $AVG_POWER" | bc)
                done

                # Calculate the average power usage
                avg_power=$(echo "scale=3; $total_power / 3" | bc)

                # Write the results to the CSV file
                echo "$M,$N,$K,$TILE_SIZE,$avg_power" >> $OUTPUT_FILE
            done
        done
    done
done

echo "Power usage results saved to $OUTPUT_FILE"

