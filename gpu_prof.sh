#!/bin/bash

# CUTLASS Profiling Script for RTX 4070

# Constants
GPU="RTX_4070"
SIZES=(1024 2048 4096 8192)
PROFILER="./tools/profiler/cutlass_profiler"

# Create output directory
OUTPUT_DIR="profiling_results_rtx4070"
mkdir -p $OUTPUT_DIR

# Create results file and write header
RESULTS_FILE="$OUTPUT_DIR/results.csv"
echo "Size,Kernel,Runtime_ms,GFLOPS,Occupancy,Memory_BW_GB/s,Power_Watts,GPU_Util,Mem_Util,SM_Clock,Mem_Clock,Temperature_C" > $RESULTS_FILE

# Function to collect GPU metrics
collect_gpu_metrics() {
    local output_file=$1
    local duration=$2

    # Ensure the output file exists and is empty
    touch "$output_file"
    truncate -s 0 "$output_file"

    # Start collecting metrics
    nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,temperature.gpu \
               --format=csv,noheader,nounits \
               -l 1 -f "$output_file" &
    local NVIDIA_SMI_PID=$!

    # Wait for the specified duration
    sleep $duration

    # Gracefully stop nvidia-smi
    if ps -p $NVIDIA_SMI_PID > /dev/null; then
        kill $NVIDIA_SMI_PID 2>/dev/null || true
    fi

    # Check if the file exists and has content
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        # Calculate averages using awk
        awk -F, '{
            power += $1; 
            gpu_util += $2; 
            mem_util += $3;
            sm_clock += $4;
            mem_clock += $5;
            temp += $6;
            count += 1
        } END {
            if (count > 0) {
                printf "%.2f,%.1f,%.1f,%.0f,%.0f,%.1f", 
                power/count, 
                gpu_util/count, 
                mem_util/count,
                sm_clock/count,
                mem_clock/count,
                temp/count
            } else {
                printf "0,0,0,0,0,0"
            }
        }' "$output_file"
    else
        echo "0,0,0,0,0,0"
    fi
}

# Function to extract value from CUTLASS profiler output
extract_metric() {
    local file=$1
    local metric=$2
    if [ -f "$file" ]; then
        grep "$metric" "$file" | awk -F': ' '{print $2}' | awk '{print $1}'
    else
        echo "0"
    fi
}

# Function to run profiling for a specific configuration
run_profiling() {
    local size=$1
    local temp_output="$OUTPUT_DIR/temp_output_${size}.txt"
    local metrics_output="$OUTPUT_DIR/metrics_${size}.txt"

    echo "Testing matrix size $size x $size"

    # Array of kernels to test
    local kernels=("sgemm" "hgemm" "s16816gemm" "h16816gemm")

    for kernel in "${kernels[@]}"; do
        echo "Testing kernel: $kernel"
        
        # Run CUTLASS profiler
        $PROFILER --kernels=$kernel \
                  --m=$size --n=$size --k=$size \
                  --warmup-iterations=5 \
                  --profiling-iterations=100 \
                  --providers=cutlass \
                  --verification-enabled=false \
                  > "$temp_output" 2>/dev/null

        # Check if profiler run was successful
        if [ ! -f "$temp_output" ]; then
            echo "Warning: Profiler output not found for $kernel size $size"
            continue
        fi

        # Collect GPU metrics during a profiling run
        echo "Collecting metrics for $kernel..."
        local metrics=$(collect_gpu_metrics "$metrics_output" 5)

        # Extract performance metrics
        local runtime=$(extract_metric "$temp_output" "Runtime")
        local gflops=$(extract_metric "$temp_output" "GFLOPs")
        local occupancy=$(extract_metric "$temp_output" "Occupancy")
        local memory_bw=$(extract_metric "$temp_output" "Memory")

        # Write results to CSV
        echo "$size,$kernel,$runtime,$gflops,$occupancy,$memory_bw,$metrics" >> "$RESULTS_FILE"

        # Clean up temporary file for this kernel
        rm -f "$temp_output"
    done

    # Clean up metrics file
    rm -f "$metrics_output"

    echo "Completed testing size $size"
}

# Main execution
echo "Starting profiling on RTX 4070..."

# Verify GPU is available
if ! nvidia-smi | grep -q "4070"; then
    echo "Error: RTX 4070 not found"
    exit 1
fi

# Test each problem size
for size in "${SIZES[@]}"; do
    run_profiling $size
done

# Generate summary
echo -e "\nPerformance Summary:"
echo "===================="
awk -F',' '
BEGIN {
    print "\nKernel Performance Summary:"
    print "------------------------"
}
NR > 1 {
    kernel = $2;
    gflops = $4;
    power = $7;
    
    if (gflops > max_gflops[kernel]) {
        max_gflops[kernel] = gflops;
        max_size[kernel] = $1;
        max_power[kernel] = power;
    }
    
    sum_gflops[kernel] += gflops;
    sum_power[kernel] += power;
    count[kernel]++;
}
END {
    printf "%-12s %-15s %-15s %-15s\n", "Kernel", "Avg GFLOPS", "Peak GFLOPS", "Size@Peak";
    printf "%-12s %-15s %-15s %-15s\n", "------", "----------", "-----------", "---------";
    for (kernel in sum_gflops) {
        printf "%-12s %-15.2f %-15.2f %-15s\n", 
            kernel, 
            sum_gflops[kernel]/count[kernel],
            max_gflops[kernel],
            max_size[kernel];
    }
}' "$RESULTS_FILE"

echo -e "\nResults saved in $OUTPUT_DIR/results.csv"

# Display hardware information
echo -e "\nHardware Information:"
echo "===================="
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv