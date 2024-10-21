#!/bin/bash

# Comprehensive GPU Performance and Energy Dataset Generator

# Matrix sizes - more granular steps for better coverage
SIZES=(
    512 768 1024 1280 1536 
    1792 2048 2304 2560 2816 
    3072 3328 3584 3840 4096 
    4352 4608 4864 5120 5376 
    5632 5888 6144 6400 6656 
    6912 7168 7424 7680 7936 
    8192
)

# GEMM Configurations
KERNELS=(
    "sgemm"            # Single precision
    "hgemm"            # Half precision
    "s16816gemm"       # Tensor Core single precision
    "h16816gemm"       # Tensor Core half precision
    "s16832gemm"       # Another Tensor Core variant
    "h16832gemm"
    "s16816gemm_tt"    # Different layouts
    "s16816gemm_tn"
    "s16816gemm_nt"
)

# Different block sizes and configurations
BLOCK_SIZES=(
    "64,64,32"
    "128,128,32"
    "128,64,32"
    "256,128,32"
    "128,256,32"
)

# Different thread block shapes
SHAPES=(
    "128x128x32"
    "256x128x32"
    "128x256x32"
    "256x256x32"
    "64x64x32"
)

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="gpu_dataset_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Create results file with comprehensive headers
RESULTS_FILE="$OUTPUT_DIR/performance_energy_dataset.csv"
echo "Timestamp,Size_M,Size_N,Size_K,Kernel,Block_Size,Shape,Layout,Runtime_ms,GFLOPS,GFLOPS_per_Watt,\
Occupancy,Memory_BW_GB/s,Power_Watts,Energy_Joules,GPU_Util,Mem_Util,SM_Clock,Mem_Clock,Temperature_C,\
L2_Cache_Hits,L1_Cache_Hits,Shared_Memory_Usage,Register_Usage,Active_Warps,\
Active_Blocks,Achieved_Occupancy" > "$RESULTS_FILE"

# Function to collect detailed GPU metrics
collect_gpu_metrics() {
    local output_file=$1
    local duration=$2

    touch "$output_file"
    truncate -s 0 "$output_file"

    # Collect detailed metrics using nvidia-smi
    nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,temperature.gpu \
               --format=csv,noheader,nounits \
               -l 1 -f "$output_file" &
    local NVIDIA_SMI_PID=$!

    # Run DCGM metrics collection in parallel if available
    if command -v dcgmi &> /dev/null; then
        dcgmi dmon -e power,temp,sm_clock,mem_clock,nvlink_bandwidth,nvlink_throughput -d $duration &
        local DCGMI_PID=$!
    fi

    sleep $duration

    if ps -p $NVIDIA_SMI_PID > /dev/null; then
        kill $NVIDIA_SMI_PID 2>/dev/null || true
    fi

    if [ ! -z "$DCGMI_PID" ] && ps -p $DCGMI_PID > /dev/null; then
        kill $DCGMI_PID 2>/dev/null || true
    fi

    # Process metrics
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        awk -F, '{
            power += $1; 
            gpu_util += $2; 
            mem_util += $3;
            sm_clock += $4;
            mem_clock += $5;
            temp += $6;
            count += 1
        } END {
            if (count > 0) printf "%.2f,%.1f,%.1f,%.0f,%.0f,%.1f", 
                power/count, gpu_util/count, mem_util/count,
                sm_clock/count, mem_clock/count, temp/count
            else printf "0,0,0,0,0,0"
        }' "$output_file"
    else
        echo "0,0,0,0,0,0"
    fi
}

# Function to run NCU profiling and extract metrics
run_ncu_profile() {
    local size=$1
    local kernel=$2
    local output_file="$OUTPUT_DIR/ncu_metrics_${size}_${kernel}.csv"

    ncu --metrics launch__waves_per_multiprocessor,\
sm__warps_active,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate,\
lts__t_sector_hit_rate,\
sm__sass_average_data_bytes_per_sector_mem_global_op_ld,\
sm__sass_average_data_bytes_per_sector_mem_shared_op_ld \
        --csv \
        $PROFILER --kernels=$kernel --m=$size --n=$size --k=$size \
        > "$output_file" 2>/dev/null

    # Extract and format metrics from NCU output
    if [ -f "$output_file" ]; then
        awk -F, 'NR>1 {printf "%s,%s,%s,%s,%s,%s", $2,$3,$4,$5,$6,$7}' "$output_file"
    else
        echo "0,0,0,0,0,0"
    fi
}

# Main profiling function
run_profiling() {
    local size=$1
    local kernel=$2
    local block_size=$3
    local shape=$4
    
    echo "Testing size:$size kernel:$kernel block:$block_size shape:$shape"

    # Run different matrix layouts
    for layout in "nn" "nt" "tn" "tt"; do
        local temp_output="$OUTPUT_DIR/temp_${size}_${kernel}_${layout}.txt"
        local metrics_output="$OUTPUT_DIR/metrics_${size}_${kernel}_${layout}.txt"

        # Run CUTLASS profiler with current configuration
        $PROFILER --kernels=$kernel \
                  --m=$size --n=$size --k=$size \
                  --warmup-iterations=5 \
                  --profiling-iterations=100 \
                  --providers=cutlass \
                  --verification-enabled=false \
                  --initialization=random \
                  --alpha=1 --beta=0 \
                  --layout=${layout} \
                  --block-size=${block_size} \
                  > "$temp_output" 2>/dev/null

        # Collect GPU metrics
        local gpu_metrics=$(collect_gpu_metrics "$metrics_output" 5)
        
        # Collect NCU metrics
        local ncu_metrics=$(run_ncu_profile $size $kernel)

        # Extract basic performance metrics
        local runtime=$(grep "Runtime" "$temp_output" | awk '{print $2}')
        local gflops=$(grep "GFLOPs" "$temp_output" | awk '{print $2}')
        local occupancy=$(grep "Occupancy" "$temp_output" | awk '{print $2}')
        local memory_bw=$(grep "Memory" "$temp_output" | awk '{print $2}')

        # Calculate energy consumption
        local power=$(echo $gpu_metrics | cut -d',' -f1)
        local energy=$(echo "$power * $runtime / 1000" | bc -l)
        local gflops_per_watt=$(echo "$gflops / $power" | bc -l)

        # Write to dataset
        echo "$(date +%s),$size,$size,$size,$kernel,$block_size,$shape,$layout,\
$runtime,$gflops,$gflops_per_watt,$occupancy,$memory_bw,$gpu_metrics,$energy,\
$ncu_metrics" >> "$RESULTS_FILE"

        # Cleanup
        rm -f "$temp_output" "$metrics_output"
    done
}

# Main execution loop
echo "Starting comprehensive dataset generation..."

# Record hardware information
nvidia-smi --query-gpu=gpu_name,driver_version,memory.total,power.limit --format=csv > "$OUTPUT_DIR/hardware_info.csv"

# Main nested loops for comprehensive coverage
for size in "${SIZES[@]}"; do
    for kernel in "${KERNELS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            for shape in "${SHAPES[@]}"; do
                run_profiling $size $kernel $block_size $shape
            done
        done
    done
done

# Generate basic statistics summary
echo -e "\nDataset Summary:"
echo "================="
echo "Total configurations tested: $(wc -l < $RESULTS_FILE)"
echo "Dataset saved in: $OUTPUT_DIR"

# Optional: Compress the dataset
tar -czf "${OUTPUT_DIR}.tar.gz" "$OUTPUT_DIR"
echo "Compressed dataset saved as: ${OUTPUT_DIR}.tar.gz"