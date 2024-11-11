#!/bin/bash

# Complete CUTLASS Extensive Profiling Script

# Matrix dimensions to test
# M_SIZES=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7680 8192)
# N_SIZES=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7680 8192)
# K_SIZES=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7680 8192)

M_SIZES=(512 1024)
N_SIZES=(512 1024)
K_SIZES=(512 1024)


# CUTLASS kernels with different configurations
KERNELS=(
    # SIMT GEMM kernels
    "cutlass_simt_sgemm_128x128_8x2_nn_align1"
    "cutlass_simt_sgemm_128x128_8x2_nt_align1"
    "cutlass_simt_sgemm_128x128_8x2_tn_align1"
    "cutlass_simt_sgemm_128x128_8x2_tt_align1"
    "cutlass_simt_hgemm_128x128_8x2_nn_align1"
    "cutlass_simt_hgemm_128x128_8x2_nt_align1"
    "cutlass_simt_hgemm_128x128_8x2_tn_align1"
    "cutlass_simt_hgemm_128x128_8x2_tt_align1"
    
    # Tensor Core GEMM kernels - Ampere
    # "cutlass_tensorop_s16816gemm_128x128_32x3_nn_align1"
    # "cutlass_tensorop_s16816gemm_128x128_32x3_nt_align1"
    # "cutlass_tensorop_s16816gemm_128x128_32x3_tn_align1"
    # "cutlass_tensorop_s16816gemm_128x128_32x3_tt_align1"
    # "cutlass_tensorop_h16816gemm_128x128_32x3_nn_align1"
    # "cutlass_tensorop_h16816gemm_128x128_32x3_nt_align1"
    # "cutlass_tensorop_h16816gemm_128x128_32x3_tn_align1"
    # "cutlass_tensorop_h16816gemm_128x128_32x3_tt_align1"
    
    # Larger tile sizes
    # "cutlass_tensorop_s16816gemm_256x128_32x3_nn_align1"
    # "cutlass_tensorop_s16816gemm_256x128_32x3_nt_align1"
    # "cutlass_tensorop_h16816gemm_256x128_32x3_nn_align1"
    # "cutlass_tensorop_h16816gemm_256x128_32x3_nt_align1"
    
    # Different alignments
    # "cutlass_simt_sgemm_128x128_8x2_nn_align4"
    # "cutlass_simt_sgemm_128x128_8x2_nn_align8"
    # "cutlass_tensorop_s16816gemm_128x128_32x3_nn_align4"
    # "cutlass_tensorop_s16816gemm_128x128_32x3_nn_align8"
)

# Different block sizes to test
BLOCK_SIZES=(
    "64,64,32"
    "128,64,32"
    "64,128,32"
    "128,128,32"
    "256,128,32"
    "128,256,32"
    "256,256,32"
)

# Stage counts to test
STAGES=(2 3 4 5)

# Epilogue operations
EPILOGUES=(
    "linear_combination"
    # "linear_combination_relu"
    # "linear_combination_gelu"
)

# Alpha/Beta combinations
SCALARS=(
    "1,0"
    "1,1"
    "0.5,0.5"
    "2,0"
)

# Matrix layouts
LAYOUTS=("nn" "nt" "tn" "tt")

# Create output directory with timestamp
OUTPUT_DIR="extensive_profiling_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Create comprehensive results file with all metrics
RESULTS_FILE="$OUTPUT_DIR/detailed_results.csv"
echo "Timestamp,M,N,K,Kernel,Layout,Block_Size,Stages,Epilogue,Alpha,Beta,\
Runtime_ms,GPU_Time_ms,CPU_Time_ms,\
SM_Occupancy_pct,Warp_Execution_Efficiency_pct,Instructions_Per_Cycle,\
Global_Memory_Read_Throughput_GBs,Global_Memory_Write_Throughput_GBs,\
L1_Cache_Hit_Rate_pct,L2_Cache_Hit_Rate_pct,\
Shared_Memory_Utilization_pct,Memory_Bandwidth_Utilization_pct,\
Arithmetic_Instructions,Logic_Instructions,Control_Instructions,Memory_Instructions,\
Divergent_Branch_pct,\
Power_Watts,Energy_Joules,\
Grid_Size,Block_Size,\
Shared_Memory_Per_Block_Bytes,Registers_Per_Thread,\
GPU_Utilization_pct,Memory_Utilization_pct,\
SM_Clock_MHz,Memory_Clock_MHz,\
Temperature_C,\
TFLOPS" > $RESULTS_FILE

# Function to check if configuration is valid
check_valid_config() {
    local m=$1
    local n=$2
    local k=$3
    local kernel=$4
    
    # Skip if dimensions are too large
    if [ $((m * n * k)) -gt $((8192 * 8192 * 8192)) ]; then
        return 1
    fi
    
    # Skip invalid tensor core configurations for small sizes
    if [[ $kernel == *"tensorop"* ]]; then
        if [ $m -lt 128 ] || [ $n -lt 128 ] || [ $k -lt 64 ]; then
            return 1
        fi
    fi
    
    # Skip invalid SIMT configurations for large sizes
    if [[ $kernel == *"simt"* ]]; then
        if [ $m -gt 8192 ] || [ $n -gt 8192 ] || [ $k -gt 8192 ]; then
            return 1
        fi
    fi
    
    return 0
}

# Function to collect NSight Compute metrics
collect_ncu_metrics() {
    local kernel=$1
    local m=$2
    local n=$3
    local k=$4
    local output_file="$OUTPUT_DIR/ncu_${m}_${n}_${k}_${kernel// /_}.csv"

    # Run NSight Compute with all relevant metrics
    ncu --metrics \
        sm__warps_active.avg.pct_of_peak_sustained_active,\
        sm__warps_eligible.avg.pct_of_peak_sustained_active,\
        sm__inst_executed_pipe_alu.avg.per_cycle_active,\
        sm__inst_executed.avg.per_cycle_active,\
        sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
        lts__t_sector_hit_rate.pct,\
        sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
        sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
        sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
        sm__sass_thread_inst_executed_op_integer_pred_on.sum,\
        sm__sass_thread_inst_executed_op_control_pred_on.sum,\
        sm__sass_thread_inst_executed_op_memory_pred_on.sum,\
        sm__sass_branch_targets.avg,\
        sm__sass_branch_targets_divergent.avg,\
        dram__bytes_read.sum,\
        dram__bytes_write.sum,\
        l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
        lts__t_bytes.sum \
        --csv \
        --target-processes all \
        $CUTLASS_PROFILER \
        --kernels=$kernel \
        --m=$m --n=$n --k=$k \
        > $output_file 2>/dev/null

    # Process metrics and return in correct format
    if [ -f "$output_file" ]; then
        awk -F',' '
        NR>1 {
            # Calculate derived metrics
            warps_active = $2
            ipc = $4
            alu_utilization = $5
            l1_hit_rate = ($6 / ($6 + $7)) * 100
            l2_hit_rate = $8
            arith_inst = $9 + $10 + $11
            integer_inst = $12
            control_inst = $13
            memory_inst = $14
            branch_divergence = ($16/$15) * 100
            memory_throughput = ($17 + $18) / (1024 * 1024 * 1024)  # Convert to GB/s
            shared_mem_util = $19 / $20 * 100

            printf "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%.2f,%.2f",
                warps_active,
                ipc,
                alu_utilization,
                l1_hit_rate,
                l2_hit_rate,
                memory_throughput,
                arith_inst,
                integer_inst,
                control_inst,
                memory_inst,
                branch_divergence,
                shared_mem_util
        }' "$output_file"
    else
        echo "0,0,0,0,0,0,0,0,0,0,0,0"
    fi
}

# Function to collect GPU metrics
collect_gpu_metrics() {
    local duration=$1
    local output_file="$OUTPUT_DIR/gpu_metrics.tmp"

    # Start collecting GPU metrics
    nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,\
clocks.sm,clocks.mem,temperature.gpu,\
memory.used,memory.total,\
enforced.power.limit \
               --format=csv,noheader,nounits \
               -l 1 -f "$output_file" &
    local NVIDIA_SMI_PID=$!

    sleep $duration
    kill $NVIDIA_SMI_PID 2>/dev/null || true

    if [ -f "$output_file" ]; then
        awk -F',' '{
            power += $1
            gpu_util += $2
            mem_util += $3
            sm_clock += $4
            mem_clock += $5
            temp += $6
            count++
        } END {
            if (count > 0) {
                printf "%.2f,%.1f,%.1f,%d,%d,%.1f",
                    power/count,
                    gpu_util/count,
                    mem_util/count,
                    sm_clock/count,
                    mem_clock/count,
                    temp/count
            } else {
                print "0,0,0,0,0,0"
            }
        }' "$output_file"
        rm "$output_file"
    else
        echo "0,0,0,0,0,0"
    fi
}

# Function to calculate theoretical peak performance
calculate_tflops() {
    local runtime=$1
    local m=$2
    local n=$3
    local k=$4
    local kernel=$5
    
    # Calculate FLOPs
    local flops=$((2 * m * n * k))
    
    # Convert runtime to seconds
    local runtime_seconds=$(echo "$runtime / 1000" | bc -l)
    
    # Calculate TFLOPS
    local tflops=$(echo "scale=3; $flops / ($runtime_seconds * 1000000000000)" | bc -l)
    echo $tflops
}

# Function to run profiling for a specific configuration
run_profiling() {
    local m=$1
    local n=$2
    local k=$3
    local kernel=$4
    local layout=$5
    local block_size=$6
    local stages=$7
    local epilogue=$8
    local alpha=$9
    local beta=${10}

    # Skip invalid configurations
    if ! check_valid_config $m $n $k "$kernel"; then
        return
    fi

    echo "Testing M=$m N=$n K=$k kernel=$kernel layout=$layout block=$block_size stages=$stages epilogue=$epilogue alpha=$alpha beta=$beta"

    local temp_output="$OUTPUT_DIR/temp_${m}_${n}_${k}_${kernel// /_}.txt"
    local metrics_output="$OUTPUT_DIR/metrics_${m}_${n}_${k}_${kernel// /_}.txt"

    # Run CUTLASS profiler
    $CUTLASS_PROFILER --kernels=$kernel \
                      --m=$m --n=$n --k=$k \
                      --alpha=$alpha --beta=$beta \
                      --layout=$layout \
                      --block-size=$block_size \
                      --stages=$stages \
                      --epilogue=$epilogue \
                      --warmup-iterations=5 \
                      --profiling-iterations=100 \
                      --providers=cutlass \
                      --verification-enabled=false \
                      > "$temp_output" 2>/dev/null

    if [ ! -f "$temp_output" ]; then
        echo "Warning: Profiler failed for configuration"
        return
    fi

    # Extract basic metrics
    local runtime=$(grep "Runtime" "$temp_output" | awk '{print $2}')
    local gpu_time=$(grep "GPU Time" "$temp_output" | awk '{print $2}')
    local cpu_time=$(grep "CPU Time" "$temp_output" | awk '{print $2}')
    
    # Collect detailed metrics
    local ncu_metrics=$(collect_ncu_metrics "$kernel" "$m" "$n" "$k")
    local gpu_metrics=$(collect_gpu_metrics 5)
    
    # Extract kernel launch information
    local grid_size=$(grep "Grid:" "$temp_output" | awk '{print $2}')
    local block_size_actual=$(grep "Block:" "$temp_output" | awk '{print $2}')
    local shared_mem=$(grep "Shared Memory:" "$temp_output" | awk '{print $3}')
    local registers=$(grep "Registers:" "$temp_output" | awk '{print $2}')

    # Calculate power and performance metrics
    local power=$(echo "$gpu_metrics" | cut -d',' -f1)
    local energy=$(echo "$power * $runtime / 1000" | bc -l)
    local tflops=$(calculate_tflops "$runtime" "$m" "$n" "$k" "$kernel")

    # Write results to file
    echo "$(date +%s),$m,$n,$k,$kernel,$layout,$block_size,$stages,$epilogue,$alpha,$beta,\
$runtime,$gpu_time,$cpu_time,\
$ncu_metrics,\
$power,$energy,\
$grid_size,$block_size_actual,\
$shared_mem,$registers,\
$gpu_metrics,\
$tflops" >> "$RESULTS_FILE"

    # Cleanup temporary files
    rm -f "$temp_output"
}

# Function to generate per-kernel summary statistics
generate_kernel_summary() {
    local summary_file="$OUTPUT_DIR/kernel_summary.csv"
    echo "Kernel,Avg_Runtime_ms,Min_Runtime_ms,Max_Runtime_ms,Avg_TFLOPS,Peak_TFLOPS,Avg_Power_W,Best_Config" > "$summary_file"

    awk -F',' '
    NR>1 {
        kernel=$5
        runtime=$12
        power=$20
        tflops=$NF
        config=sprintf("M=%s,N=%s,K=%s,Layout=%s,Block=%s", $2, $3, $4, $6, $7)
        
        count[kernel]++
        sum_runtime[kernel] += runtime
        sum_power[kernel] += power
        sum_tflops[kernel] += tflops
        
        if (runtime < min_runtime[kernel] || min_runtime[kernel] == "") {
            min_runtime[kernel] = runtime
            best_config[kernel] = config
        }
        if (runtime > max_runtime[kernel]) {
            max_runtime[kernel] = runtime
        }
        if (tflops > peak_tflops[kernel]) {
            peak_tflops[kernel] = tflops
        }
    }
    END {
        for (kernel in count) {
            printf "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s\n",
                kernel,
                sum_runtime[kernel]/count[kernel],
                min_runtime[kernel],
                max_runtime[kernel],
                sum_tflops[kernel]/count[kernel],
                peak_tflops[kernel],
                sum_power[kernel]/count[kernel],
                best_config[kernel]
        }
    }' "$RESULTS_FILE" >> "$summary_file"
}

# Function to generate size-based summary statistics
generate_size_summary() {
    local summary_file="$OUTPUT_DIR/size_summary.csv"
    echo "Size,Avg_Runtime_ms,Best_Runtime_ms,Avg_TFLOPS,Peak_TFLOPS,Best_Kernel" > "$summary_file"

    awk -F',' '
    NR>1 {
        size=sprintf("%sx%sx%s", $2, $3, $4)
        kernel=$5
        runtime=$12
        tflops=$NF
        
        count[size]++
        sum_runtime[size] += runtime
        sum_tflops[size] += tflops
        
        if (runtime < min_runtime[size] || min_runtime[size] == "") {
            min_runtime[size] = runtime
            best_kernel[size] = kernel
        }
        if (tflops > peak_tflops[size]) {
            peak_tflops[size] = tflops
        }
    }
    END {
        for (size in count) {
            printf "%s,%.2f,%.2f,%.2f,%.2f,%s\n",
                size,
                sum_runtime[size]/count[size],
                min_runtime[size],
                sum_tflops[size]/count[size],
                peak_tflops[size],
                best_kernel[size]
        }
    }' "$RESULTS_FILE" | sort -t'x' -k1,1n -k2,2n -k3,3n >> "$summary_file"
}

# Main execution
echo "Starting extensive CUTLASS profiling..."

# Check for required tools
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

if ! command -v ncu &> /dev/null; then
    echo "Error: NSight Compute (ncu) not found. Please install CUDA toolkit."
    exit 1
fi

if [ -z "$CUTLASS_PROFILER" ]; then
    echo "Error: CUTLASS_PROFILER environment variable not set"
    exit 1
fi

# Log hardware information
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,compute_mode \
           --format=csv > "$OUTPUT_DIR/hardware_info.csv"

# Calculate total valid configurations
echo "Calculating total configurations..."
TOTAL_CONFIGS=0
for m in "${M_SIZES[@]}"; do
    for n in "${N_SIZES[@]}"; do
        for k in "${K_SIZES[@]}"; do
            for kernel in "${KERNELS[@]}"; do
                for layout in "${LAYOUTS[@]}"; do
                    for block_size in "${BLOCK_SIZES[@]}"; do
                        for stages in "${STAGES[@]}"; do
                            for epilogue in "${EPILOGUES[@]}"; do
                                for scalar in "${SCALARS[@]}"; do
                                    if check_valid_config $m $n $k "$kernel"; then
                                        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Total valid configurations to test: $TOTAL_CONFIGS"

# Main testing loop with progress tracking
CURRENT_CONFIG=0
START_TIME=$(date +%s)

for m in "${M_SIZES[@]}"; do
    for n in "${N_SIZES[@]}"; do
        for k in "${K_SIZES[@]}"; do
            for kernel in "${KERNELS[@]}"; do
                for layout in "${LAYOUTS[@]}"; do
                    for block_size in "${BLOCK_SIZES[@]}"; do
                        for stages in "${STAGES[@]}"; do
                            for epilogue in "${EPILOGUES[@]}"; do
                                for scalar in "${SCALARS[@]}"; do
                                    IFS=',' read alpha beta <<< "$scalar"
                                    
                                    if check_valid_config $m $n $k "$kernel"; then
                                        CURRENT_CONFIG=$((CURRENT_CONFIG + 1))
                                        ELAPSED=$(($(date +%s) - START_TIME))
                                        ETA=$(( (ELAPSED * (TOTAL_CONFIGS - CURRENT_CONFIG)) / CURRENT_CONFIG ))
                                        
                                        printf "Progress: %d/%d (%d%%) - Elapsed: %dh:%dm:%ds - ETA: %dh:%dm:%ds\r" \
                                            $CURRENT_CONFIG $TOTAL_CONFIGS $((CURRENT_CONFIG * 100 / TOTAL_CONFIGS)) \
                                            $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)) \
                                            $((ETA/3600)) $((ETA%3600/60)) $((ETA%60))
                                        
                                        run_profiling $m $n $k "$kernel" "$layout" "$block_size" "$stages" "$epilogue" "$alpha" "$beta"
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo -e "\nGenerating summary reports..."
generate_kernel_summary
generate_size_summary

# Create results archive
echo "Creating results archive..."
tar -czf "${OUTPUT_DIR}.tar.gz" "$OUTPUT_DIR"

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo "Profiling complete!"
echo "Total time: ${TOTAL_TIME}s"
echo "Results saved in ${OUTPUT_DIR}.tar.gz"
echo "Configurations tested: $CURRENT_CONFIG / $TOTAL_CONFIGS"
    
