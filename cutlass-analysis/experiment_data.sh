#!/bin/bash

# Configuration arrays for matrix dimensions
M_SIZES=(512 1024 2048 128256)
N_SIZES=(512 1024 2048)
K_SIZES=(512 1024 2048)

# CUTLASS kernels with different configurations
KERNELS=(
    "cutlass_simt_sgemm_128x128_8x2_nn_align1"
    "cutlass_simt_sgemm_128x128_8x2_nt_align1"
    "cutlass_simt_sgemm_128x128_8x2_tn_align1"
    "cutlass_simt_sgemm_128x128_8x2_tt_align1"
)

# Block size configurations
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
STAGES=(2)

# Epilogue operations
EPILOGUES=("linear_combination")

# Alpha/Beta combinations
SCALARS=(
    "1,0"
    "1,1"
    "0.5,0.5"
    "2,0"
)

# Matrix layouts
LAYOUTS=("nn" "nt" "tn" "tt")

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Validate environment and dependencies
validate_environment() {
    log_message "Validating environment..."
    
    # Check CUDA toolkit
    if ! command -v nvcc &> /dev/null; then
        log_message "Error: CUDA toolkit not found. Please load CUDA module."
        return 1
    fi
    
    # Check required tools
    local required_tools=(nvidia-smi ncu bc awk)
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_message "Error: $tool not found. Please ensure all required tools are installed."
            return 1
        fi
    done
    
    # Verify GPU access
    if ! nvidia-smi &> /dev/null; then
        log_message "Error: Cannot access GPU. Please check GPU availability."
        return 1
    fi
    
    # Check CUTLASS profiler
    if [ ! -f "$CUTLASS_PROFILER" ]; then
        log_message "Error: CUTLASS profiler not found at $CUTLASS_PROFILER"
        return 1
    fi
    
    return 0
}

# Collect device specifications
collect_device_specs() {
    local output_file="$1"
    log_message "Collecting device specifications..."
    
    # Create a temporary file for raw device info
    local temp_device_info="${output_file}.tmp"
    
    # Query basic device properties with correct field names
    nvidia-smi --query-gpu=\
name,\
compute_cap,\
clocks.max.memory,\
clocks.max.graphics,\
power.limit,\
temperature.gpu,\
compute_mode,\
memory.total,\
memory.free,\
memory.used,\
utilization.gpu,\
utilization.memory \
        --format=csv,noheader > "$temp_device_info"
    
    # Calculate memory bandwidth (if needed)
    local memory_clock=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader | head -n1)
    local bus_width=$(nvidia-smi --query-gpu=pcie.link.width.max --format=csv,noheader | head -n1)
    if [[ $memory_clock =~ ^[0-9]+$ ]] && [[ $bus_width =~ ^[0-9]+$ ]]; then
        local memory_bandwidth=$(echo "scale=2; $memory_clock * $bus_width / 8" | bc)
        echo "memory_bandwidth,$memory_bandwidth" >> "$output_file"
    fi
    
    # Process the collected data
    cat "$temp_device_info" > "$output_file"
    
    # Cleanup
    rm -f "$temp_device_info"
}

# Collect NCU metrics
collect_ncu_metrics() {
    local kernel="$1"
    local m="$2"
    local n="$3"
    local k="$4"
    local output_file="$5"

    log_message "Collecting NCU metrics for kernel $kernel (M=$m, N=$n, K=$k)..."

    ncu --metrics \
"sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_eligible.avg.pct_of_peak_sustained_active,\
sm__inst_executed.avg.per_cycle_active,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
dram__bytes.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__t_bytes.sum,\
lts__t_bytes.sum,\
sm__warps_launched.sum,\
sm__cycles_elapsed.sum,\
sm__cycles_active.sum,\
sm__cycles_stalled.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
dram__cycles_elapsed.avg,\
dram__cycles_active.avg,\
l2_atomic_throughput,\
l2_tex_read_throughput,\
l2_tex_write_throughput,\
shared_load_throughput,\
shared_store_throughput,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_alu.avg.per_cycle_active,\
sm__inst_executed_pipe_fma.avg.per_cycle_active,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
gpu__time_duration.sum" \
        --csv \
        "$CUTLASS_PROFILER" \
        --kernels="$kernel" \
        --m="$m" --n="$n" --k="$k" \
        > "$output_file" 2>/dev/null

    # Check if metrics collection was successful
    if [ ! -f "$output_file" ] || [ ! -s "$output_file" ]; then
        log_message "Warning: Failed to collect NCU metrics"
        return 1
    fi
}

# Collect power metrics
collect_power_metrics() {
    local duration=$1
    local output_file="$2"
    
    log_message "Collecting power metrics for ${duration}s..."
    
    nvidia-smi \
        --query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits \
        -l 1 > "$output_file" &
    local NVIDIA_SMI_PID=$!
    
    sleep "$duration"
    kill $NVIDIA_SMI_PID 2>/dev/null
    
    # Check if power metrics collection was successful
    if [ ! -f "$output_file" ] || [ ! -s "$output_file" ]; then
        log_message "Warning: Failed to collect power metrics"
        return 1
    fi
}

# Analyze kernel characteristics
analyze_kernel_characteristics() {
    local kernel="$1"
    local m="$2"
    local n="$3"
    local k="$4"
    local output_file="$5"
    
    log_message "Analyzing kernel characteristics..."
    
    # Calculate arithmetic intensity for GEMM
    local flops=$((2 * m * n * k))
    local bytes_accessed=$(( (m * k + k * n + m * n) * 4 ))  # For single precision
    local arithmetic_intensity=$(echo "scale=2; $flops / $bytes_accessed" | bc)
    
    # Determine if kernel uses shared memory
    local uses_shared_memory=0
    if [[ "$kernel" == *"shared"* ]]; then
        uses_shared_memory=1
    fi
    
    # Analyze computation pattern
    local computation_pattern="GEMM"
    if [[ "$kernel" == *"tensor"* ]]; then
        computation_pattern="TENSOR_CORE"
    fi
    
    # Save characteristics
    {
        echo "kernel_name,$kernel"
        echo "arithmetic_intensity,$arithmetic_intensity"
        echo "uses_shared_memory,$uses_shared_memory"
        echo "computation_pattern,$computation_pattern"
        echo "problem_size_m,$m"
        echo "problem_size_n,$n"
        echo "problem_size_k,$k"
    } > "$output_file"
}

# Run profiling for a single configuration
run_profiling() {
    local m=$1
    local n=$2
    local k=$3
    local kernel="$4"
    local layout="$5"
    local block_size="$6"
    local stages=$7
    local epilogue="$8"
    local alpha=$9
    local beta=${10}
    
    log_message "Testing configuration: M=$m N=$n K=$k kernel=$kernel layout=$layout block=$block_size stages=$stages epilogue=$epilogue alpha=$alpha beta=$beta"
    
    # Create temporary directory for this run
    local temp_dir="$OUTPUT_DIR/temp/${m}_${n}_${k}_${kernel// /_}"
    mkdir -p "$temp_dir"
    
    # Define output files
    local perf_output="${temp_dir}/perf.txt"
    local metrics_output="${temp_dir}/metrics.csv"
    local power_output="${temp_dir}/power.csv"
    local device_specs="${temp_dir}/device_specs.csv"
    local kernel_chars="${temp_dir}/kernel_chars.csv"
    
    # Collect device specifications
    collect_device_specs "$device_specs"
    
    # Analyze kernel characteristics
    analyze_kernel_characteristics "$kernel" "$m" "$n" "$k" "$kernel_chars"
    
    # Run CUTLASS profiler
    timeout 30s $CUTLASS_PROFILER \
        --kernels="$kernel" \
        --m=$m --n=$n --k=$k \
        --alpha=$alpha --beta=$beta \
        --layout="$layout" \
        --block-size="$block_size" \
        --stages=$stages \
        --epilogue="$epilogue" \
        --warmup-iterations=3 \
        --profiling-iterations=10 \
        --providers=cutlass \
        --verification-enabled=false \
        > "$perf_output" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        log_message "Warning: Profiler timeout or error"
        return 1
    fi
    
    # Extract runtime
    local runtime=$(grep "Runtime:" "$perf_output" | awk '{print $2}')
    if [[ ! $runtime =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        log_message "Warning: Invalid runtime value"
        return 1
    fi
    
    # Collect NCU metrics
    collect_ncu_metrics "$kernel" "$m" "$n" "$k" "$metrics_output"
    
    # Collect power metrics
    collect_power_metrics 2 "$power_output"
    
    # Process and combine all metrics
    local processed_metrics=$(awk -F, 'NR==2{print $0}' "$metrics_output")
    local power_metrics=$(awk -F, 'NR==1{print $0}' "$power_output")
    local device_info=$(cat "$device_specs")
    local kernel_info=$(cat "$kernel_chars")
    
    # Calculate derived metrics
    local flops=$((2 * m * n * k))
    local tflops=$(echo "scale=3; $flops / ($runtime * 1000000)" | bc)
    local energy=$(echo "scale=3; $(echo "$power_metrics" | cut -d',' -f1) * $runtime / 1000" | bc)
    local gflops_per_watt=$(echo "scale=3; $tflops * 1000 / $(echo "$power_metrics" | cut -d',' -f1)" | bc)
    
    # Write to results file
    echo "$(date +%s),$m,$n,$k,$kernel,$layout,$block_size,$stages,$epilogue,$alpha,$beta,\
$runtime,$processed_metrics,$power_metrics,$device_info,$kernel_info,$energy,$gflops_per_watt,$tflops" >> "$RESULTS_FILE"
    
    log_message "Runtime: ${runtime}ms, TFLOPS: $tflops, Energy: ${energy}J, GFLOPS/W: $gflops_per_watt"
    
    # Cleanup temporary files
    rm -rf "$temp_dir"
    return 0
}

# Initialize results file with header
initialize_results() 

{
        echo "Timestamp,M,N,K,Kernel,Layout,Block_Size,Stages,Epilogue,Alpha,Beta,Runtime_ms,"
        echo "SM_Occupancy,Warp_Execution_Efficiency,IPC,ALU_Pipe_Utilization,"
        echo "DRAM_Bytes_Total,DRAM_Bytes_Read,DRAM_Bytes_Write,L1_Bytes_Total,L2_Bytes_Total,"
        echo "Warps_Launched,Cycles_Total,Cycles_Active,Cycles_Stalled,"
        echo "L1_Hit_Rate,L2_Hit_Rate,DRAM_Cycles_Elapsed,DRAM_Cycles_Active,"
        echo "L2_Atomic_Throughput,L2_Read_Throughput,L2_Write_Throughput,"
        echo "Shared_Load_Throughput,Shared_Store_Throughput,"
        echo "Tensor_Instructions_Active_Pct,ALU_IPC,FMA_IPC,"
        echo "FADD_Instructions,FMUL_Instructions,FFMA_Instructions,"
        echo "Device_Name,Memory_Total,Memory_Clock,Graphics_Clock,Power_Limit,Temperature,"
        echo "Compute_Mode,Compute_Capability,Max_Threads_Per_SM,Max_Threads_Per_Block,"
        echo "Max_Shared_Memory_Per_Block,Max_Registers_Per_Block,SM_Count,Memory_Bandwidth,"
        echo "Kernel_Name,Arithmetic_Intensity,Uses_Shared_Memory,Computation_Pattern,"
        echo "Problem_Size_M,Problem_Size_N,Problem_Size_K,"
        echo "Power_Draw,SM_Clock,Mem_Clock,Temperature,GPU_Utilization,Memory_Utilization,"
        echo "Energy_Joules,GFLOPS_per_Watt,TFLOPS" | tr -d '\n' > "$RESULTS_FILE"
}

# Main execution function
main() {
    # Set CUTLASS profiler path
    if [ -z "$CUTLASS_PROFILER" ]; then
        CUTLASS_PROFILER="/home/poh2005/cutlass/build/tools/profiler/cutlass_profiler"
    fi
    
    # Validate environment
    if ! validate_environment; then
        log_message "Environment validation failed. Exiting..."
        exit 1
    fi
    
    # Create output directory with timestamp
    OUTPUT_DIR="cutlass_profiling_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    RESULTS_FILE="$OUTPUT_DIR/results.csv"
    
    # Initialize results file
    initialize_results
    
    # Calculate total configurations for progress tracking
    local total_configs=0
    for m in "${M_SIZES[@]}"; do
        for n in "${N_SIZES[@]}"; do
            for k in "${K_SIZES[@]}"; do
                for kernel in "${KERNELS[@]}"; do
                    for layout in "${LAYOUTS[@]}"; do
                        for block_size in "${BLOCK_SIZES[@]}"; do
                            for stages in "${STAGES[@]}"; do
                                for epilogue in "${EPILOGUES[@]}"; do
                                    for scalar in "${SCALARS[@]}"; do
                                        total_configs=$((total_configs + 1))
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
    
    log_message "Starting profiling of $total_configs configurations..."
    
    # Track progress and timing
    local current_config=0
    local start_time=$(date +%s)
    local last_update=0
    
    # Create a progress file
    local progress_file="$OUTPUT_DIR/progress.txt"
    
    # Main profiling loop
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
                                        
                                        current_config=$((current_config + 1))
                                        
                                        # Update progress at most once per second
                                        local current_time=$(date +%s)
                                        if [ $((current_time - last_update)) -ge 1 ]; then
                                            # Calculate progress statistics
                                            local elapsed=$((current_time - start_time))
                                            local eta=0
                                            if [ $current_config -gt 0 ]; then
                                                eta=$((elapsed * (total_configs - current_config) / current_config))
                                            fi
                                            
                                            # Format progress message
                                            printf "Progress: %d/%d (%d%%) - Elapsed: %02d:%02d:%02d - ETA: %02d:%02d:%02d\n" \
                                                $current_config $total_configs $((current_config * 100 / total_configs)) \
                                                $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)) \
                                                $((eta/3600)) $((eta%3600/60)) $((eta%60)) > "$progress_file"
                                            
                                            last_update=$current_time
                                        fi
                                        
                                        # Run profiling for current configuration
                                        if ! run_profiling $m $n $k "$kernel" "$layout" "$block_size" \
                                            $stages "$epilogue" "$alpha" "$beta"; then
                                            log_message "Warning: Profiling failed for configuration $current_config"
                                            # Log failed configuration
                                            echo "$m,$n,$k,$kernel,$layout,$block_size,$stages,$epilogue,$alpha,$beta" \
                                                >> "$OUTPUT_DIR/failed_configs.csv"
                                        fi
                                        
                                        # Optional: Add delay between runs to allow GPU to cool down
                                        sleep 1
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
    
    log_message "Profiling complete. Processing results..."
    
    # Process and analyze results
    process_results
    
    # Create results archive
    log_message "Creating results archive..."
    tar -czf "${OUTPUT_DIR}.tar.gz" "$OUTPUT_DIR"
    
    log_message "Results saved in ${OUTPUT_DIR}.tar.gz"
}

# Process and analyze the collected results
process_results() {
    log_message "Processing results and generating summary..."
    
    # Create summary directory
    local summary_dir="$OUTPUT_DIR/summary"
    mkdir -p "$summary_dir"
    
    # Generate basic statistics
    awk -F',' 'NR>1 {
        sum_runtime += $12
        sum_tflops += $(NF)
        sum_energy += $(NF-2)
        sum_efficiency += $(NF-1)
        count++
    }
    END {
        print "Average Runtime (ms)," sum_runtime/count
        print "Average TFLOPS," sum_tflops/count
        print "Average Energy (J)," sum_energy/count
        print "Average GFLOPS/W," sum_efficiency/count
    }' "$RESULTS_FILE" > "$summary_dir/basic_stats.csv"
    
    # Find best configurations for different metrics
    {
        echo "Metric,M,N,K,Kernel,Layout,Block_Size,Value"
        # Best TFLOPS
        awk -F',' 'NR>1 {
            if($NF > max_tflops) {
                max_tflops = $NF
                best_tflops = $2","$3","$4","$5","$6","$7","$NF
            }
        }
        END {
            print "Best_TFLOPS," best_tflops
        }' "$RESULTS_FILE"
        
        # Best Energy Efficiency
        awk -F',' 'NR>1 {
            if($(NF-1) > max_efficiency) {
                max_efficiency = $(NF-1)
                best_efficiency = $2","$3","$4","$5","$6","$7","$(NF-1)
            }
        }
        END {
            print "Best_GFLOPS_per_Watt," best_efficiency
        }' "$RESULTS_FILE"
        
        # Lowest Runtime
        awk -F',' 'NR>1 {
            if(NR==2 || $12 < min_runtime) {
                min_runtime = $12
                best_runtime = $2","$3","$4","$5","$6","$7","$12
            }
        }
        END {
            print "Best_Runtime," best_runtime
        }' "$RESULTS_FILE"
    } > "$summary_dir/best_configs.csv"
    
    # Generate summary plots if gnuplot is available
    if command -v gnuplot &> /dev/null; then
        generate_plots "$summary_dir"
    fi
    
    log_message "Results processing complete. Summary available in $summary_dir"
}

# Generate visualization plots
generate_plots() {
    local summary_dir="$1"
    
    # Plot TFLOPS vs Problem Size
    gnuplot <<EOF
    set terminal png size 1200,800
    set output "$summary_dir/tflops_vs_size.png"
    set title "TFLOPS vs Problem Size"
    set xlabel "Problem Size (M×N×K)"
    set ylabel "TFLOPS"
    set grid
    plot "$RESULTS_FILE" using (\$2*\$3*\$4):(\$NF) with points title "Performance"
EOF
    
    # Plot Energy Efficiency vs Problem Size
    gnuplot <<EOF
    set terminal png size 1200,800
    set output "$summary_dir/efficiency_vs_size.png"
    set title "Energy Efficiency vs Problem Size"
    set xlabel "Problem Size (M×N×K)"
    set ylabel "GFLOPS/Watt"
    set grid
    plot "$RESULTS_FILE" using (\$2*\$3*\$4):($(NF-1)) with points title "Energy Efficiency"
EOF
}

# Handle interrupts gracefully
trap 'echo -e "\nScript interrupted. Cleaning up..."; exit 1' INT TERM

# Run main function with command line arguments
main "$@"