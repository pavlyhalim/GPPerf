# GPPerf
Understanding GEMM Performance and Energy on NVIDIA Ada Lovelace: A Machine Learning-Based Analytical Approach


Our goal is to predict the runtime and energy comsumption of SGEMM on NVIDIA GPU given different matrix sizes, block sizes, and tile sizes. We implemented a naive tiled matrix multiplication kernel and used it to gather the data for different tile sizes. We also used cutlass to gather the data with more advanced configurations. We then trained a model to predict the performance and energy consumption of SGEMM given different configurations.

[Docs](https://docs.google.com/document/d/1DSFfXMxL58vp3B_QiVPaZDEwmR09cprOR4Ce817I7bQ/edit?tab=t.0)

## Installation
Our analysis used cuda5 node.
**For the Ada Lovelace GPU on cuda5, we have already built the `cutlass_profiler` binary. You can directly use it to profile the kernels, and skip the installation process.**
If you want to build the `cutlass_profiler` binary yourself, you can follow the instructions below.
Clone the [cutlass repo](https://github.com/NVIDIA/cutlass.git). Then install cutlass by running the following commands (this works for Ada Lovelace architecture. For other nodes, you need to change the architecture flag):
```bash
git clone https://github.com/NVIDIA/cutlass.git

export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

mkdir build && cd build

cmake .. -DCUTLASS_NVCC_ARCHS=89             # compiles for NVIDIA Ada Lovelace GPU architecture

make cutlass_profiler -j12
```
The commands are adapted from the cutlass [quick start guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md).

## Usage
### Profiling using the cutlass profiler
```bash
export CUTLASS_PROFILER="YOUR_CUTLASS_DIRECTORY/build/tools/profiler/cutlass_profiler" #change YOUR_CUTLASS_DIRECTORY to your path to the cutlass profiler. For example, /home/username/cutlass/build/tools/profiler/cutlass_profiler

bash prof.sh
```

The results will be saved in the `cutlass_profiling_20241118_191220/results.csv` file. 
The results are then cleaned and reformatted in the following way.
| Timestamp   | M   | N   | K   | Kernel Name                             | Layout | Blocksize1 | Blocksize2 | Blocksize3 | Stage | Combination Type   | Alpha | Beta | Runtime  | Power  | Clock SM | Clocks Meme | Temp | GPU Util | Mem Util | GPU Name                  | Version | Clocks Max Mem | Graphics Max Clock | Power Limit | State   | Total Memory | Free Memory | Used Memory | GPU Util1 | Mem Util2 | Kernel Name                             | Arithmetic Intensity | Uses Shared Memory | Computation Pattern | Energy    | TFlops   |
|-------------|-----|-----|-----|-----------------------------------------|--------|------------|------------|------------|-------|--------------------|-------|------|----------|--------|----------|--------------|------|----------|----------|---------------------------|---------|----------------|--------------------|--------------|---------|--------------|-------------|-------------|-----------|-----------|-----------------------------------------|----------------------|-------------------|---------------------|-----------|----------|
| 1731372844  | 512 | 512 | 512 | cutlass_simt_sgemm_128x128_8x2_nn_align1 | nn     | 64         | 64         | 32         | 2     | linear_combination | 1     | 0    | 0.04352  | 70.17  | 2760     | 10251        | 47   | 100%     | 0%       | NVIDIA GeForce RTX 4070   | 8.9     | 10501 MHz      | 3105 MHz          | 200.00 W     | Default | 12282 MiB    | 11827 MiB   | 176 MiB     | 100%      | 0%        | cutlass_simt_sgemm_128x128_8x2_nn_align1 | 85.33               | 0                 | GEMM                | 87902.151 | 6168.094 |

Then run the model to predict the performance and energy of the kernel. 
```bash
python model.py
```

### Profiling using our tiled matrix multiplication kernel
```bash
nvcc matmul.cu -lcublas -o matmul.o
./matmul.o 512 512 512 8 #M N K TILE_SIZE
```
Sample output:
```
Running tiled matrix multiplication with M=512, N=512, K=512, TILE_SIZE=8
Tiled MM result sample: 1024 1024 1024 1024 1024
cuBLAS result sample: 1024 1024 1024 1024 1024
Execution time: 5.63299 ms
```
The "Tiled MM result sample" are results from our kernel, and the "cuBLAS result sample" are results from cuBLAS, which serves as a verification of the correctness of our kernel. The "Execution time" is the runtime of our kernel. We used the `cudaEventRecord` API to measure the runtime.

You can use the script `matmul_runtime_prof.sh` to gather the runtime data, and `matmul_power_prof.sh` to gather the power data.
The runtime data is stored in "execution_times.csv" with the following format:
```
M,N,K,Tile Size,Execution Time
512,512,512,8,6.56141
...
```
The power data is stored in "power_usage_results.csv" with the following format:
```
M,N,K,Tile Size,Average Power Usage (W)
512,512,512,1,32.038
...
```
