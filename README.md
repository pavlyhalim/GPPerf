# GPPerf
Energy-Aware GPU Performance Prediction and Optimization Framework

[Docs](https://docs.google.com/document/d/1DSFfXMxL58vp3B_QiVPaZDEwmR09cprOR4Ce817I7bQ/edit?tab=t.0)

## Installation
clone the [cutlass repo](https://github.com/NVIDIA/cutlass.git). Then install cutlass by running the following commands (this works on cuda5 node. For other nodes, you may need to change the architecture flag):
```bash
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
./matmul.o 512 512 512 16 # M N K TILE_SIZE
```