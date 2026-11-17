# Experiment 5: Neural Network Layers in CUDA

## Overview

This project implements basic neural network layers using custom CUDA kernels. It demonstrates how neural network operations can be executed on an NVIDIA GPU and compares their performance with a CPU implementation.

The experiment implements matrix multiplication with bias, ReLU activation, and sigmoid activation. It also performs a two-layer neural network forward pass, validates CPU and GPU outputs, and measures execution time and GPU speedup.

## Aim

To implement and evaluate basic neural network layers using custom CUDA kernels, execute the same forward pass on the CPU and GPU, validate numerical correctness, measure execution time, and calculate GPU speedup.

## Technologies Used

- **Programming Language:** C++ with CUDA
- **GPU Programming:** NVIDIA CUDA
- **Compiler:** NVIDIA NVCC
- **Development Environment:** Windows 10/11
- **Build Tools:** Visual Studio 2022 Build Tools
- **GPU:** NVIDIA CUDA-capable GPU

## Neural Network Architecture

The experiment uses a two-layer feed-forward neural network.

### Forward Propagation

**Layer 1:**

```text
Z1 = X × W1 + b1
A1 = ReLU(Z1)
```

**Layer 2:**

```text
Z2 = A1 × W2 + b2
Y  = Sigmoid(Z2)
```

### Network Configuration

| Parameter | Value |
|---|---:|
| Batch Size | 256 |
| Input Size | 128 |
| Hidden Layer Size | 64 |
| Output Size | 10 |

This experiment implements inference (forward propagation) only. It does not include model training, loss calculation, an optimizer, or backpropagation.

## CUDA Kernels Implemented

### 1. Matrix Multiplication with Bias

Computes the matrix multiplication of two matrices and adds a bias to each output column.

```text
C[row, col] = Σ(A[row, k] × B[k, col]) + bias[col]
```

Each CUDA thread calculates one output matrix element.

### 2. ReLU Activation

ReLU replaces negative values with zero while keeping positive values unchanged.

**Formula:**

```text
ReLU(x) = max(0, x)
```

### 3. Sigmoid Activation

Sigmoid maps input values into a range between 0 and 1.

**Formula:**

```text
Sigmoid(x) = 1 / (1 + exp(-x))
```

Each CUDA thread processes one array element for the activation kernels.

## CPU and GPU Implementation

The project executes the same neural network forward pass on both CPU and GPU.

**CPU execution:**
- Uses ordinary C++ loops and functions.
- Performs matrix multiplication and activation operations sequentially.
- Measures average CPU computation time.

**GPU execution:**
- Allocates memory on the GPU.
- Transfers input data and weights from host memory to device memory.
- Launches CUDA kernels in sequence.
- Copies the final output back to the CPU.
- Measures kernel-only and end-to-end execution time.

## Correctness Validation

The project tests the three CUDA kernels individually:

- Matrix multiplication with bias
- ReLU activation
- Sigmoid activation

It then compares the CPU and GPU forward-pass outputs using the maximum absolute difference.

The expected validation output is:

```text
Matrix multiplication + bias: PASSED
ReLU: PASSED
Sigmoid: PASSED

CPU and GPU outputs: MATCH (within tolerance)
```

## Performance Evaluation

The experiment measures:

1. Average CPU computation time.
2. Average GPU kernel-only execution time.
3. Average GPU end-to-end execution time.
4. Kernel-only speedup.
5. End-to-end speedup.

### Speedup Formulas

```text
Kernel-only Speedup =
CPU Time / GPU Kernel-only Time
```

```text
End-to-end Speedup =
CPU Time / GPU End-to-end Time
```

### Reference Results

The following values are taken from the lab manual's demonstrated workstation. They are reference values, not guaranteed results on every computer.

| Metric | Reference Value |
|---|---:|
| CPU Computation Time | 0.526860 ms |
| GPU Kernel-only Time | 0.026517 ms |
| GPU End-to-end Time | 0.066010 ms |
| Kernel-only Speedup | 19.868913x |
| End-to-end Speedup | 7.981518x |
| Maximum Absolute Difference | 0.000000 |

**Note:** Actual execution times and speedup depend on the GPU, CPU, CUDA version, drivers, and system configuration.

## Prerequisites

Before running the project, install:

1. NVIDIA GPU driver.
2. NVIDIA CUDA Toolkit with `nvcc`.
3. Visual Studio 2022 Build Tools.
4. Desktop development with C++ workload.
5. MSVC v143 x64/x86 build tools.
6. Windows 10/11 SDK.
7. An NVIDIA CUDA-capable GPU.

### Verify Installation

Open PowerShell and execute:

```powershell
nvidia-smi
```

Check CUDA compiler:

```powershell
nvcc --version
```

Check whether NVCC is available:

```powershell
where.exe nvcc
```

Open the **x64 Native Tools Command Prompt for VS 2022** or an appropriate Visual Studio Developer Command Prompt and check:

```cmd
where cl
```

## Compilation

Navigate to the project directory:

```powershell
cd path\to\Experiment5_Neural_Network_Layers_CUDA
```

Compile the CUDA program:

```powershell
nvcc -O2 neural_network.cu -o neural_network.exe
```

If compilation is successful, the executable `neural_network.exe` will be generated.

## Execution

Run the program:

```powershell
.\neural_network.exe
```

The program performs individual kernel tests, executes the CPU and GPU forward passes, validates numerical correctness, and displays performance measurements.

## Expected Output

The output should contain:

```text
EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA

GPU detected: [Your GPU Name]

Testing individual CUDA kernels...

Matrix multiplication + bias: PASSED
ReLU: PASSED
Sigmoid: PASSED

Forward-pass dimensions:
Input: 256 x 128
Hidden layer: 256 x 64
Output: 256 x 10

Average CPU compute time: [Measured Value] ms
Average GPU kernel-only time: [Measured Value] ms
Average GPU end-to-end time: [Measured Value] ms

Kernel-only speedup: [Calculated Value]x
End-to-end speedup: [Calculated Value]x

Maximum absolute output difference: [Measured Value]

CPU and GPU outputs: MATCH (within tolerance)
```

The GPU name and timing values will vary depending on the machine.

## Project Structure

```text
Experiment5_Neural_Network_Layers_CUDA/
│
├── neural_network.cu
├── README.md
├── build.bat
├── run.bat
├── .gitignore
│
└── results/
    └── reference-results.md
```

- `neural_network.cu` – CUDA kernels, CPU implementation, GPU forward pass, and performance measurements.
- `README.md` – Project documentation.
- `build.bat` – Script to compile the CUDA program.
- `run.bat` – Script to execute the compiled program.
- `.gitignore` – Excludes generated binaries and temporary build files.
- `results/reference-results.md` – Reference performance measurements from the lab manual.

## Learning Outcomes

After completing this experiment, the following concepts are demonstrated:

- Understanding CUDA kernels, blocks, threads, and grid dimensions.
- Implementing matrix multiplication with bias using CUDA.
- Implementing ReLU and sigmoid activation functions.
- Allocating and transferring memory between CPU and GPU.
- Executing a neural network forward pass on the GPU.
- Comparing CPU and GPU numerical outputs.
- Measuring GPU execution time and calculating speedup.

## Conclusion

This experiment demonstrates the implementation of basic neural network layers using custom CUDA kernels. By comparing CPU and GPU implementations, it illustrates how parallel processing can accelerate neural network computations.

The correctness checks verify that the GPU implementation produces results matching the CPU implementation within the specified tolerance. Performance measurements help distinguish kernel execution time from total execution time, including memory transfers.

---

**Experiment:** 5 – Neural Network Layers in CUDA  
**Subject:** Cloud / GPU Computing Laboratory  
**Platform:** Windows, NVIDIA CUDA Toolkit, Visual Studio 2022 Build Tools
