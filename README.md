# CUDA vector addition

This is a simple example of CUDA code in C/C++ that demonstrates how to use CUDA to add two vectors. This code allocates memory on both the CPU (host) and GPU, initializes the input vectors, copies the input vectors to the GPU, performs the vector addition on the GPU using a CUDA kernel, copies the result back to the CPU, and verifies that the output is correct.

## Usage

1. Compile the code with NVCC:

```sh
nvcc vector_add.cu -o vector_add
```

2. This command compiles the CUDA code into an executable named `vector_add`.

Run the program:

```sh
./vector_add
```

This command executes the compiled program `vector_add`. The program should print the message "Vector addition completed successfully." if the vector addition was performed correctly.

## Explanation

The code consists of a single C++ file that defines a CUDA kernel and the main program. Here is an overview of what the code does:

1. Defines a CUDA kernel called `vector_add` that adds two vectors and stores the result in a third vector

2. Defines the main function, which allocates memory on both the CPU (host) and GPU, initializes the input vectors, copies the input vectors to the GPU, calls the `vector_add` kernel, copies the output vector back to the CPU, and verifies that the output is correct

3. Allocates memory on the CPU using new

4. Allocates memory on the GPU using cudaMalloc

5. Copies the input vectors from the CPU to the GPU using cudaMemcpy

6. Defines the number of threads and blocks to use in the `vector_add` kernel

7. Calls the `vector_add` kernel using `<<<...>>>` notation

8. Copies the output vector from the GPU to the CPU using cudaMemcpy

9. Verifies that the output vector is correct by comparing it to the expected output

10. Frees the memory allocated on both the CPU and GPU

## Dev

To compile and run the code, you will need:

- C/C++ compiler, such as GCC or Clang.

- CUDA toolkit, which includes the CUDA library and the NVIDIA CUDA compiler (NVCC).

- NVIDIA GPU with CUDA support.


[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
[CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

