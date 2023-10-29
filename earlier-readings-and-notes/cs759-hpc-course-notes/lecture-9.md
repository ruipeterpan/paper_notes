# Lecture 9: GPU Memory Spaces

## Lecture Summary

* GPU computing: generalities
* GPU computing: execution configuration
* ~~GPU computing: scheduling execution~~

## Prerequisite: Parallelism

![Coarse Grain vs. Fine Grain Parallelism](<../../.gitbook/assets/Screen Shot 2021-02-26 at 5.44.27 PM.png>)

* Coarse grain parallelism: Good for CPUs
  * Few tasks
  * Tasks are heterogeneous
  * Tasks are in general complex, lots of control flow
  * Example: {Bake a cake, make coffee, watch lectures} at the same time
* Fine grain parallelism: Very good for GPUs, ok for CPUs
  * A lot, a lot of tasks
  * Tasks are basically identical
  * Tasks are in general pretty straightforward, lots of math, not much control flow
  * Example: Image processing (lots of pixels to deal with)

## GPU Computing

* GPGPU: General Purpose GPU Computing
  * Started in the early 2000s using graphics libraries
  * GPUs had high bandwidths
  * Data need to be moved into the GPU to process it (this may be a bottleneck!)
    * PCIe: 16-32 GB/s
    * NVLink: 5-12 times faster than PCIe 3
    * The tradeoff is worth it if the data transfer overhead is smaller than our gain
  * Idea: Use the GPU as a co-processor to handle big, parallel jobs
    * In the meanwhile, the CPU handles control of execution & corner tasks
* CUDA: Compute Unified Device Architecture, distributed by NVIDIA
  * Eliminated the graphics-constraints associated with GPGPU
  * Enables a general-purpose programming model
* GPUs:
  * Is a co-processor to the CPU/host
  * Has its own memory (device memory)
  * Runs many threads in parallel
  * The data parallel portion of an application runs on the devices as kernels executed in parallel by many threads
  * As compared to CPU threads:
    * GPUs threads are extremely lightweight
    * A GPU needs 1000s of threads for full efficiency
* Compute capability vs. CUDA version:
  * Compute capability: Refers to hardware
  * CUDA version: Refers to software that manages the hardware
* Compatibility issues
  * The CUDA driver API is backward, but not forward compatible
    * Code that works for CUDA 8.0 should work for 11.0, but not the other way around

![The CUDA execution model](<../../.gitbook/assets/Screen Shot 2021-02-26 at 6.00.17 PM.png>)

* CUDA host stream
  * The CUDA runtime places all calls that invoke the GPU in a stream (i.e., ordered collection) of calls
    * The stream is FIFO: In the picture above, Kernel1 is only called after Kernel0 finishes
  * Asynchronicity between host and device: The host continues execution right after launching a kernel
    * Synchronization can be forced
* Three opportunities for asynchronous:
  * The GPU and CPU work in async mode
  * The GPU has three engines that can work at the same time (copy-in, copy-out, execution)
  * Multiple GPUs can work at the same time on one host
* Language supported by CUDA
  * C/C++: [Check out this introduction by NVIDIA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

## CUDA: First Example

```
#include<cuda.h>
#include<iostream>

__global__voidsimpleKernel(int* data)
{
    //this adds a value to a variable stored in global memory
    data[threadIdx.x] += 2*(blockIdx.x+ threadIdx.x);
}

int main()
{
    const int numElems= 4;
    int hostArray[numElems], *devArray;
    
    //allocate memory on the device (GPU); zero out all entries in this device array 
    cudaMalloc((void**)&devArray, sizeof(int) * numElems);
    cudaMemset(devArray, 0, numElems* sizeof(int));
    
    //invoke GPU kernel, with one block that has four threads
    simpleKernel<<<1,numElems>>>(devArray);
    
    //bring the result back from the GPU into the hostArray
    cudaMemcpy(&hostArray, devArray, sizeof(int) * numElems, cudaMemcpyDeviceToHost);
    
    //print out the result to confirm that things are looking good 
    std::cout << "Values stored in hostArray: " << std::endl;
    for (int i = 0; i < numElems; i++)
        std::cout<< hostArray[i] << std::endl;
    
    //release the memory allocated on the GPU 
    cudaFree(devArray);
    return 0;
}
```

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 6.16.57 PM.png>)

## GPU Execution Configuration

* Nomenclature
  * Host: The CPU executing the "master" thread
  * Device: GPU card, connected to the host through a PCIe connection
  * The host instructs the device to execute kernels
  * Defining the execution configuration: The process in which the host tells the device how many threads should each execute kernels



```
__global__ void kernelFoo(...); // declaration

dim3 DimGrid(100, 50);        // 2D grid structure, w/ total of 5000 thread blocks 
dim3 DimBlock(4, 8, 8);       // 3D block structure, with 256 threads per block 

kernelFoo<<<DimGrid, DimBlock>>>(...arg list...);
```

* The concept of "block" is important since it represents the entity that gets executed by an SM (stream multiprocessor)
* Threads in each block:
  * The threads can be organized as a 3D structure (x, y, z)
  * Max x- or y- dimension of a block is 1024
  * Max z- dimension of a block is 64
  * Max # threads per block is 1024
* Threads and blocks have indices
* 3D layout:
  * Most of the time people use 1D
  * This simplifies memory addressing when processing multi-dimensional data
    * Handling matrices
    * Solving PDEs on 3D subdomains

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 7.07.08 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 7.08.02 PM.png>)

## Example: Matrix Multiplication

* Scope:
  * Only global memory (no shared memory)
  * Matrix will have a small dimension (one block of threads only)
  * Focus on `threadIdx` usage & memory transfer between host and device

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 10.12.04 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 10.14.11 PM.png>)

### Code

Note that the following kernel is launched using `MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd)` where dimGrid is (1,1,1) and dimBlock is (WIDTH, WIDTH).

![Device-side kernel function](<../../.gitbook/assets/Screen Shot 2021-02-26 at 10.16.08 PM.png>)

* Words of wisdom: In GPU computing, we use as many threads as data items (tasks, jobs) we have to perform **(Number of threads == Number of data items)**
* Understanding what thread does what job is a very common source of error in GPU computing

Typically, in each kernel, we do ...

```
__global__ void multiply_ab(int* a, int* b, int* c, int size)
{
    int whichEntry = threadIdx.x + blockIdx.x * blockDim.x;
    if (whichEntry < size)  // ... this because ...
        c[whichEntry] = a[whichEntry] * b[whichEntry];
}
```

... because all blocks launched have the same number of threads, and we need to prevent out-of-bounds indexing. Say we have an array of 1493 elements and we launch two blocks of 1024 threads each, some threads will not do work.

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 10.22.14 PM.png>)

> That's probably one of the instances, probably many instances, when you regret that you took 759, because this is not fun.    -- Prof. Dan Negrut
