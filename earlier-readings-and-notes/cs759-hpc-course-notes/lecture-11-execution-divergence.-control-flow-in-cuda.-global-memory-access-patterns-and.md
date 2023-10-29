# Lecture 11: Execution Divergence. Control Flow in CUDA. CUDA Shared Memory Issues.

## Lecture Summary

* Last time
  * GPU Computing: Execution Scheduling
    * Block scheduler (at the GPU level)
    * Warp scheduler (at the SM level)
  * Thread Divergence
* Today
  * Aspects related to how GPU memory operations take place

## The NVIDIA GPU Memory Ecosystem

![From high vantage point (2 blocks w/ 2 threads each)](<../../.gitbook/assets/Screen Shot 2021-02-20 at 12.45.24 PM.png>)

Each thread can:

* R/W per-thread registers&#x20;
* R/W per-thread local memory&#x20;
* R/W per-block shared memory&#x20;
* R/W per-grid global memory&#x20;
* Read only per-grid constant memory&#x20;
* Read only per-grid texture memory&#x20;
* Read only per-grid surface memory

Some aspects of Local Memory:

* Physically, local memory does not exist
  * In reality, data stored in local memory is placed in cache or the global memory at run time or by the compiler
* It's specific to one thread and not visible to any other thread
* Local memory has the same latency as global memory, unless cached

Different memories:

* Global memory: Main means of communicating R/W data between host and device. cudaMalloc(), cudaFree(), and cudaMemcpy() operate here. Note that there are four types of cudaMemcpy transfers ({host/device} to {host/device}), and things happen over a PCIe connection.
* Texture and Constant memories: Constants initialized by host, contents available to all threads.&#x20;

Global, texture and constant memories are accessible by host (done at high latency, low bandwidth).

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 11.20.44 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 11.20.58 PM.png>)

![Memory Access Times](<../../.gitbook/assets/Screen Shot 2021-02-20 at 12.52.54 PM.png>)

![Storage Locations](<../../.gitbook/assets/Screen Shot 2021-02-20 at 12.53.19 PM.png>)

![The 3 most important GPU memory spaces](<../../.gitbook/assets/Screen Shot 2021-02-20 at 12.54.01 PM.png>)

## Case Studies: Matrix Multiplication, Revisited

Purpose:

* See an example where the use of multiple blocks of threads play a central role
* Highlight the use/role of the shared memory
* Point out the \_\_syncthreads() function call (synchronizes all threads in a block)



* The previous example: Low arithmetic intensity, a lot of unnecessary movements from global memory to device
* **Rule of thumb: If the data that you, as a thread, use can also be used by another thread in your block, then you should consider using shared memory**
* To use shared memory:
  * Partition data into data subsets (tiles) that each fits into shared memory
  * Handle each data subset (tile) with one thread block by:
    * Loading the tile from global memory into shared memory, using multiple threads to exploit memory-level parallelism
    * Performing the computation on the tile from shared memory; each thread can efficiently multi-pass over any data element of the tile

![](<../../.gitbook/assets/Screen Shot 2021-02-20 at 1.05.39 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-02-20 at 1.05.57 PM.png>)

* `__syncthreads()` synchronizes all threads in a block
  * Used to avoid RAW/WAR/WAW hazards when accessing shared or global memory
  * Be very careful when using it in a conditional
* 3 ways to set aside shared memory:
  * Statically, declare inside a kernel
  * Through the execution configuration (see code block below)
  * Dynamically, via CUDA driver API `cuFuncSetSharedSize()` (out of scope)

```
__global__ void MyFunc(float*) // __device__ or __global__ function 
{
    extern __shared__ float shMemArray[];
    // Size of shMemArray determined through the execution configuration
    // You can use shMemArrayas you wish here...
}

// invoke like this. Ns indicates the size in bytes to be allocated in shared memory
MyFunc<<< Dg, Db, Ns>>>(parameter);
```

![Example: Reversing an array using dynamic shared memory](<../../.gitbook/assets/Screen Shot 2021-02-26 at 11.31.07 PM.png>)

![How different technology fetches data into shared memory](<../../.gitbook/assets/Screen Shot 2021-02-26 at 11.33.21 PM.png>)

* Each SM has shared memory organized in 32 memory banks
  * Successive 32-bit words map to successive banks
  * Each bank has a bandwidth of 32 bits per clock cycle
* ShMem and L1 cache draw on the same physical memory inside an SM

![](<../../.gitbook/assets/Screen Shot 2021-02-26 at 11.48.42 PM.png>)
