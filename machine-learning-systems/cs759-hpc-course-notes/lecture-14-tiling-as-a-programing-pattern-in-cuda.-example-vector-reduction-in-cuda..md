# Lecture 14: CUDA Case Studies. \(1\) 1D Stencil Operation. \(2\) Vector Reduction in CUDA.

## Lecture Summary

* Last time
  * Atomic operations
  * Things that shape the speed of execution of a kernel
    * The concept of "occupancy" and what impacts it \(how many threads per block, how many registers/thread, how much ShMem/block\)
  * Rules of thumb, for good execution speed in GPU computing
  * The nvcc toolchain, and how code is sent to host or gpu compilers
* Today
  * Case studies: parallel reduction on the GPU & 1D convolution
  * Looking beyond today: some more GPU computing feature, but looking for a while into optimization features

![Application optimization process](../../.gitbook/assets/screen-shot-2021-02-27-at-7.36.50-pm.png)

## 1D Stencil Operation

![What the algorithm does](../../.gitbook/assets/screen-shot-2021-02-27-at-7.46.52-pm.png)

![Serial implementation](../../.gitbook/assets/screen-shot-2021-02-27-at-7.28.39-pm.png)

![Parallel implementation](../../.gitbook/assets/screen-shot-2021-02-27-at-7.29.00-pm.png)

![nvprof pointed out spaces for optimizations](../../.gitbook/assets/screen-shot-2021-02-27-at-7.31.13-pm.png)

![Use pinned memory \(pinned memory cannot be paged out by the OS\)](../../.gitbook/assets/screen-shot-2021-02-27-at-7.31.34-pm.png)

![Data partitioning example \(overlapping compute &amp; memory\)](../../.gitbook/assets/screen-shot-2021-02-27-at-7.38.03-pm.png)

![Performance improvements](../../.gitbook/assets/screen-shot-2021-02-27-at-7.39.04-pm.png)

![Optimization summary](../../.gitbook/assets/screen-shot-2021-02-27-at-7.40.12-pm.png)

## Vector Reduction in CUDA

![What the algorithm does \(summing all entries in an array\)](../../.gitbook/assets/screen-shot-2021-02-27-at-7.47.28-pm.png)

Problem: Ideally we want to synchronize across all thread blocks, but CUDA does not have global synchronization. Our workaround is to decompose into multiple kernels.

* Optimization goal: Reaching GPU peak performance
  * Choosing the right metric
    * GFLOP/s: for compute-bound kernels
    * Bandwidth: for memory-bound kernels
* Reductions have low arithmetic intensity \(1 flop/2 elements loaded\), so we should go for peak bandwidth

![Interleaved addressing: highly divergent warps are inefficient, and % operator is very slow](../../.gitbook/assets/screen-shot-2021-02-27-at-8.57.23-pm.png)

![Change which thread works on what. New problem: shared memory bank conflicts](../../.gitbook/assets/screen-shot-2021-02-27-at-8.58.08-pm.png)

![Sequential addressing](../../.gitbook/assets/screen-shot-2021-02-27-at-9.00.15-pm.png)

* Kernel 4: Replace single load w/ two loads and first add of the reduction
* Kernel 5: Loop unrolling \(unroll last warp\)
* Kernel 6: Completely unrolling \(using templates\)
* Kernel 7: Multiple elements per thread

