# Lecture 14: Tiling as a Programing Pattern in CUDA. Example: Vector Reduction in CUDA.

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



