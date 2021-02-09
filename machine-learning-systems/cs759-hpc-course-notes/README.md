---
description: >-
  ECE/ME/EMA/CS 759: High Performance Computing for Engineering Applications,
  Spring 2021 by Prof. Dan Negrut
---

# High Performance Computing Course Notes

## Acknowledgments

* All slides/files linked are accessible on Box using a UW-Madison account
* Almost every figure and piece of code in these notes is excerpted from Prof. Dan Negrut's course slides. Some of the slides are taken from other places by Prof. Negrut--he cited those in his slides.
* [Slides from ME459 \(Computing Concepts for Applications in Engineering\)](https://uwmadison.app.box.com/s/943jyv29y4u145uajfedgxamhn4ru9qx)

## Table of Contents

| Date | Title | Recommended Readings |
| :--- | :--- | :--- |
| 1/25 | [Course Overview](lecture-1-course-overview.md) | [Basic Linux Command Line Usage](https://www.lynda.com/Linux-tutorials/Learning-Linux-Command-Line/753913-2.html); Slurm usage \(ME459 p95-97\) |
| 1/27 | [From Code to Instructions. The FDX Cycle. Instruction Level Parallelism.](lecture-2-from-code-to-instructions.-the-fdx-cycle.-instruction-level-parallelism..md) | C recap \(ME459 p114-\); [Euler usage](https://uwmadison.app.box.com/s/eu45vz9uc1a913i831b1saiu554ueb4z) |
| 1/29 | [Superscalar architectures. Measuring Computer Performance. Memory Aspects.](lecture-3-superscalar-architectures.-measuring-computer-performance.-memory-aspects..md) | gdb recap \(ME459 p649-\); Ch.5 of the [C book](https://www.amazon.com/Programming-Language-2nd-Brian-Kernighan/dp/0131103628) |
| 2/1 | [The memory hierarchy. Caches.](lecture-4-the-memory-hierarchy.-caches..md) | Build mgmt & cmake \(ME459 p354-\) |
| 2/3 | [Caches, wrap up. Virtual Memory.](lecture-5-caches-wrap-up.-virtual-memory..md) | Git \(ME459 p449-\); [How to Write a Git Commit](https://chris.beams.io/posts/git-commit/) |
| 2/5 | [The Walls to Sequential Computing. Moore’s Law.](lecture-6-the-walls-to-sequential-computing.-moores-law..md) |  |
| 2/8 | [Parallel Computing. Flynn’s Taxonomy. Amdahl’s Law.](lecture-8-parallel-computing.-flynns-taxonomy.-amdahls-law..md) |  |
| 2/10 | GPU Computing Intro. The CUDA Programming Model. CUDA Execution Configuration |  |
| 2/12 | GPU Memory Spaces. |  |
| 2/15 | GPU Scheduling Issues. |  |
| 2/17 | Execution Divergence. Control Flow in CUDA. Global Memory Access Patterns and Implications. |  |
| 2/19 | CUDA Shared Memory Issues. Atomic operations in CUDA. |  |
| 2/22 | Using the CUDA profiler. Example: 1D Stencil Operation. |  |
| 2/24 | Tiling as a Programing Pattern in CUDA. Example: Vector Reduction in CUDA. |  |
| 2/26 | CUDA Optimization Issues. Resource Utilization Issues. Parallel Prefix Scan on the GPU. Using Multiple Streams in CUDA. |  |
| 3/1 | Streams, and overlapping data copy with execution. |  |
| 3/3 | GPU Computing: Advanced Features. |  |
| 3/5 | GPU Computing with thrust and cub. |  |
| 3/8 | Hardware aspects relevant in multi-core, shared memory parallel computing. |  |
| 3/10 | Multi-core Parallel Computing with OpenMP. Parallel Regions. |  |
| 3/12 | OpenMP Work Sharing. |  |
| 3/15 | OpenMP Work Sharing. |  |
| 3/17 | OpenMP NUMA Aspects. Caching and OpenMP. |  |
| 3/19 | Critical Thinking. Code Optimization Aspects. |  |
| 3/22 | Computing with Supercomputers. |  |
| 3/24 | MPI Parallel Programming General Introduction. Point-to-Point Communication. |  |
| 3/26 | MPI Parallel Programming Point-to-Point communication: Blocking vs. Non-blocking sends. |  |
| 3/29 | MPI Parallel Programming: MPI Collectives. |  |











