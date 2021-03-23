---
description: >-
  ECE/ME/EMA/CS 759: High Performance Computing for Engineering Applications,
  Spring 2021 by Prof. Dan Negrut
---

# High Performance Computing Course Notes

## Acknowledgments

* All slides/files linked are accessible on Box using a UW-Madison account
* Almost every figure and piece of code in these notes is excerpted from Prof. Dan Negrut's course slides. Some of the slides are taken from other places by Prof. Negrut -- he cited those in his slides.
* [Slides for ME759 \(of the whole semester\)](https://uwmadison.app.box.com/s/oboe3t95di8rne0g002ydj8tpd0pwwkt)
* [Slides from ME459 \(Computing Concepts for Applications in Engineering\)](https://uwmadison.app.box.com/s/943jyv29y4u145uajfedgxamhn4ru9qx)

## Table of Contents

| Date | Title | Recommended Readings |
| :--- | :--- | :--- |
| 1/25 | [Lecture 1: Course Overview](lecture-1-course-overview.md) | [Basic Linux Command Line Usage](https://www.lynda.com/Linux-tutorials/Learning-Linux-Command-Line/753913-2.html); Slurm usage \(ME459 p95-97\) |
| 1/27 | [Lecture 2: From Code to Instructions. The FDX Cycle. Instruction Level Parallelism.](lecture-2-from-code-to-instructions.-the-fdx-cycle.-instruction-level-parallelism..md) | C recap \(ME459 p114-\); [Euler usage](https://uwmadison.app.box.com/s/eu45vz9uc1a913i831b1saiu554ueb4z) |
| 1/29 | [Lecture 3: Superscalar architectures. Measuring Computer Performance. Memory Aspects.](lecture-3-superscalar-architectures.-measuring-computer-performance.-memory-aspects..md) | gdb recap \(ME459 p649-\); Ch.5 of the [C book](https://www.amazon.com/Programming-Language-2nd-Brian-Kernighan/dp/0131103628) |
| 2/1 | [Lecture 4: The memory hierarchy. Caches.](lecture-4-the-memory-hierarchy.-caches..md) | Build mgmt & cmake \(ME459 p354-\) |
| 2/3 | [Lecture 5: Caches, wrap up. Virtual Memory.](lecture-5-caches-wrap-up.-virtual-memory..md) | Git \(ME459 p449-\); [How to Write a Git Commit](https://chris.beams.io/posts/git-commit/) |
| 2/5 | [Lecture 6: The Walls to Sequential Computing. Moore’s Law.](lecture-6-the-walls-to-sequential-computing.-moores-law..md) |  |
| 2/8 | [Lecture 7: Parallel Computing. Flynn’s Taxonomy. Amdahl’s Law.](lecture-8-parallel-computing.-flynns-taxonomy.-amdahls-law..md) |  |
| 2/10 | [Lecture 8: GPU Computing Intro. The CUDA Programming Model. CUDA Execution Configuration](lecture-8-gpu-computing-intro.-the-cuda-programming-model.-cuda-execution-configuration.md) |  |
| 2/12 | [Lecture 9: GPU Memory Spaces.](lecture-9.md) |  |
| 2/15 | [Lecture 10: GPU Scheduling Issues.](lecture-10-gpu-scheduling-issues..md) |  |
| 2/17 | [Lecture 11: Execution Divergence. Control Flow in CUDA. CUDA Shared Memory Issues.](lecture-11-execution-divergence.-control-flow-in-cuda.-global-memory-access-patterns-and.md) |  |
| 2/19 | [Lecture 12: Global Memory Access Patterns and Implications.](lecture-12-cuda-shared-memory-issues..md) |  |
| 2/22 | [Lecture 13: Atomic operations in CUDA. GPU ode optimization rules of thumb.](lecture-12-cuda-shared-memory-issues.-atomic-operations-in-cuda..md) |  |
| 2/24 | [Lecture 14: CUDA Case Studies. \(1\) 1D Stencil Operation. \(2\) Vector Reduction in CUDA](lecture-14-tiling-as-a-programing-pattern-in-cuda.-example-vector-reduction-in-cuda..md) |  |
| 2/26 | [Lecture 15: CUDA Case Studies. \(3\) Parallel Prefix Scans on the GPU. Using Multiple Streams in CUDA.](lecture-15-cuda-optimization-issues.-resource-utilization-issues.-parallel-prefix-scan-on-the-gpu..md) |  |
| 3/1 | [Lecture 16: Streams, and overlapping data copy with execution.](lecture-16-streams-and-overlapping-data-copy-with-execution..md) |  |
| 3/3 | [Lecture 17: GPU Computing: Advanced Features.](lecture-17-gpu-computing-advanced-features.-unified-memory-usage..md) |  |
| 3/5 | [Lecture 18: GPU Computing with thrust and cub.](lecture-18-gpu-computing-with-thrust-and-cub..md) |  |
| 3/8 | [Lecture 19: Hardware aspects relevant in multi-core, shared memory parallel computing.](lecture-19-hardware-aspects-relevant-in-multi-core-shared-memory-parallel-computing..md) |  |
| 3/10 | [Lecture 20: Multi-core Parallel Computing with OpenMP. Parallel Regions.](lecture-20-multi-core-parallel-computing-with-openmp.-parallel-regions..md) |  |
| 3/12 | [Lecture 21: OpenMP Work Sharing.](lecture-21-openmp-work-sharing..md) |  |
| 3/15 | Lecture 22: OpenMP Work Sharing. |  |
| 3/17 | Lecture 23: OpenMP NUMA Aspects. Caching and OpenMP. |  |
| 3/19 | Lecture 24: Critical Thinking. Code Optimization Aspects. |  |
| 3/22 | Lecture 25: Computing with Supercomputers. |  |
| 3/24 | Lecture 26: MPI Parallel Programming General Introduction. Point-to-Point Communication. |  |
| 3/26 | Lecture 27: MPI Parallel Programming Point-to-Point communication: Blocking vs. Non-blocking sends. |  |
| 3/29 | Lecture 28: MPI Parallel Programming: MPI Collectives. Overview of topics covered in the class. |  |











