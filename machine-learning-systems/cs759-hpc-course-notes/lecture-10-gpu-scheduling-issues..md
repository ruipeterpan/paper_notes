# Lecture 10: GPU Scheduling Issues.

## Lecture Summary

* Wrap up GPU computing: generalities, execution configuration
* GPU computing: scheduling execution

## Using Multiple Blocks

![](../../.gitbook/assets/screen-shot-2021-02-15-at-11.17.17-am.png)

## Execution Scheduling Issues

![Thread Index vs. Thread ID](../../.gitbook/assets/screen-shot-2021-02-15-at-11.23.19-am.png)

Scheduling questions:

* What is the order for the blocks to be executed?
* When/How are the threads in a block executed?

Two levels of schedulers:

1. Device-level scheduler \(NVIDIA GigaThread engine\): Assigns \(large numbers of\) blocks to \(small numbers of\) SMs that signal that they have “excess capacity”
2. SM-level scheduler \(more interesting\): Schedules the execution of the threads in a block onto the SM functional units

### SM-Level Scheduling

![Note that tensor cores are not present in older architectures](../../.gitbook/assets/screen-shot-2021-02-15-at-11.38.15-am.png)

* Each block of threads are divided into 32-thread warps
  * 32: Selected by NVIDIA
  * Warp: A group of 32 thread of consecutive IDs, basic scheduling unit on the SM
* SM hardware implements almost-zero-overhead warp scheduling/switching

![SM Architecture Specifications](../../.gitbook/assets/screen-shot-2021-02-15-at-11.49.35-am.png)











