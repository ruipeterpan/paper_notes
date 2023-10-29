# Lecture 16: Streams, and overlapping data copy with execution.

## Lecture Summary

* Last time
  * Case study: Parallel prefix scan
  * Using streams in GPU computing
* Today
  * Wrap up streams in GPU computing: increasing problem size; improving execution speeds
  * Debugging & profiling GPU code: some nuts and bolts

## Streams

### Example 0

* Stream 1 & 2 are defined and initialized already
  * Use the two copy sub-engines at the same time: copy in (stream1), copy out (stream2)
  * Postpone launching of myKernel in stream2until the copy operation in stream1is completed

```
cudaEvent_t event;
cudaEventCreate(&event);                           // create event
cudaMemcpyAsync(d_in, in, size, H2D, stream1);     // 1) H2D copy of new input
cudaEventRecord(event, stream1);                   // record event
cudaMemcpyAsync(out, d_out, size, D2H, stream2);   // 2) D2H copy of previous result
cudaStreamWaitEvent(stream2, event);               // wait for event in stream1
myKernel<<<1000, 512, 0, stream2>>>(d_in, d_out);  // 3) GPU must wait for 1 and 2
someCPUfunction(blah, blahblah)                    // this gets executed right away

```

### Example 1

![](<../../.gitbook/assets/Screen Shot 2021-03-01 at 11.23.24 AM.png>)

![Stage 3 enqueues the set of GPU operations that need to be undertaken (the "chunkification")](<../../.gitbook/assets/Screen Shot 2021-03-01 at 11.23.40 AM.png>)

![Concurrency (manual pipelining)](<../../.gitbook/assets/Screen Shot 2021-03-01 at 11.26.23 AM.png>)

### Example 2.1

* Similar to example 1, but with two streams to increase the speed of execution
* This actually doesn't give a big speedup (62 ms -> 61 ms)

![](<../../.gitbook/assets/Screen Shot 2021-03-01 at 11.33.11 AM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-03-01 at 11.33.24 AM.png>)

![Note that the kernel stays the same](<../../.gitbook/assets/Screen Shot 2021-03-01 at 11.33.38 AM.png>)

![There is actually no overlap of copy & execution...](<../../.gitbook/assets/Screen Shot 2021-03-01 at 11.38.44 AM.png>)

### Example 2.2

![](<../../.gitbook/assets/Screen Shot 2021-03-01 at 11.39.16 AM.png>)

* Streams recap
  * Concurrency brings two flavors:
    * The copy and the execution engines of the GPU working at the same time
    * Several different kernels being executed at the same time on the GPU
* CUDA/GPU computing recap
  * Generally, any application that fits the SIMD paradigm can benefit from using GPUs
    * Good speedups at a small time and financial investment
  * Hardware is changing faster than software&#x20;

## Debugging & Profiling in CUDA

### cuda-gdb

* gdb but with more things that need our attention
* For more usage, see the slides
  * Program execution control
  * Thread focus
  * Program state inspection (stack trace, source variables, memory, HW registers, code disassembly)
  * Run-time error detection (cuda-memcheck)
  * Tips, best practices, and misc notes
* I still prefer `printf()`, change my mind. /s

### Profiling

* Nsight Compute (only focus on GPU; ncu to collect data, ncu-ui to visualize interactively)
* Nsight Systems (focus on the whole system)
* nvprof (being deprecated rn)
