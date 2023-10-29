# \[2021 FAST] CheckFreq: Frequent, Fine-Grained DNN Checkpointing

## One-line Summary

CheckFreq pipelines checkpointing with computation for automated, frequent, fine-grained checkpointing in DNN training.

## Paper Structure Outline

1. Introduction
2. Background
3. The Current State of Checkpointing
   1. Checkpointing is Incorrect
   2. Checkpointing is Inefficient
   3. Summary
4. CheckFreq: Design and Implementation
   1. Goals
   2. CheckFreq Recovery Guarantees
   3. Design
      1. Checkpointing Mechanism
      2. Checkpointing Policy
   4. Implementation
5. Evaluation
   1. Experimental Setup
   2. Accuracy Implications
   3. Performance of Checkpointing Mechanism
      1. Checkpoint Stalls
      2. Breakdown of Benefits
   4. Checkpointing Policy
   5. Recovery Time
   6. End-to-End Training
6. Discussion
7. Related Work
8. Conclusion

## Background & Motivation

During DNN training, checkpointing is performed to ensure fault tolerance. Current checkpointing schemes are synchronous, thus leading to large checkpoint stalls. Furthermore, due to bigger models and larger datasets, epoch times are increasing. Typically, checkpointing is performed at epoch boundaries and the checkpointing frequency needs to be set manually. → We need fine-grained, iteration-level checkpointing.

## Design and Implementation

CheckFreq is an automated checkpointing framework for DNN training.

### Mechanism: Low-cost, pipelined checkpointing

#### Low checkpoint stalls: 2-phase DNN-aware checkpointing

CheckFreq decouples the traditional checkpointing into two phases: `snapshot()` and `persist()`. `snapshot()` serializes the training state and copies it from the GPU memory to a user-space buffer in CPU memory. `persist()` writes the serialized content to disk. These two phases are pipelined with DNN training computation.

![](<../../.gitbook/assets/Screen Shot 2021-05-21 at 2.44.34 PM.png>)

In the optimal case, as the model weights are synchronized in the last phase of an iteration, we can pipeline the `snapshot()` with the forward & backward pass of the next iteration, minimizing the checkpoint stall.

The authors also found that doing the snapshot on the GPU has an orders-of-magnitude lower cost than that on the CPU, as the latter involves a memory copy from GPU to CPU. Therefore, if spare GPU memories are available, the snapshot is done on the GPU memory.

#### Maintain data invariant: Resumable data iterator

Current data iterators do not guarantee the order of data items after resuming. CheckFreq resolves this by using a seed that is a function of the epoch number to reconstruct the shuffle order after resuming.

![](<../../.gitbook/assets/Screen Shot 2021-05-21 at 3.07.00 PM.png>)

### Policy: When to checkpoint?

#### Initial frequency: Systematic online profiling

The key idea is to come up with a frequency of checkpointing every k iterations such that:

1. The cost of 1 checkpoint can be amortized over k iterations
2. The runtime overhead of checkpointing is within a user-defined threshold of the actual compute time (say 5%)

To accomplish this, CheckFreq profiles: the iteration time (Ti), time to perform weight update (Tw), time to create an in-memory GPU copy (Tg), time to create an in-memory CPU copy (Tc), time to write to storage (Ts), size of checkpoint (m), peak GPU memory utilization (M), and total GPU memory (Mmax). Then, the frequency is determined as follows:

![](<../../.gitbook/assets/Screen Shot 2021-05-21 at 2.56.32 PM.png>)

#### Adaptive rate tuning: Manages interference from other jobs

Consider the following example

![](<../../.gitbook/assets/Screen Shot 2021-05-21 at 3.17.16 PM.png>)

* Isolated: When a job runs alone, the checkpointing overhead is kept at 5% as specified by the user
* Static: When another job space-shares the same GPU, checkpointing at the previous frequency results in a 35% overhead
* Adaptive: CheckFreq's adaptive policy reduced the checkpoint frequency and keeps the overhead at 5%

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2021-05-21 at 3.19.34 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-05-21 at 3.20.27 PM.png>)

## Links

* [Paper PDF](https://www.usenix.org/system/files/fast21-mohan.pdf)
* [Presentation video at FAST '21](https://www.youtube.com/watch?v=E3uaeaqfjcY)
* [Presentation slides at FAST '21](https://www.usenix.org/sites/default/files/conference/protected-files/fast21\_slides\_mohan.pdf)
* [msr-fiddle/CheckFreq on GitHub](https://github.com/msr-fiddle/CheckFreq)
