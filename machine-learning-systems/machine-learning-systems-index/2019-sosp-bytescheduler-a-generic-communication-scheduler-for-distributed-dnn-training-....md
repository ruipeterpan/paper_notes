---
description: ...Acceleration
---

# \[2019 SOSP] ByteScheduler: A Generic Communication Scheduler for Distributed DNN Training ...

## Summary

Priority-based communication scheduling + tensor partitioning: acceleration! Fig. 2 is a good toy example that showcases why the default order of communication (FIFO) in current ML frameworks is suboptimal. However, prior systems ([P3](https://arxiv.org/abs/1905.03960) and [TicTac](https://arxiv.org/abs/1803.03288)) that try to tackle this are not generic, in the sense that each of them only targets one combination of DL framework & network stack. Moreover, existing work does not adapt well to different system setups.

In contrast, ByteScheduler is generic (framework/communication method-agnostic), which required some intricate engineering efforts/techniques. Also, ByteScheduler proposes a BO-based auto-tuning algorithm to search for the best system parameters (e.g., tensor partition sizes) under different environments (DNN models, communication paradigms, bandwidth, etc.).

![The training speedup of priority scheduling is 44%!](<../../.gitbook/assets/Screen Shot 2022-02-06 at 2.20.12 PM.png>)

## Background & Motivation

In distributed DNN training using data parallelism, the default ML framework engines execute communication operations in a FIFO order, as the underlying communication stack (PS/all\_reduce, TCP/RDMA) is inherently based on FIFO queues. However, this is suboptimal: if some communication operations are prioritized, the training can be sped up.

Tensor partitioning is a technique that enables more flexible priority-based scheduling. Without partitioning, a large, low-priority tensor might block high-priority tensors. Instead, the tensors can be partitioned before being en-queued, and high-priority tensor partitions can jump ahead of the queue after they arrive.

## Design & Implementation

### Which layer should ByteScheduler be implemented in to make it more general?

![](<../../.gitbook/assets/Screen Shot 2022-02-06 at 3.35.38 PM.png>)

The five original layers are shown above. After some thoughtful thinking, the authors placed ByteScheduler at the high-level API implementation layer in the framework. For each ML framework, a shim layer ("plugin") is designed to wrap the original operation into a unified "CommTask" abstraction.

### Unified abstraction for communication tasks

A single interface, `Core.enqueue(CommTask)`, is exposed to the plugins. Once a communication tensor arrives, it is first wrapped into a CommTask. Then, the Core partitions it into SubCommTasks and decides when to send each. Four CommTask interfaces are implemented:

* CommTask.partition(size): Partitions a CommTask into multiple SubCommTasks with tensors no larger than the specified size. This invokes a callback in the plugin, as tensor partitioning is framework-dependent. This has a low overhead, as DL frameworks provide zero-copy APIs for tensor partitioning.
* CommTask.notify\_ready(): The engine uses this interface to notify the Core about a tensor being ready, so it can be actually scheduled.
* CommTask.start(): The Core calls this to let engines and the underlying communication stacks send the tensor.
* CommTask.notify\_finish(): The framework engines notify the Core once the communication (push/pull/all\_reduce) finishes so that the Core can continue scheduling more Tasks.

### Interaction with framework engines and crossing the global barrier

![](<../../.gitbook/assets/Screen Shot 2022-02-06 at 4.00.28 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-02-06 at 4.01.11 PM.png>)

### Auto-tuning partition size and credits using Bayesian Optimization

![](<../../.gitbook/assets/Screen Shot 2022-02-06 at 4.09.14 PM.png>)

## Comparisons with P3 and TicTac

P3 and TicTac, both in MLSys '19, employ similar ideas and techniques (transmission prioritization via tensor partitioning & reordering). However, both systems target specific training setups (e.g., P3 targets MXNet PS + TCP), while ByteScheduler devotes a significant chunk of engineering efforts on the system design so that it not only outperforms prior systems but also works well with different training configurations.

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2022-02-06 at 4.10.57 PM.png>)

The paper provided the reasoning for different speedups in different setups.

## Links & References

* [Paper PDF](https://i.cs.hku.hk/\~cwu/papers/yhpeng-sosp19.pdf)
* [Presentation video at SOSP '19](https://www.youtube.com/watch?v=UL1\_69lI9BE)
* [bytescheduler on GitHub](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler)
