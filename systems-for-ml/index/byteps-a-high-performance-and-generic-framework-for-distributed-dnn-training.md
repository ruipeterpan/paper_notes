---
description: >-
  A Unified Architecture for Accelerating Distributed DNN Training in
  Heterogeneous GPU/CPU Clusters
---

# BytePS: A high performance and generic framework for distributed DNN training

### One-liner Summary

### Table of Contents

1. Introduction
2. Background
   1. Distributed DNN Training
   2. All-reduce
   3. Parameter Server \(PS\)
3. Motivation and BytePS Architecture
   1. Motivation
   2. Architecture Overview
4. BytePS Communication Design
   1. Inter-machine Communication
      1. Communication Efficiency Analysis
   2. Intra-machine Communication
      1. PCIe-only Topology
      2. NVLink-based Topology
      3. Discussion
5. Summation Service
6. Implementation
   1. Multi-Stage Pipeline
   2. Address RDMA Performance Issues
   3. BytePS Usage
7. Evaluation
   1. Inter-machine Microbenchmarks
   2. Leverage CPU Machines
   3. Adapt to Intra-machine Topology
   4. Scalability
8. Observations and Discussion
9. Related Work
10. Conclusion



### Background & Motivation

Existing architectures \(all\_reduce and parameter server\) for distributed DNN training are insufficient.

![The two architectures for distributed training based on data parallelism: All-reduce and PS](../../.gitbook/assets/screen-shot-2020-11-30-at-11.23.14-am.png)

![Even with ByteScheduler, we are still 30% away from the optimal performance](../../.gitbook/assets/screen-shot-2020-11-30-at-11.17.12-am.png)

The paper analyzed three problems that lead to this slowdown, and then presented a solution to each problem:

1. Sub-optimal Inter-machine Communication
2. Sub-optimal Intra-machine Communication
3. The CPU Bottleneck

#### Sub-optimal Inter-machine Communication

![](../../.gitbook/assets/screen-shot-2020-11-30-at-11.27.44-am.png)

For allreduce, the CPUs are not leveraged properly as the communication is between GPUs only. For PS, if there are insufficient CPUs for servers, bottlenecks may be created. **The solution is an optimal communication strategy that unifies allreduce and PS.**

#### Sub-optimal Intra-machine Communication

![](../../.gitbook/assets/screen-shot-2020-11-30-at-11.29.52-am.png)

There are often multiple GPUs in a GPU machine IRL. The internal topology is also a network, which results in bottlenecks. **The solution is intra-machine optimizations that accommodate diverse intra-machine topologies.**

#### The CPU Bottleneck

In the parameter server setup, the GPU workers send the gradients to CPU servers. The CPU servers first aggregate the gradients received, and then update the parameters using the optimizer function. The problem is that CPUs might not match network rates, thus creating bottlenecks. **The solution is a summation service that moves parameter updates from CPUs to GPUs.**

### Design and Implementation

#### Sub-optimal Inter-machine Communication

PS only uses links between CPUs and GPUs and does not utilize the bandwidth between GPU machines. In the allreduce setup, the communication solely relies on inter-GPU communications, not utilizing the CPU at all. BytePS takes the best of both worlds and combines these two strategies. In the paper, the authors presented an optimal partition strategy that adjusts the proportion of CPU-GPU and GPU-GPU communications.

![](../../.gitbook/assets/screen-shot-2020-11-30-at-11.57.04-am.png)

#### Sub-optimal Intra-machine Communication

![I might come back to this later, this is a bit complicated to understand for now :P](../../.gitbook/assets/screen-shot-2020-11-30-at-12.00.35-pm.png)

#### The CPU Bottleneck

The PS server's role can be divided into two parts: Gradient Summation & Parameter Update. Typically, forward propagation and backward propagation get placed on GPUs, while gradient summation and parameter update are placed on GPUs. The authors found that the gradient summation step is CPU-friendly, while the parameter update step is heavy. To resolve this issue, the authors presented the Summation Service, which moves the parameter update to GPUs to resolve the aforementioned bottleneck.

#### System Architecture Overview

![](../../.gitbook/assets/screen-shot-2020-11-30-at-12.05.47-pm.png)

### Evaluation

### New Vocabulary

* Network Interface Controller \(NIC\)
* PCI Express \(PCIe\)

### Links

* [Paper PDF](https://www.usenix.org/system/files/osdi20-jiang.pdf)
* [Presentation Video at OSDI '20](https://www.youtube.com/watch?v=j8PHNglSZX8&feature=emb_logo&ab_channel=USENIX)
* [Presentation Slides PDF](https://www.usenix.org/sites/default/files/conference/protected-files/osdi20_slides_jiang.pdf)
* [BytePS on GitHub](https://github.com/bytedance/byteps)

