# MLSys Papers - Short Notes

## \[2019 arXiv] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

This work proposes tensor parallelism (TP), where tensors are partitioned across devices and are only aggregated for operations that require the whole tensor. A key insight of TP is that matrix multiplication can be split between multiple GPUs to parallelize computation and save memory.

&#x20;![](../../.gitbook/assets/parallelism-tp-parallel\_gemm.png)

Each transformer layer consists of a self-attention block followed by a two-layer, multi-layer perceptron (MLP). To parallelize an MLP, column parallelism can be used to split the matrix multiplication, and synchronizations are not needed until the very end of the computation. Parallelizing the multi-headed attention layers is even easier since they are already inherently parallel. As a result, each transformer layer requires two allreduce during the forward pass and two allreduce during the backward pass.

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.43.11 PM.png>)

Note that using TP requires a super fast network for near-theoretical-optimal performance, and in real life, TP is usually used in conjugation with other forms of parallelism.

## \[2019 MLSys] BlueConnect: Decomposing All-Reduce for Deep Learning on Heterogeneous Network Hierarchy

BlueConnect adapts to the hierarchy of communication bandwidths by leveraging topology-awareness to fully utilize the heterogeneous network architecture. It decomposes all-reduce (reduce-scatter + all-gather) into multiple stages of parallelizable reduce-scatter & all-gather, which provides more granularity and flexibility to map operations to the heterogeneous underlying network hierarchy.

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.06.26 PM.png>)

## \[2020 MLSys] Blink: Fast and Generic Collectives for Distributed ML

This paper address the problem of link under-utilization due to **topology heterogeneity** in distributed ML training. Topology heterogeneity mainly comes from (1) differing server configurations (e.g, different NVLink topologies across generations of DGX nodes) and (2) scheduler’s topology-agnostic placements/allocations (e.g., an 8-GPU job uses 3 GPUs in an 8-GPU DGX node and 5 GPUs from another). To handle topology heterogeneity from hardware generations or partial allocations from cluster schedulers, Blink **dynamically generates optimal communication primitives for a given topology**. Blink models collective communication operations as flows on a directed graph and uses a spanning-tree packing algorithm to maximize link bandwidth utilization.

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 1.45.47 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 1.40.16 PM.png>)

## \[2021 ICML] Boosting the Throughput and Accelerator Utilization of Specialized CNN Inference Beyond Increasing Batch Size

Serving specialized CNNs (e.g., for offline video analytics) have low arithmetic intensity, leading to the severe under-utilization of server-grade accelerators. Increasing the batch size is a popular technique to boost the arithmetic intensity, utilization, and application-level throughput by amortizing the cost of loading a CNN’s weights from memory. However, it suffers from diminishing returns. This paper proposes a technique to **redesign specialized CNNs** with the purpose of **boosting the inference utilization and throughput**. The key insight is that, once arithmetic intensity has plateaued due to increased batch size, reading/writing activations accounts for most of the memory traffic in specialized CNNs. The authors show that this memory traffic can be significantly reduced, while performing the same number of FLOPs, by jointly decreasing the size of the batch of input/output activations for a layer and increasing the layer’s width.

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 11.59.45 AM.png>)

Compared to vanilla CNNs, FoldedCNNs have improvements on the throughput and the accelerator utilization while suffering slight accuracy loss.

## \[2021 arXiv] Synthesizing Collective Communication Algorithms for Heterogeneous Networks with TACCL

TACCL encodes a profiled topology and input size into a synthesis problem to generate optimized communication algorithms.

NCCL uses the topology of GPU connections and NIC placement along with buffer size to decide between two main types of communication algorithms — Ring and Tree, but it is agnostic to the exact performance profile of the links, and thus is often multiple times slower than TACCL’s custom collectives

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 1.49.42 PM.png>)

## \[2021 SC] Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines

Chimera is yet another pipeline parallelism paradigm. Compared with the other STOA systems, it reduces more compute idleness and has a more balanced activation memory consumption.

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.22.11 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-07-11 at 2.23.27 PM.png>)

