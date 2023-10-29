# \[2021 EuroSys] NextDoor: Accelerating graph sampling for graph machine learning using GPUs

## One-line Summary

In Graph Neural Network (GNN) training, existing approaches use CPUs to sample the graph before using GPUs to train the GNN, but sampling is a major overhead (up to 62% of training time). Nextdoor uses GPUs to accelerate graph sampling by up to 4x, and its main contributions are:

1. Simple abstractions & API to express diverse graph sampling algorithms
2. A new "transit parallel" approach to increase the parallelism of graph sampling
3. Optimizations (load balancing & caching) to improve GPU utilization

![Nextdoor structure](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.23.09 PM.png>)

Takeaways from Shivaram's group meeting after discussing this paper include:

* In evaluations, except from the relative numbers, post absolute values as well
* Parallelization works very well on GPUs, more CPU-based tasks may be identified and transformed into GPU-based tasks with high levels of parallelization
* Nextdoor introduces this abstraction/API that "bounds" graph sampling algorithms so that they can be properly parallelized

## Paper Structure Outline

1. Introduction
2. Background and Motivation
   1. Representation Learning on Graphs
   2. Requirements for GPU Performance
3. An Abstraction for Graph Sampling
4. Graph Sampling using NEXTDOOR
   1. Programming API
   2. Use Cases
5. Paradigms for Graph Sampling on GPUs
   1. Sample-Parallelism
   2. Transit-Parallelism
6. Efficient Transit Parallelism on GPUs
   1. Sampling in Individual Transit Sampling
   2. Transit-Parallel Collective Transit Sampling
   3. Unique Neighbors
   4. Graph Sampling using Multiple GPUs
   5. Integration in GNNs using Python API
   6. Advantages of NEXTDOOR's API
7. Alternative Graph Processing Systems
8. Evaluation
   1. Execution Time Breakdown
   2. Graph Sampling Performance
   3. Alternative GPU-Based Abstractions
   4. Sampling Large Graphs
   5. Sampling on Multiple GPUs
   6. End-to-End Integration in GNN Systems
9. Related Work
10. Conclusion

## Background & Motivation

### Background 1: How GNN training works

* GNNs maps vertices of (input) graphs to an embedding in an N-dimensional space so that the similarity of embeddings between nodes indicate their similarity in the network
  * The embeddings are then used for many downstream tasks (e.g., product recommendation, clustering)
* There are two types of GNNs, and this work focuses on the first one (they're more common):
  * Sampling-based GNNs samples the input graph and train using these samples
  * Whole-Graph-based GNNs train on the whole input graph directly
* Workflow of Sampling-based GNNs: First, a graph sampling algorithm is used to sample the input graph, and the samples are then used for data parallel training
* Currently, most implementations use CPUs for sampling, because the implementation is easier

![Different graph sampling algorithms](<../../.gitbook/assets/Screen Shot 2021-08-20 at 7.55.05 PM.png>)

### Background 2: How to best utilize GPUs

* Level of parallelism should be high (in GPU computing, # threads == # samples)
* Accesses to the global memory should be coalesced and aligned
* Shared memory and registers for each SM can be used as software-managed cache
* Avoid warp divergence

![](<../../.gitbook/assets/Screen Shot 2021-08-20 at 7.55.58 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.08.19 PM.png>)

### Motivation: Graph sampling on CPUs is a major overhead

![Existing implementations spend as much as 62% of the training time on graph sampling](<../../.gitbook/assets/Screen Shot 2021-08-13 at 1.45.24 PM.png>)

Currently, graph sampling is done on CPUs because of the ease of implementation. Nextdoor attempts to provide both easy-to-implement and fast graph sampling.

## Design and Implementation

### Powerful abstraction/API to express sampling algorithms

* Input to Nextdoor
  * A graph
  * An initial set of samples, each with >=1 root vertices
  * User-defined functions to describe the algorithm
* Output of Nextdoor: An expanded set of examples
* Nextdoor abstractions:
  * A graph sampling appplication runs for k steps
  * At each step i,
    * A transit vertex for i is a vertex whose neighbors may be added to the sample
    * Sample mi of those neighbors
  * There are two types of sampling:
    * Individual transit sampling: Sample mi neighbors per-transit-node
    * Collective transit sampling: Sample mi neighbors per-sample

![Example algorithms expressed using this abstraction](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.03.07 PM.png>)

![Required user-defined functions](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.03.55 PM.png>)

![Use cases of Nextdoor](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.04.36 PM.png>)

### Transit parallel to increase parallelism

* Status quo: One thread for each sample -> poor parallelism. How can we increase the parallelism?
* Sample parallel: In each thread, one neighbor of a transit vertex is sampled, and samples are assigned to consecutive threads
  * The parallelism is better
  * However, sample parallel suffers from irregularity: The access to the global memory is random, and shared memory/registers cannot be used as caches
* Transit parallel: Assign samples with common transits to consecutive threads
  * A GroupBy operation is needed to invert the sample-transit mapping to a transit-sample mapping
  * Here, consecutive threads access edges of the same transit vertices. Therefore, the global memory accesses are coalesced, and shared memory/registers can be used for caches
* The Nextdoor API exposes three levels of parallelism
  1. Each transit is mapped to a threadblock
  2. Each sample is assigned to a group of mi threads at step i
  3. Each thread samples one neighbor

![Sample parallel](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.15.13 PM.png>)

![Transit parallel](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.15.34 PM.png>)

### Optimization techniques for GPUs (load balancing, caching)

Nextdoor uses different {types of kernels, caching strategies, neighbor access strategy, transit scheduling strategy} to process transit vertices based on the number of neighbors to sample for the transit vertex, which helps to best utilize the memory/compute resources

![Sub warp: A set of contiguous threads of the same warp assigned to the same sample. All sub warps have the same size, which is determined using sampleSize function for the current step.](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.17.36 PM.png>)

## Evaluation

![End-to-end speedups for GNN training](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.20.22 PM.png>)

![Nextdoor against existing graph sampling implementations](<../../.gitbook/assets/Screen Shot 2021-08-20 at 8.20.57 PM.png>)

The original paper also included some microbenchmarks of speedups of SP, TP, and the overhead of the GroupBy operation, etc.

## Links & References

* [Paper PDF](https://marcoserafini.github.io/projects/nextdoor/nextdoor.pdf)
* Presentation video at EuroSys '21 ([Long](https://www.youtube.com/watch?v=GsffY0j6tVE\&list=PLzDuHU-z7gNjuSbEYCFXZtWAl3nAdNF2f\&index=19) & [Short](https://www.youtube.com/watch?v=lwB7KcMIpkQ\&list=PLzDuHU-z7gNghxOWGcdLK\_xWtqHjxaYTm\&index=19))
* [Presentation slides at EuroSys '21](https://2021.eurosys.org/docs/presentations/6-Jangda%20-%20Abhinav%20Jangda.pdf)
* Graph sampling algorithms referenced in Nextdoor
  * [DeepWalk](https://arxiv.org/pdf/1403.6652.pdf)
  * [node2vec](https://arxiv.org/pdf/1607.00653.pdf)
  * [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf)
  * [FastGCN](https://arxiv.org/pdf/1801.10247.pdf)
  * [ClusterGCN](https://arxiv.org/pdf/1905.07953.pdf)
  * [LADIES](https://arxiv.org/pdf/1911.07323.pdf)
