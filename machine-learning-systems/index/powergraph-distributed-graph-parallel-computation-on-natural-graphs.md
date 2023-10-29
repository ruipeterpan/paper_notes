# \[2012 OSDI] PowerGraph: Distributed Graph-Parallel Computation on Natural Graphs

## One-line Summary

The key contributions are:

* GAS (gather, apply, scatter) programming model
* Using vertex cut instead of edge cut to layout data for power-law graphs
* Balancing computation & minimizing communication

## Paper Structure Outline

1. Introduction
2. Graph-Parallel Abstractions
   1. Pregel
   2. GraphLab
   3. Characterization
3. Challenges of Natural Graphs
4. PowerGraph Abstraction
   1. GAS Vertex-Programs
   2. Delta Caching
   3. Initiating Future Computation
      1. Bulk Synchronous Execution
      2. Asynchronous Execution
   4. Comparison with GraphLab/Pregel
5. Distributed Graph Placement
   1. Balanced p-way Vertex-Cut
   2. Greedy Vertex-Cuts
6. Abstraction Comparison
   1. Computation Imbalance
   2. Communication Imbalance
   3. Runtime Comparison
7. Implementation and Evaluation
   1. Graph Loading and Placement
   2. Synchronous Engine (Sync)
   3. Asynchronous Engine (Async)
   4. Async. Serializable Engine (Async+S)
   5. Fault Tolerance
   6. MLDM Applications
8. Related Work
9. Conclusions and Future Work

## Background & Motivation

* **Background 1: Natural Graphs**
  * Graphs IRL (e.g., social networks/the Internet) follow a power-law degree distribution
    * A small subset of the vertices have very high degrees, while most vertices have a small degree
  * Existing graph-parallel frameworks depend on a balanced degree distribution for performance

![](<../../.gitbook/assets/Screen Shot 2021-07-22 at 3.07.26 PM.png>)

* **Background 2: Existing frameworks (**[**Pregel**](pregel-a-system-for-large-scale-graph-processing.md)**, GraphLab) cannot handle natural graphs well**
  * Work balancing: Existing graph-parallel frameworks treat vertices symmetrically and have storage/communication/computation costs linear in degree
  * Partitioning: Pregel/GraphLab depends on partitioning the graph, which is hard to do in natural graphs. Their solution, random partitioning, is bad.
  * Communication/storage: Major bottlenecks at high-degree vertices due to the skewed distribution
  * Computation: Existing frameworks do not parallelize individual vertex programs, limiting their scalability in skewed graphs

## **Design and Implementation**

### GAS Model

* Gather: Information from adjacent vertices/edges is reduced by a generalized "sum" operation (commutative and associative)
* Apply: The gathered sum is used with the current value to update the current vertex value
* Scatter: The new value is used to update data on adjacent edges

### Vertex-Cuts instead of Edge-Cuts

* Edge-Cuts
  * Every vertex is placed on a machine, and edges span across machines
    * If adjacent vertices are on different machines, they use "ghost" vertices -> changes need to be synchronized to ghosts
  * In natural graphs, there are lots of edges spanned across machines; Balanced edge-cut algorithms perform poorly, so GraphLab and Pregel uses randomized placement (bad)
* Vertex-Cuts
  * Every edge is placed on a machine, and vertices may be across machines
    * Intuition: The distribution of vertex degree is highly skewed, but the number of vertices adjacent to a given edge is constant (always 2)
    * Each vertex is replicated ("mirrors") across the machines where its adjacent edges lie
  * This results in a better balance for natural graphs

![](<../../.gitbook/assets/Screen Shot 2021-07-22 at 3.23.24 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-07-22 at 3.34.43 PM.png>)

### Delta Caching, Execution Model

* Delta caching
  * At each vertex, the accumulator values are cached, and the scatter function can return a delta value to directly apply to the neighboring cached accumulator.
  * If this value is not returned, the neighboring cache is cleared
* Execution model: Sync vs. Async
  * Sync (bulk synchronous)
    * 3 "minor-steps": Gather for all active vertices -> Apply -> Scatter
    * Barrier after each minor-step; Changes are committed at the end of each minor-step and visible on the next
  * Async (asynchronous)
    * Changes are immediately available to other vertices
    * Execute active vertices as cores become available

## Evaluation

### Reduced vertex replication/communication costs

![](<../../.gitbook/assets/Screen Shot 2021-07-22 at 3.30.10 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-07-22 at 3.56.15 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-07-22 at 3.58.33 PM.png>)

## Links

* [Paper PDF](https://www.usenix.org/system/files/conference/osdi12/osdi12-final-167.pdf)
* [Presentation slides by 6.886 @ MIT](https://people.csail.mit.edu/jshun/6886-s20/lectures/lecture11-2.pdf)
* [Presentation slides by CS 744 @ UW-Madison](http://pages.cs.wisc.edu/\~shivaram/cs744-fa20-slides/cs744-powergraph-notes.pdf)
* [PowerGraph on GitHub](https://github.com/jegonzal/PowerGraph)
