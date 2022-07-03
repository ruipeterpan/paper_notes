---
description: ...Computing
---

# \[2012 NSDI] Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster ...

## One-line Summary

Spark is a generalized MapReduce model (in-memory Hadoop :P). It uses RDDs to support in-memory computations.

## Paper Structure Outline

1. Introduction
2. Resilient Distributed Datasets (RDDs)
   1. RDD Abstraction
   2. Spark Programming Interface
      1. Example: Console Log Mining
   3. Advantages of the RDD Model
   4. Applications Not Suitable for RDDs
3. Spark Programming Interface
   1. RDD Operations in Spark
   2. Example Applications
      1. Logistic Regression
      2. PageRank
4. Representing RDDs
5. Implementation
   1. Job Scheduling
   2. Interpreter Integration
   3. Memory Management
   4. Support for Checkpointing
6. Evaluation
   1. Iterative Machine Learning Applications
   2. PageRank
   3. Fault Recovery
   4. Behavior with Insufficient Memory
   5. User Applications Built with Spark (In-mem analytics, traffic modeling, twitter spam classification)
   6. Interactive Data Mining
7. Discussion
   1. Expressing Existing Programming Models
   2. Leveraging RDDs for Debugging
8. Related Work
9. Conclusion

## Background & Motivation

* Programmability
  * Most real applications require multiple MapReduce stages:
    * Google indexing pipeline: 21 steps
    * Analytics queries: 2-5 steps
    * Iterative algorithms: 10s of steps
  * Multi-step jobs create spaghetti code
    * 21 MapReduce steps -> 21 mapper & reducer classes
* Performance
  * MapReduce only provides one pass of computation (must write data to file system in between)
  * Expensive for apps that need to reuse data (e.g., multi-step algorithms like PageRank, interactive data mining)

## Design and Implementation

### Apache Spark

* Simple, functional API
  * 5x - 10x less code than MapReduce
  * Parallel transformations on collections
  * Available in Scala, Python, Java, and R
  * Can trace how operations are chained -> free type checking!
  * Mimics local programs
* Performance
  * In-memory computing primitives
    * In-mem caching: LRU
    * Caches also get cleared when workers go away
  * Optimization across operators
* Lazy evaluation/execution
* Fault tolerance
  * Lineage graphs: Records of transformations that created this RDD
    * Assumption: Input file is still available
    * Assumption: Storage for lineage graphs is stable (similar to the MapReduce master)
* Job scheduling
  * Captures RDD dependency graph
  * Pipelines function into "stages"
  * Cache-aware for data reuse, locality (move computation to where data is cached)
  * Partition-aware to avoid shuffles

### RDDs

* Immutable, partitioned collection of objects
* Can be cached in memory for faster reuse
* Operations on RDDs: Transformations (build RDDs) & Actions (compute results)

## Links & References

* [Paper PDF](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf)
* [CS 744 course notes](https://pages.cs.wisc.edu/\~shivaram/cs744-fa21-slides/cs744-spark-notes.pdf)
