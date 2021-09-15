# AlloX: Compute Allocation in Hybrid Clusters

## One-line Summary

Allocate interchangeable resources in a hybrid cluster \(CPU, GPU, FPGA, problem-specific accelerator\);

Scheduling -&gt; Min-cost bipartite matching problem, provides dynamic fair allocation

reduce avg jct while providing fairness and preventing starvation

## Paper Structure Outline

1. Introduction
2. Background & Motivation
   1. Interchangeable Resources
3. Algorithm Design
   1. Optimal Approach for Queued Up Jobs
      1. Generate input for the matching problem
      2. Solve the matching problem
      3. Convert the matching solution to job scheduling
   2. Handling Online Arrivals
   3. Incorporating Fairness
      1. Existing Fair Allocation Algorithms are Insufficient
      2. Our Idea
      3. Incorporating Fairness into AlloX
4. AlloX Implementation
   1. Estimator
   2. Scheduler
   3. Placer
   4. Operational Issues
5. Evaluation
   1. Experimental Methodology
      1. Baselines
   2. AlloX Performance
      1. Experiments on a Cluster
      2. Simulation Results
      3. Starvation
      4. Performance and Fairness Trade-offs
   3. Sensitivity Analysis
      1. Estimation Errors
      2. Profiling Overhead
6. Related Work
7. Concluding Remarks
8. Acknowledgments

## Background & Motivation

### Background X: Heterogeneous/interchangeable resources

CPU, GPU, distinct speedup rates for different applications

### Motivation X: Current schedulers do not consider 

best fit: pick the optimal config for each job. This creates load imbalance \(heavy load on GPUs while CPUs idle\)

join the shortest queue: choose resource with the shortest completion time. This is short-sighted \(each job optimizes for itself w/o considering later jobs\)

shortest job first: maintain ordered queue of all jobs by increasing processing time. Whenever a resource becomes avaialable, scheulde job wiht the shortest run time



## Design and Implementation



## Evaluation



## Links

* [Paper PDF](https://www.mosharaf.com/wp-content/uploads/allox-eurosys20.pdf)

