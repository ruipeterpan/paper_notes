# PyTorch Distributed: Experiences on Accelerating Data Parallel Training

### One-liner Summary

This paper presents the design, implementation, and evaluation of the PyTorch distributed data parallel module.

### Paper Structure Outline

1. INTRODUCTION
2. BACKGROUND
   1. PyTorch
   2. Data Parallelism
   3. AllReduce
3. SYSTEM DESIGN
   1. API
   2. Gradient Reduction
      1. A Naive Solution
      2. Gradient Bucketing
      3. Overlap Computation with Communication
      4. Gradient Accumulation
   3. Collective Communication
4. IMPLEMENTATION
   1. Python Front-end
   2. Core Gradient Reduction
5. EVALUATION
   1. Latency Breakdown
   2. Bucket Size
   3. Scalability
   4. Round-Robin Process Group
6. DISCUSSION
   1. Lessons Learned
   2. Future Improvements
      1. Gradient Order Prediction
      2. Layer Dropping
      3. Gradient Compression
7. RELATED WORK
8. CONCLUSION

### Background & Motivation

### Design and Implementation

### Evaluation

### New Vocabulary

* 
### Links

* [Paper PDF](https://arxiv.org/pdf/2006.15704.pdf)









