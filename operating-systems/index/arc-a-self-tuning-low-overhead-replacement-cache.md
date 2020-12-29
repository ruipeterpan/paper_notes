# ARC: A Self-Tuning, Low Overhead Replacement Cache

## One-line Summary

> If you understand this, you get the basic idea. But as with all complicated systems, the devil is in the details.

## Paper Structure Outline

1. Introduction
   1. The Problem
   2. Our Contributions
   3. A Brief Outline of the Paper
2. Prior Work: A Brief Review
   1. Offline Optimal
   2. Recency
   3. Frequency
   4. Recency and Frequency
   5. Temporal Distance Distribution
   6. Caching using Multiple Experts
   7. Ghost Caches
   8. Summary
3. A Class of Replacement Policies
   1. Double Cache and a Replacement Policy
   2. A New Class of Policies
   3. LRU
4. Adaptive Replacement Cache
   1. Fixed Replacement Cache
   2. The Policy
   3. Learning
   4. Scan-Resistant
   5. Extra History
5. Experimental Results
   1. Traces
   2. OLTP
   3. Two Traces: P8 and P12
   4. ARC and 2Q
   5. ARC and MQ
   6. ARC and LRU
   7. ARC is Self-Tuning and Empirically Universal
   8. A Closer Examination of Adaptation in ARC
6. Conclusions

## Background & Motivation

## Design and Implementation

## Evaluation

## New Vocabulary

* 
## Links

* [Paper PDF](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
* [Presentation slides by the authors @ University of Houston](http://www2.cs.uh.edu/~paris/6360/PowerPoint/ARC.ppt)

