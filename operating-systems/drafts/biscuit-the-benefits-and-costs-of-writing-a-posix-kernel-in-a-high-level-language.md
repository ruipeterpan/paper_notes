# Biscuit: The benefits and costs of writing a POSIX kernel in a high-level language

## One-line Summary

This paper analyzes \(duh\) the benefits and costs of writing a POSIX kernel in a high-level language, Go.

## Paper Structure Outline

1. Introduction
2. Related work
3. Motivation
   1. Why C?
   2. Why an HLL?
4. Overview
5. Garbage Collection
   1. Go's collector
   2. Biscuit's heap size
6. Avoiding heap exhaustion
   1. Approach: reservations
   2. How Biscuit reserves
   3. Static analysis to find s
      1. Basic MAXLIVE operation
      2. Handling loops
      3. Kernel threads
   4. Limitations
   5. Heap exhaustion summary
7. Implementation
8. Evaluation
   1. Biscuit's use of HLL features
   2. Potential to reduce bugs
   3. Experimental Setup
   4. HLL tax
   5. GC delays
   6. Sensitivity to heap size
   7. Go versus C
      1. Ping-pong
      2. Page-faults
   8. Biscuit versus Linux
   9. Handling kernel heap exhaustion
   10. Lock-free lookups
9. Discussion and future work
10. Conclusions

## Background & Motivation

## Design and Implementation

## Evaluation

## New Vocabulary

* 
## Links

* [Paper PDF](https://www.usenix.org/system/files/osdi18-cutler.pdf)
* [Presentation Audio at OSDI '18](https://www.usenix.org/conference/osdi18/presentation/cutler)
* [Presentation Slides](https://www.usenix.org/sites/default/files/conference/protected-files/osdi18_slides_cutler.pdf)











