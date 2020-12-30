# All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications

## One-line Summary

> If you understand this, you get the basic idea. But as with all complicated systems, the devil is in the details.

## Paper Structure Outline

1. Introduction
2. Persistence Properties
   1. An Example
   2. Study and Results
      1. Atomicity
      2. Ordering
   3. Summary
3. The Application-Level Intelligent Crash Explorer \(ALICE\)
   1. Usage
   2. Crash States and APMs
      1. Logical Operations
      2. Abstract Persistence Models
      3. Constructing crash states.
   3. Finding Application Requirements
   4. Static Vulnerabilities
   5. Implementation
   6. Limitations
4. Application Vulnerabilities
   1. Workloads and Checkers
   2. Overview
      1. Databases and Key-Value Stores
      2. Version Control Systems
      3. Virtualization and Distributed Systems
   3. Vulnerabilities Found
   4. Common Patterns
      1. Atomicity across System Calls
      2. Atomicity within System Calls
      3. Ordering between System Calls
      4. Durability
      5. Summary
   5. Impact on Current File Systems
   6. Evaluating New File-System Designs
   7. Discussion
5. Related Work
6. Conclusion

## Background & Motivation

## Design and Implementation

## Evaluation

## New Vocabulary

* 
## Links

* Paper PDF
* Presentation Video at xxx
* Presentation Slides
* * xxx on GitHub
* 
