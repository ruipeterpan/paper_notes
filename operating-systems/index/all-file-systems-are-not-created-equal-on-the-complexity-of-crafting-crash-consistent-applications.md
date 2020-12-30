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

To provide crash consistency for update-in-place file systems, journaling is performed. A high-level overview of journaling:

* Intuition
  * Before updating the file system, write a note describing the update
  * Make sure note is safely on disk
  * Once the note is safe, update the file system
  * If interrupted, read the note and redo updates
* Protocol
  * Write the data \(no pointers to it\) - Optional
  * Write the note: Journal Metadata
  * Make sure the note is durably written: Journal Commit
  * Update the in-place metadata: Checkpointing
  * Replay the note: Recovery



## Design and Implementation

## Evaluation

## New Vocabulary

* 
## Links

* [Paper PDF](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-pillai.pdf)
* [Presentation Video at OSDI '14](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/pillai)
* [Presentation Slides at OSDI '14](https://www.usenix.org/sites/default/files/conference/protected-files/osdi14_slides_pillai.pdf)

