# ReVirt: Enabling Intrusion Analysis through Virtual-Machine Logging and Replay

### One-liner Summary

### Paper Structure Outline

1. Introduction
2. Virtual Machines
3. UMLinux
   1. UMLinux Structure and Operation
   2. Trusted Computing Base for UMLinux
4. Logging and Replaying UMLinux
   1. Overview
   2. ReVirt
   3. Cooperative Logging
   4. Alternative Architectures for Logging and Replay
   5. Using ReVirt to Analyze Attacks
5. Experiments
   1. Virtualization Overhead
   2. Validating ReVirt Correctness
   3. Logging and Replaying Overhead
   4. Analyzing an Attack
6. Related Work
7. Future Work
8. Conclusions

### Background & Motivation

A typical system relies on event loggers \(login attempts, TCP connection requests, file system mount requests, etc.\) for post-attack analysis. Current system loggers have two problems: They lack integrity because they assume the operating system kernel is trustworthy. Also, they lack completeness as they do not log enough information to recreate some attacks in hindsight. ReVirt removes the dependency on the OS by moving it into a virtual machine and logging below the virtual machine. ReVirt also logs enough information to replay a long-term execution of the virtual machine instruction-by-instruction.

### Design and Implementation

### Evaluation

### New Vocabulary

* 
### Links

* [Paper PDF](https://web.eecs.umich.edu/virtual/papers/dunlap02.pdf)
* Presentation Video at xxx
* Presentation Slides
* * xxx on GitHub
* 








