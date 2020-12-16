# LegoOS: A Disseminated, Distributed OS for Hardware Resource Disaggregation

### One-liner Summary

The traditional monolithic server model in datacenters is having issues in resource utilization, elasticity, heterogeneity, and failure handling. LegoOS breaks down traditional OS functionalities into hardware components like Lego bricks and connects them with fast networks.

### Paper Structure Outline

1. Introduction
2. Disaggregate Hardware Resource
   1. Limitations of Monolithic Servers
   2. Hardware Resource Disaggregation
   3. OSes for Resource Disaggregation
3. The Splitkernel OS architecture
4. LegoOS Design
   1. Abstraction and Usage Model
   2. Hardware Architecture
   3. Process Management
      1. Process Management and Scheduling
      2. ExCache Management
      3. Supporting Linux Syscall Interface
   4. Memory Management
      1. Memory Space Management
      2. Optimization on Memory Accesses
   5. Storage Management
   6. Global Resource Management
   7. Reliability and Failure Handling
5. LegoOS Implementation
   1. Hardware Emulation
   2. Network Stack
   3. Processor Monitor
   4. Memory Monitor
   5. Storage Monitor
   6. Experience and Discussion
6. Evaluation
   1. Micro- and Macro-benchmark Results
   2. Application Performance
   3. Failure Analysis
7. Related Work
8. Discussion and Conclusion
9. 
### Background & Motivation

### Design and Implementation

### Evaluation

### New Vocabulary

* Monolithic server: A single server that contains all the hardware resources \(typically a processor, some main memory, and a disk or an SSD\) that are needed to run a user program.

### Links

* [Paper PDF](https://www.usenix.org/system/files/osdi18-shan.pdf)
* [Presentation Video at OSDI '18](https://www.youtube.com/watch?v=GX74Q2-ZOQE)
* [Presentation Video at USENIX ATC '19](https://www.youtube.com/watch?v=KJqYHuL59_s)
* [Presentation Slides](https://www.usenix.org/sites/default/files/conference/protected-files/osdi18_slides_shan.pdf)
* [LegoOS on GitHub](https://github.com/WukLab/LegoOS)









