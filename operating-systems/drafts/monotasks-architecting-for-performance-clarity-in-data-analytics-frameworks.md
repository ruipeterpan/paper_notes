# Monotasks: Architecting for Performance Clarity in Data Analytics Frameworks

### One-liner Summary

Instead of breaking jobs into tasks that pipeline many resources, we break the jobs into monotasks , each of which uses a single resource. This contributes to accelerating performance in data analytics frameworks, and it also makes the analysis on performance bottlenecks easier for users.

### Paper Structure Outline

1. INTRODUCTION
2. BACKGROUND
   1. Architecture of data analytics frameworks
   2. The challenge of reasoning about performance
3. MONOTASKS ARCHITECTURE
   1. Design
   2. How are multitasks decomposed into monotasks?
   3. Scheduling monotasks on each worker
   4. How many multitasks should be assigned concurrently to each machine?
   5. How is memory access regulated?
4. IMPLEMENTATION
5. MONOTASKS PERFORMANCE
   1. Experimental setup
   2. Does getting rid of fine-grained pipelining hurt performance?
   3. When is MonoSpark slower than Spark?
   4. When is MonoSpark faster than Spark?
6. REASONING ABOUT PERFORMANCE
   1. Modeling performance
   2. Predicting runtime on different hardware
   3. Predicting runtime with deserialized data
   4. Predicting with both hardware and software changes
   5. Understanding bottlenecks
   6. Can this model be used for Spark?
7. LEVERAGING CLARITY: AUTO-CONFIGURATION
8. LIMITATIONS AND OPPORTUNITIES
9. RELATED WORK
10. CONCLUSION

### Background & Motivation

### Design and Implementation

### Evaluation

### New Vocabulary

* 
### Links

* Paper PDF
* Presentation Video at xxx
* Presentation Slides
* * xxx on GitHub
* 








