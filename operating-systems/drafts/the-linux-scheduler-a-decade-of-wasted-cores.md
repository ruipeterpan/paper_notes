# The Linux Scheduler: a Decade of Wasted Cores

### One-liner Summary

~~Linus gets roasted.~~

![](../../.gitbook/assets/screen-shot-2020-12-04-at-12.21.52-am.png)

This paper pinpoints some performance bugs in the Linux scheduler and proposes fixes, during which the authors developed tools for checking and understanding these bugs.

### Paper Structure Outline

1. Introduction
2. The Linux Scheduler
   1. On a single-CPU system, CFS is very simple
   2. On multi-core systems, CFS becomes quite complex
      1.  The load balancing algorithm
      2. Optimizations
3. Bugs
   1. The Group Imbalance bug
   2. The Scheduling Group Construction bug
   3. The Overload-on-Wakeup bug
   4. The Missing Scheduling Domains bug
   5. Discussion
4. Tools
   1. Online Sanity Checker
   2. Scheduler Visualization tool
5. Lessons Learned
6. Related Work 
7. Conclusion

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








