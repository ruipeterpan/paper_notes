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

* LKML: The Linux Kernel Mailing List
* 
### Links

* [Paper PDF](https://www.ece.ubc.ca/~sasha/papers/eurosys16-final29.pdf)
* [Presentation Slides](http://www.i3s.unice.fr/~jplozi/wastedcores/files/talk.pdf)
* [Patches for Linux kernel 4.1 on GitHub](https://github.com/jplozi/wastedcores)
* [Discussion 1 on LKML](https://lkml.org/lkml/2016/4/23/194)
* [Discussion 2 on LKML](https://lkml.org/lkml/2016/4/25/176)
* [Paper reading notes](https://blog.acolyer.org/2016/04/26/the-linux-scheduler-a-decade-of-wasted-cores/) on [the morning paper](https://blog.acolyer.org/)
* 








