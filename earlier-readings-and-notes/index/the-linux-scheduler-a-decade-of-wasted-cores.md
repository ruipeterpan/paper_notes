# The Linux Scheduler: a Decade of Wasted Cores

## One-line Summary

~~Linus gets roasted.~~

![](<../../.gitbook/assets/Screen Shot 2020-12-04 at 12.21.52 AM.png>)

This paper pinpoints some performance bugs in the Linux scheduler (especially in multi-core systems) and proposes fixes, during which the authors developed tools for checking and understanding these bugs.

![](<../../.gitbook/assets/Screen Shot 2020-12-20 at 4.57.45 PM.png>)

## Paper Structure Outline

1. Introduction
2. The Linux Scheduler
   1. On a single-CPU system, CFS is very simple
   2. On multi-core systems, CFS becomes quite complex
      1. &#x20;The load balancing algorithm
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
6. Related Work&#x20;
7. Conclusion

## Background & Motivation

The Linux kernel's process scheduler underwent three main periods of evolutions:&#x20;

1. v0.01\~v2.4.x: [the very first scheduler](https://dev.to/satorutakeuchi/a-brief-history-of-the-linux-kernel-s-process-scheduler-the-very-first-scheduler-v0-01-9e4)
2. v2.6.0\~v2.6.22: O(1) scheduler
3. v2.6.23\~: Completely Fair Scheduler (CFS)

{% hint style="info" %}
There were also variations like the Brain Fuck Scheduler (BFS) which works better than CFS on desktop Linux systems with <16 cores, but it does not scale well to large-scale systems (4096 processors / NUMA), and it had some other drawbacks so it was never merged into the mainline Linux kernel.
{% endhint %}

Modern Linux uses a Completely Fair Scheduler (CFS) which implements the Weighted Fair Queueing (WFQ) algorithm. On a single-CPU system, the CFS is really simple -- the CFS does time-slicing among running threads to achieve fair sharing. On multi-core systems, however, things get a bit messy -- To address scalability and keep the context switches fast, per-core runqueues are used, and in order for the scheduling algorithm to work correctly and efficiently, the runqueues must be kept balanced. The optimizations done by the load-balancing algorithm is complex and lead to bugs.

## Bugs & Performance Improvement after Fixing&#x20;

### The Group Imbalance bug

![](<../../.gitbook/assets/Screen Shot 2020-12-15 at 11.15.37 PM.png>)

> When a core attempts to steal work from another node, or, in other words, from another scheduling group, it does not examine the load of every core in that group, it only looks at the group’s average load. If the average load of the victim scheduling group is greater than that of its own, it will attempt to steal from that group; otherwise it will not. This is the exact reason why in our situation the underloaded cores fail to steal from the overloaded cores on other nodes. They observe that the average load of the victim node’s scheduling group is not any greater than their own. The core trying to steal work runs on the same node as the high-load R thread; that thread skews up the average load for that node and conceals the fact that some cores are actually idle. At the same time, cores on the victim node, with roughly the same average load, have lots of waiting threads.

The fix is simple: When the algorithm compares the load of scheduling groups, it should be comparing the minimum loads instead of the average loads.&#x20;

> If the minimum load of one scheduling group is lower than the minimum load of another scheduling group, it means that the ﬁrst scheduling group has a core that is less loaded than all cores in the other group, and thus a core in the ﬁrst group must steal from the second group. This algorithm ensures that no core of the second group will remain overloaded while a core of the ﬁrst group has a smaller load, thus balancing the load across cores.

### The Scheduling Group Construction bug

This bug requires specific hardware topology to trigger (2 nodes that are two hops apart).

![](<../../.gitbook/assets/Screen Shot 2020-12-20 at 4.40.57 PM.png>)

The root cause for this bug is that scheduling group construction is not adapted to modern NUMA machines. In the above example, the first two scheduling examples look like this:

`{0, 1, 2, 4, 6}, {1, 2, 3, 4, 5, 7}`

Notice how node 1 and 2 are included in both scheduling groups. The fix is to modify the construction of scheduling groups so that each core uses scheduling groups constructed from its perspective.

### The Overhead-on-Wakeup bug

![](<../../.gitbook/assets/Screen Shot 2020-12-20 at 4.45.26 PM.png>)

When a thread goes to sleep on node X and the thread that wakes it up later is running on that same node, the scheduler only considers the cores of node X for scheduling the awakened thread. If all cores of node X are busy, the thread will miss opportunities to use idle cores on other machines. The fix is to alter the code that is executed when a thread wakes up.

> We wake up the thread on the local core – i.e. the core where the thread was scheduled last – if it is idle; otherwise if there are idle cores in the system, we wake up the thread on the core that has been idle for the longest amount of time. If there are no idle cores, we fall back to the original algorithm to find the core where the thread will wake up.

### The Missing Scheduling Domains bug

When a core is disabled and then re-enabled using the `/proc` interface, load balancing between any NUMA nodes is no longer performed.

![](<../../.gitbook/assets/Screen Shot 2020-12-20 at 4.51.23 PM.png>)

During code refractoring, the Linux developers dropped the call to the function that regenerates domains across NUMA nodes. The fix is to simply add it back.

### Performance improvements after the bug fixes

![](<../../.gitbook/assets/Screen Shot 2020-12-20 at 4.56.29 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2020-12-20 at 4.56.53 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2020-12-20 at 4.57.20 PM.png>)



## Tools

The authors developed two tools that help with understanding the bugs:

1. **Online sanity checker**: It verifies that no core is idle while other core's runqueue has waiting threads. It's fine if such conditions exist for a short period, but an alert is raised if it persists.
2. **Scheduler visualizer**: It shows the scheduling activity over time. Some of the graphs (e.g., figure 3) was produced using the tool.

## Discussion

* Scheduling is complicated. More optimizations to the scheduler will be proposed due to the fast-evolving hardware. With optimizations comes complexity, and the scheduler should be designed so that it can integrate the modularized optimizations with ease.
* Visualization is important. Checking the aforementioned bugs using conventional tools is tricky.

## New Vocabulary

* LKML: The Linux Kernel Mailing List

## Links

* [Paper PDF](https://www.ece.ubc.ca/\~sasha/papers/eurosys16-final29.pdf)
* [Presentation Slides](http://www.i3s.unice.fr/\~jplozi/wastedcores/files/talk.pdf)
* [Patches for Linux kernel 4.1 on GitHub](https://github.com/jplozi/wastedcores)
* [Discussion 1 on LKML](https://lkml.org/lkml/2016/4/23/194)
* [Discussion 2 on LKML](https://lkml.org/lkml/2016/4/25/176)
* [Paper reading notes](https://blog.acolyer.org/2016/04/26/the-linux-scheduler-a-decade-of-wasted-cores/) on [the morning paper](https://blog.acolyer.org/)
* [A brief history of the Linux Kernel's process scheduler: The very first scheduler, v0.01](https://dev.to/satorutakeuchi/a-brief-history-of-the-linux-kernel-s-process-scheduler-the-very-first-scheduler-v0-01-9e4)
