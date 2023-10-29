# Scheduler Activations: Effective Kernel Support for the User-Level Management of Parallelism

## One-line Summary

The authors present a new kernel interface and user-level thread package that provide the same functionality as kernel threads without compromising the performance and flexibility advantages of user-level management of parallelism.

## Paper Structure Outline

1. Introduction
   1. The Problem
   2. The Goals of this Work
   3. The Approach
2. User-Level Threads: Performance Advantages and Functionality Limitations
   1. The Case for User-Level Thread Management
   2. Sources of Poor Integration in User-Level Threads Built on the Traditional Kernel Interface
3. Effective Kernel Support for the User-Level Management of Parallelism
   1. Explicit Vectoring of Kernel Events to the User-Level Thread Scheduler
   2. Notifying the Kernel of User-Level Events Affecting Processor Allocation
   3. Critical Sections
4. Implementation
   1. Processor Allocation Policy
   2. Thread Scheduling Policy
   3. Performance Enhancements
   4. Debugging Consideration
5. Performance
   1. Thread Performance
   2. Upcall Performance
   3. Application Performance
6. Related Ideas
7. Summary

## Background & Motivation

Threads are built either at the user-level or kernel-level.

* User-level
  * Advantages
    * Requires no kernel intervention, good performance: fast thread mgmt operations (context switches)
    * Flexible: Customizable for applications
  * Limitations
    * Poor integration with system services: Implemented over kernel-level threads, which block and are preempted w/o notifying user-level thread
    * Performs poorly during I/O, preemption, and page faults due to overhead of kernel trapping
    * Scheduled obliviously w.r.t. the user-level thread state
* Kernel-level
  * Advantages
    * Each thread gets mapped to a physical processor while it is running
  * Limitations
    * Bad performance: Requires kernel intervention (switch into the kernel) for thread mgmt operations (fork, join, wait, signal, etc.)
    * Not as flexible (implemented in the kernel, so the scheduling policy cannot be changed easily later on)

The authors argue that kernel-level threads are inherently worse than user-level threads (extra kernel trap and copy operations), but IRL user-level threads many exhibit poor performance/incorrect behavior in multiprocessor systems. They then attempt to take the best of both worlds by building a new kernel interface and a variant of a user-level thread library that communicates effectively with the kernel to combine the functionality of kernel-level threads and the performance and flexibility of user-level threads.

## Design and Implementation

Main contributions:

* The kernel allocates a virtual multiprocessor to each application
* Applications:
  * Told how many and which processors it has
  * A user-level application's own thread scheduler decides which threads to run on its allocated physical processors
* OS:
  * The kernel is told how many threads an application would like to run so it can try to allocate that many physical processors for it
  * Complete control over which processors are given to which application

Scheduler Activations (SA) is a kernel mechanism that provides a communication structure between the kernel processor and the user-level thread system. This is a vectored event that causes the user-level thread system (via an up-call) to reconsider its scheduling decision of which threads to run on which processors when events (processor allocations and deallocations) need to take place.

The following roles are performed by SAs:

* Notify the user-level thread system of kernel events
* Provides the execution context for execution of user-level threads
* Provides space for saving user-level thread context in the kernel when a thread is stopped by the kernel

> Scheduler Activation Upcalls:
>
> * Add this processor
> * Processor has been preempted
> * Scheduler activation has blocked
> * Scheduler activation has unblocked
>
> Application downcalls:
>
> * Add more processors
> * This processor is idle

![Things that happen on an I/O request/completion. T1: Add two processors, user-level library picks two threads. T2: Thread 1 on SA A blocks in kernel, notified of that with a new SA C, library picks to run T3 on SA C. T3: Thread 1 finishes I/O, for the kernel to notify the user-level, take SA from B and use SA D to tell library both 1 and 2 can continue. T4: Use SA D to run t1.](<../../.gitbook/assets/Screen Shot 2020-12-31 at 1.03.40 PM.png>)

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2020-12-31 at 1.09.34 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2020-12-31 at 1.10.09 PM.png>)

> * Performance degrades slowly when available memory drops, and then more sharply once the application's working set does not fit in memory
> * Application performance with original FastThreads degrades more quickly:  a user level thread blocks in the kernel →  the application loses that physical processor for the duration of the I/O.
> * Application performance is better with modified FastThreads than with Topaz because most thread operations can be implemented without kernel involvement.

{% hint style="info" %}
* Topaz: Kernel-level thread
* orig FastThrds: User-level thread
* new FastThrds: Scheduler activation
{% endhint %}

## Links

* [Paper PDF](https://flint.cs.yale.edu/cs422/doc/sched-act.pdf)
* [CS 736 reviews from Spring 2015's offering](http://pages.cs.wisc.edu/\~swift/classes/cs736-sp15/blog/2015/03/scheduler\_activations\_effectiv.html)
* [Course slides from CS 443 @ Northwestern](http://www.cs.northwestern.edu/\~fabianb/classes/cs-443-s05/SchedAct.pps)
* Reading notes from [U of Waterloo](https://cs.uwaterloo.ca/\~brecht/servers/readings/Summaries/Seltzer-OS/readings/anderson-1992.html) and [Stanford](http://infolab.stanford.edu/\~daswani/quals/anderson92%20-%20scheduler%20activations.htm)
* Thanks to Jiaxin Lin for the paper review notes!

{% file src="../../.gitbook/assets/16-SchedAct+ResourceContainers.pptx" %}
CS 736 course slides on Scheduler Activations and Resource Containers
{% endfile %}
