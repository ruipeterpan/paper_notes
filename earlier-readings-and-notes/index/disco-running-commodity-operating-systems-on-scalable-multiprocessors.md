# Disco: Running Commodity Operating Systems on Scalable Multiprocessors

## One-line Summary

Disco uses virtual machines to run multiple commodity operating systems on large-scale shared-memory multiprocessors. Disco VMM hides NUMA-ness from non-NUMA aware OSes, requires low effort to implement, and introduces moderate overhead due to virtualization.

## Paper Structure Outline

1. Introduction
2. Problem Description
3. A Return to Virtual Machine Monitors
   1. Challenges Facing Virtual Machines
4. Disco: A Virtual Machine Monitor
   1. Disco's Interface
   2. Implementation of Disco
      1. Virtual CPUs
      2. Virtual Physical Memory
      3. NUMA Memory Management
      4. Virtual I/O Devices
      5. Copy-on-write Disks
      6. Virtual Network Interface
   3. Running Commodity Operating Systems
      1. Necessary Changes for MIPS Architecture
      2. Device Drivers
      3. Changes to the HAL
      4. Other Changes to IRIX
   4. SPLASHOS: A Specialized Operating System
5. Experimental Results
   1. Experimental Setup and Workloads
   2. Execution Overheads
   3. Memory Overheads
   4. Scalability
   5. Dynamic Page Migration and Replication
6. Related Work
   1. System Software for Scalable Shared Memory Machines
   2. Virtual Machine Monitors
   3. Other System Software Structuring Techniques
   4. ccNUMA Memory Management
7. Conclusions

## Background & Motivation

The motivation is to enable existing commodity operating systems to handle Non-Uniform Memory Access (NUMA) architectures. Instead of modifying existing operating systems to run on scalable shared-memory multiprocessors, an additional layer (VM monitor) is inserted between the hardware and the OS.

![Course notes by Prof. Andrea. Left: SMP (symmetrical multiprocessor uniform memory access machine), right: cc-NUMA](<../../.gitbook/assets/Screen Shot 2021-01-03 at 7.41.45 PM.png>)

Cache-coherent Non-Uniform Memory Architecture (cc-NUMA) makes hardware scalable, while SMP ensures the same performance to all memory from everywhere. Both ensure correctness, though.

## Design and Implementation

![Disco is a layer between OSes and hardware](<../../.gitbook/assets/Screen Shot 2021-01-03 at 7.46.13 PM.png>)

The advantages of using virtual machines in the context of this work are:

* The Disco layer understands the NUMA architecture
* It's a portability layer
* Monitors are smaller and easier to understand & trust than operating systems
* Allows to run different OSes concurrently (almost unmodified)

The drawbacks of using virtual machines are:

* Overhead: cost of virtualizing
  * Time: VMM (Disco) acts as an emulator. Most instructions can just run, but privileged instructions + TLB instructions must be trapped & emulated
  * Space: Multiple copies (OS code & each OS's file cache) waste memory
* Resource management: Lack of information to make good policy decisions
  * Lost information about what is being used
    * CPU - idle thread
    * Memory - pages on the free list
* Communication and Sharing problems:
  * Hard to communicate between standalone VMs
  * Most OSes require exclusive access to disks

![High-level challenges of using virtual machines](<../../.gitbook/assets/Screen Shot 2021-01-03 at 7.52.20 PM.png>)

![How Disco virtualizes CPU. Three priviliged levels: user, supervisor, and kernel.](<../../.gitbook/assets/Screen Shot 2021-01-03 at 7.53.38 PM.png>)

![How Disco virtualizes memory. Users generate virtual addresses, OS translates to physical addreses, Disco translates to machine addresses.](<../../.gitbook/assets/Screen Shot 2021-01-03 at 7.56.46 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.04.35 PM.png>)

![Records copy-on-write to track shared data efficiently.](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.27.31 PM.png>)

![Send becomes additional mapping (emulate device); Copy becomes additional mapping](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.27.39 PM.png>)

![Changes Disco made to IRIX to improve performance](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.29.50 PM.png>)

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.36.57 PM.png>)

![Pmake & Database do a lot of syscalls, often traps into Disco, which then goes to kernel. The extra 16% overhead for those workloads is due to the extra work handling TLB misses. The kernel time being less is because Disco zeros the pages (does work instead of IRIX).](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.37.10 PM.png>)

![Disco does a good job sharing buffer cache space across VMs and sharing IRIX text.](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.40.11 PM.png>)

![No migration + replication, just looking at how much more scalable is Disco than IRIX due to optimizations of not having locks in which IRIX does a bad job at. IRIX on 8-processor cc-NUMA machine. 2VM -> 8VM actually improves because Disco does not have bad lock](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.40.22 PM.png>)

![Much less time accessing remote memory, more local memory.](<../../.gitbook/assets/Screen Shot 2021-01-03 at 8.49.23 PM.png>)

This paper started off VMWare (which was founded by authors of Disco in 1998 and successfully commercialized this work) and revived virtual machines for the next 20 years. Now VMs are commodities, and every cloud provider and virtually every enterprise uses VMs today.

## New Vocabulary

* IRIX: A variety of UNIX System V with BSD extensions.
* NUMA: [What is NUMA?](https://www.kernel.org/doc/html/v4.18/vm/numa.html)

## Links

* [Paper PDF](https://bob.cs.ucdavis.edu/assets/ecs251/bugnion97.pdf)
* [Paper review notes from CS 443 @ Northwestern by Joseph Paris](https://users.cs.northwestern.edu/\~fabianb/classes/cs-443-s05/review-disco-jparis.pdf)
* [Discussion panel from CS 736 @ UW-Madison](http://pages.cs.wisc.edu/\~swift/classes/cs736-fa12/blog/2012/09/disco\_running\_commodity\_operat.html)
* [Lecture slides from CS 262a @ Berkeley by Prof. Ion Stoica and Ali Ghodsi](https://ucbrise.github.io/cs262a-spring2018/notes/10-VMs-Disco-Xen.pdf)

{% file src="../../.gitbook/assets/21-Disco-Instructor-Notes Combined with Questions.pdf" %}
Prof. Andrea's notes on Disco
{% endfile %}
