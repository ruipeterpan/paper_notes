# seL4: Formal Verification of an OS Kernel

## One-line Summary

seL4 is the first operating system microkernel that guarantees functional correctness by adding formal verifications.

## Paper Structure Outline

1. Introduction
2. Overview
   1. seL4 programming model
   2. Kernel design process
   3. Formal verification
3. Kernel Design for Verification
   1. Global variables and side effects
   2. Kernel memory management
   3. Concurrency and non-determinism
   4. I/O
4. seL4 Verification
   1. Abstract specification
   2. Executable specification
   3. C implementation
   4. Machine model
   5. The proof
5. Experience and Lessons Learnt
   1. Performance
   2. Verification effort
   3. The cost of change
6. Related Work
7. Conclusions

## Background & Motivation

The security and reliability of a computer system depend on that of the underlying kernel. With microkernels, formal verifications guarantee the absence of bugs.

## Design and Implementation

## Evaluation

> Traditionally, formal veriﬁcation was limited to small codebases and languages other than C, but recently, through heroic efforts of several OS researchers, formal veriﬁcation was applied to OS kernels \[11, 12, 21\]. Even so, these state-of-the-art methods do not apply to the bugs described here, because they are limited to single-threaded environments and do not have a way of reasoning about time. Formal tools work by describing the system as series of state transitions, pre-conditions and post-conditions, and then reason whether any state transitions may lead to violation of post-conditions given the possible pre-conditions. The problem is that in our environment, short and intermittent violations of post-conditions \(i.e., idle cores in the presence of waiting threads\) are totally acceptable. It is the long-term violations that are problematic. Unfortunately, existing tools do not have the mechanisms allowing to reason how timing affects transient violation of invariants. Extending these tools to work in multithreaded environments and to reason about time could make them more suitable, but having Linux developers write formal speciﬁcations will be another hurdle for adoption of these tools.
>
> ---- The Linux Scheduler: a Decade of Wasted Cores

## New Vocabulary

* Formal verification: It is the process of checking whether the behavior of a system satisfies some properties \(correctness, etc.\) with respect to a certain formal specification or property, using formal methods of mathematics.
* Microkernel: A microkernel is a kernel that provides minimal services of process and memory management. The user services and kernel services are implemented in different address spaces. The advantages of microkernels include:
  * The architecture is small and isolated, so it functions better
  * It's easily expandable if new services are to be added to the user address space
* Classes of kernels: There are in general three types of kernels: monolithic, microkernel, and hybrid. See [this article](https://somedudesays.com/2020/02/monolithic-kernels-microkernels-and-everything-in-between/) for detailed explanations of each.
* [The L4 microkernel family](https://en.wikipedia.org/wiki/L4_microkernel_family): A family of microkernels generally used to implement UNIX-like operating systems.

## Links

* [Paper PDF](https://www.sigops.org/s/conferences/sosp/2009/papers/klein-sosp09.pdf)
* [The seL4 Microkernel](https://sel4.systems/)
* [The seL4 Microkernel Whitepaper](https://sel4.systems/About/seL4-whitepaper.pdf)
* [Lecture slides from CS 262a @ Berkeley by Prof. Ion Stoica and Ali Ghodsi](https://ucbrise.github.io/cs262a-fall2020/notes/16-microkerenels-Mach-seL4.pdf)











