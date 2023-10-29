# OptFS: Optimistic Crash Consistency

## One-line Summary

The authors present the optimistic crash consistency and an optimistic journaling system (OptFS) which implements optimistic crash consistency and maintains consistency to the same extent as pessimistic journaling while achieving the same performance as with probabilistic consistency. &#x20;

## Paper Structure Outline

1. Introduction
2. Pessimistic Crash Consistency
   1. Disk Interface
   2. Pessimistic Journaling
   3. Flushing Performance Impact
3. Probabilistic Crash Consistency
   1. Quantifying Probabilistic Consistency
   2. Factors affecting P\_inc (probability of inconsistency)
      1. Workload
      2. Queue Size
   3. Summary
4. Optimistic Crash Consistency
   1. Asynchronous Durability Notification
   2. Optimistic Consistency Properties
   3. Optimistic Techniques
      1. In-Order Journal Recovery
      2. In-Order Journal Release
      3. Checksums
      4. Background Write after Notification
      5. Reuse after Notification
      6. Selective Data Journaling
   4. Durability vs. Consistency
5. Implementation of OptFS
   1. Asynchronous Durability Notifications
   2. Handling Data Blocks
   3. Optimistic Techniques
6. Evaluation
   1. Reliability
   2. Performance
   3. Resource consumption
   4. Journal size
7. Case Studies
   1. Atomic Update within Gedit
   2. Temporary Logging in SQLite
8. Related Work
9. Conclusion

## Background & Motivation

![\~5x performance difference based on flushing!](<../../.gitbook/assets/Screen Shot 2021-01-04 at 8.58.01 PM.png>)

In file system journaling, pessimistic journaling (default) incurs extra work due to unnecessary flushing (assuming crash does not happen). In probabilistic journaling, typical operations may or may not result in much reordering, so the disk is only sometimes in an inconsistent state and thus flushes can be disabled. Although probabilistic crash consistency does not guarantee consistency after a crash, many practitioners use it due to performance degradation from flushes.

![The authors define crash inconsistency probability as the proportion of vulnerability window (due to re-ordering) length. P\_inc = Time in window(s) / Total I/O time.](<../../.gitbook/assets/Screen Shot 2021-01-04 at 9.52.38 PM.png>)

![If a workload is mostly read oriented, there is little chance of inconsistency, as the FS state is not update frequently. Early commit is the largest contributor to P\_inc, accounting for over 90% of inconsistency across all workloads.](<../../.gitbook/assets/Screen Shot 2021-01-04 at 9.57.30 PM.png>)

![Types of re-ordering](<../../.gitbook/assets/Screen Shot 2021-01-04 at 10.10.46 PM.png>)

{% hint style="info" %}
* D: Data write
* J\_M: Loggin Metadata
* J\_C: Logging Commit
* M: Checkpointing
{% endhint %}

The idea of optimistic crash consistency comes from the optimistic concurrency control (OCC) from distributed transaction systems.

## Design and Implementation

OptFS decouples `fsync()` into two novel primitives: `dsync()` for immediate durability as well as ordering, and `osync()` for write ordering/atomic updates but only eventual durability.

### Optimistic techniques

A number of techniques are used: In-order journal recovery and release, checksums, background writes after notification, reuse after notification, selective data journaling.

#### Checksums

Checksums (over D and J\_M into J\_C) remove the need for ordering writes. Optimistic crash consistency eliminates the need for ordering during transaction commit by generalizing metadata transactional checksums to include data blocks. During recovery, transactions are discarded upon checksum mismatch.

#### Asynchronous Durability Notifications (ADN)

ADNs are used to delay checkpointing a transaction until it has been committed durably (M is only written when D, J\_M and J\_C are all written). Fortunately, this delay does not affect application performance, as applications block until the transaction is committed, not until it is checkpointed. Additional techniques are required for correctness in scenarios such as block reuse and overwrite. ADNs improve performance because:

* The disk can schedule blocks from the cache to platter in the best order
* The file system can do other work while waiting for ADN
* (Main) The user applications do not have to wait for ADN

#### Selective data journaling

![Selective data journaling places both data and metadata in the journal to keep locality](<../../.gitbook/assets/Screen Shot 2021-01-04 at 10.08.01 PM.png>)

### Implementation

OptFS is implemented as a variant of the ext4 file system inside Linux 3.2.

## Evaluation

![OptFS achieves good reliability](<../../.gitbook/assets/Screen Shot 2021-01-04 at 9.19.17 PM.png>)

![OptFS has a good performance when crashes are rare and I/Os are infrequent. However, overwrites have a bad performance](<../../.gitbook/assets/Screen Shot 2021-01-04 at 9.21.17 PM.png>)

![The FS needs to consume more CPU & memory for maintaining extra information like for delayed checkpointing.](<../../.gitbook/assets/Screen Shot 2021-01-04 at 9.22.21 PM.png>)

![Among the many techniques for ensuring consistency, OptFS is the only one that is flexible enough to accomodate all circumstances and has the best {consistency, performance, availability, durability}.](<../../.gitbook/assets/Screen Shot 2021-01-04 at 9.01.59 PM.png>)

## New Vocabulary

* [Optimistic concurrency control (OCC)](https://en.wikipedia.org/wiki/Optimistic\_concurrency\_control)
* [Checksums](https://www.howtogeek.com/363735/what-is-a-checksum-and-why-should-you-care/)

## Links

* [Paper PDF](https://research.cs.wisc.edu/adsl/Publications/optfs-sosp13.pdf)
* [Presentation video at SOSP '13](https://www.youtube.com/watch?v=3BXYHBOU3Ec\&ab\_channel=AssociationforComputingMachinery%28ACM%29)
* [Source code for OptFS (Linux ext4 implementation)](https://research.cs.wisc.edu/adsl/Software/optfs/)
* Thanks to Guanzhou Hu & Pei-Hsuan Wu for the paper review notes!

{% file src="../../.gitbook/assets/OptFS+Alice.pptx" %}
Prof. Andrea's CS 736 course slides on ALICE and OptFS
{% endfile %}
