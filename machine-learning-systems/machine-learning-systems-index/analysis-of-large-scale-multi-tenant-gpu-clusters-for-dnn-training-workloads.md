# \[2019 ATC] Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads

## One-line Summary

This paper presents a characterization study of large-scale GPU clusters for DNN training. It uncovers some inefficiencies in cluster utilization and presents some lessons for better cluster manager decisions.

This paper means a lot to me in that this is the first paper that I read through thoroughly :)

## Paper Structure Outline

1. Introduction
2. Philly: System Overview
   1. Workloads
   2. Cluster Architecture
   3. Job Scheduling and Execution Workflow
   4. Data Collection and Analysis
3. Impact of Locality Awareness
   1. Queueing Delays
      1. Impact of Locality-Driven Scheduling
   2. GPU utilization
      1. Impact of Distributed Learning
4. Training Progress and Completion
   1. Effectiveness of Training Iterations
   2. Job Failures
      1. Failure Classification
      2. Failure Frequency
      3. Runtime to Failure
      4. Impact of GPUs Allocated
5. Design Implications for Future Schedulers
6. Related Work
7. Conclusion

## Background & Motivation

DNN-based workloads are different from traditional big data analytics workloads in two ways:

1. Cluster utilization: GPUs represent a monolithic resource that cannot be shared at a fine granularity across users
2. Workload: Deep learning frameworks require gang scheduling, reducing the flexibility of scheduling and job's elasticity of runtime failures

The authors first present an overview of Philly, a large, multi-tenant GPU-based cluster for production-scale deep learning tasks. Then, they present a detailed workload characterization and study how factors like gang scheduling, locality requirements, and failures might affect cluster utilization.

## Microsoft Philly

![](<../../.gitbook/assets/Screen Shot 2020-12-31 at 3.42.57 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2020-12-31 at 4.02.21 PM.png>)

The three main steps in Fig. 1 are:

1. **Incoming jobs and queuing**: The scheduler needs to perform gang scheduling while being locality-aware. Each production group is provided with a virtual cluster and a quota (in terms of #GPUs to each virtual cluster).
2. **Job placement and utilization**: The scheduler aims to maximize locality and minimize fragmentation of resources (from smaller jobs, e.g. 1-GPU jobs). There is a trade-off between colocation and distribution, though, as placing different jobs on the same server could lead to lower GPU utilization (because of interference in shared resources like RDMA and PCIe).
3. **Training progress and completion**: Jobs can finish with three statuses: passed, killed, or unsuccessful. Failed jobs are retried a few times to overcome non-deterministic failures.

The logs are collected over a 75-day period and it consists of 96260 jobs over 14 virtual clusters. There are three main sources of the logs:

1. YARN scheduler log: job arrival time, # GPUs requested, GPU allocation status, job finish status
2. stdout & stderr logs from ML frameworks
3. Ganglia monitoring system log: Per-minute statistics on hardware usage (CPU, memory, network, GPU utilization)

## Impact of Locality Awareness

### Queueing Delays

![The scheduler works in practice to trade-off locality for lower scheduling delay](<../../.gitbook/assets/Screen Shot 2020-12-31 at 5.44.01 PM.png>)

![There are two types of delays: Fair-share denotes fairness (which is common in conventional data analytics clusters), while fragmentation denotes locality requirement and resource fragmentation (which is more prevalent in DL clusters).](<../../.gitbook/assets/Screen Shot 2020-12-31 at 5.44.25 PM.png>)

### GPU Utilization

![GPU utilization is low (lower in distributed training) because of (1) distribution across servers and (2) intra-server interference](<../../.gitbook/assets/Screen Shot 2020-12-31 at 5.36.39 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2020-12-31 at 5.55.48 PM.png>)

![GPU utilization when running 8 and 16 GPU jobs on dedicated servers](<../../.gitbook/assets/Screen Shot 2020-12-31 at 5.56.15 PM.png>)

![In general, DL training jobs underutilize GPU processing cycles regardless of their job sizes.](<../../.gitbook/assets/Screen Shot 2020-12-31 at 5.56.57 PM.png>)

Relaxing locality constraints:

* High intra-server locality
  * Pros
    * High communication efficiency
  * Cons
    * Long queueing time
* Low intra-server locality
  * Pros
    * Low queueing time
  * Cons
    * Contention in the use of network
    * Risk of intra-server interference (across jobs)

## Training Progress and Completion

![A significant fraction (30.7%) of jobs are either termintated by users or are unsuccessful. These jobs constitute \~55% of the total GPU time.](<../../.gitbook/assets/Screen Shot 2020-12-31 at 5.58.08 PM.png>)

As observed in the above table, it is important to understand the reasons behind these failures, as fewer unsuccessful jobs would mean more resources for successful jobs.

### Training Iterations

![](<../../.gitbook/assets/Screen Shot 2020-12-31 at 6.02.05 PM.png>)

\~80% of passed jobs require all epochs executed to reach the lowest loss. However, an average of 62% and 56% (for passed jobs and killed jobs, respectively) GPU times for each job are used to improve the convergence accuracy by merely 0.1%. This suggests that jobs can be terminated early to save considerable resources.

### Job Failures

![](<../../.gitbook/assets/Screen Shot 2020-12-31 at 6.06.22 PM.png>)

The failures happen across the whole stack: Infrastructure (GPU, HDFS, resource scheduler), ML frameworks (PyTorch, TensorFlow), and user programs (shitty code :P). A failure classifier is used to analyze the causes of the job failures. The most important failures are from these classifications:

1. **Incorrect inputs**: Model files or input data stored in the external HDFS storage cannot be read
2. **Semantic error**: Errors that happen due to library version mismatch or other dependencies of the user training program not being setup correctly
3. **Model checkpoint error**: The job is not able to successfully create a model checkpoint after a certain number of epochs complete. This is usually due to either transient error in HDFS or HDFS name node recovery
4. **MPI runtime failure**: This is usually due to either a failure of network connection to peer MPI process, or possibly an internal failure of the MPI daemon itself
5. **Job preempted**: YARN reclaims any GPU currently in use to schedule another job
6. **Invalid memory access**: Training job dies because of violating access on memory address space e.g., using an invalid pointer value, or having race condition while copying data. This failure is observed in both CPU memory and memory allocated for GPU access

An analysis on the failure frequency show that:

1. Failures repeat for the same job/user
2. User/programming errors lead to a lot of failures

An analysis on the runtime to failure (RTF) shows that:

1. RTF exhibits high variability with many short RTFs
2. Infrastructure failures occur infrequently but have much longer RTF

The authors also find that large jobs with programming semantic errors tend to fail a while after execution.

## Guidelines

1. **Prioritize locality**: As the lack of locality impacts both utilization and job runtime, and because DNN training jobs are long-running, schedulers should trade queueing delay for adhering to locality constraints.
2. **Mitigate interference**: As different jobs on a single server might interfere with each other, schedulers should aim to isolate the jobs on dedicated servers while implementing techniques like migration for defragmentation to support the locality constraints of jobs that need more GPUs.
3. **Improve failure handling**: To catch failures early before they are scheduled on a cluster and thus prevent resources from being wasted, each incoming job should be scheduled on a small dedicated pool of servers/a single GPU to catch simple programming and configuration errors from multi-GPU jobs. Another possible improvement is for clusters to predictively mitigate failures by proactively observing related failures. For example, the scheduler should stop retrying for failure categories like incorrect data input and continue retrying for network timeouts.

## Links

* [Paper PDF](https://www.usenix.org/system/files/atc19-jeon.pdf)
* [Full presentation video at USENIX ATC '19](https://www.youtube.com/watch?v=FoA1M7wAZ3I\&ab\_channel=USENIX)
* [Lightning talk at USENIX '19](https://www.youtube.com/watch?v=ClEpCcZru\_Q\&ab\_channel=MyeongjaeJeon)
* [Full presentation slides](https://www.usenix.org/sites/default/files/conference/protected-files/atc19-slides-jeon.pdf)
* [Lightning talk slides](https://www.usenix.org/sites/default/files/conference/protected-files/atc19\_slides\_lt\_jeon.pdf)
* [Philly traces on GitHub](https://github.com/msr-fiddle/philly-traces)
* [Project Fiddle](https://www.microsoft.com/en-us/research/project/fiddle/)
