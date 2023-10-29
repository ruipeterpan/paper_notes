# \[2018 OSDI] Gandiva: Introspective Cluster Scheduling for Deep Learning

## One-line Summary

The authors present Gandiva, a cluster scheduling framework that employs techniques like time-slicing, migration, intra-job elasticity, and dynamic priority.

## Paper Structure Outline

1. Introduction
2. Background
3. DLT Job Characteristics
   1. Sensitivity to locality
   2. Sensitivity to interference
   3. Intra-job predictability
4. Design
   1. Mechanisms
   2. Scheduling Policy
      1. Reactive Mode
      2. Introspective Mode
5. Implementation
   1. Scheduler
   2. Modifications to DL toolkits
6. Evaluation
   1. Micro-benchmarks
   2. Model exploration in a multi-job
   3. Cluster experiments: time-slicing and packing
   4. Cluster experiments: time-slicing and migration
7. Related Work
8. Conclusion

## Background & Motivation

Today's DNN schedulers (e.g., YARN, Kubernetes) treat deep learning jobs naively (as if they are traditional big-data jobs): A job is scheduled on a set of GPUs exclusively, and the job holds the GPUs until completion. There are some problems:

1. High Latency (head-of-line blocking): Long DNN jobs have runtimes of hours and days, so we need time-slicing of jobs. However, GPUs are not efficiently virtualizable.
2. Low Efficiency (fixed decision at the job-placement time): Need the ability to migrate jobs, and the sensitivity to locality varies across jobs.

DLT jobs have the following characteristics:

1. Sensitivity to locality: Different models have various levels of sensitivity to intra-server and inter-server locality that a DLT scheduler needs to take into account.
2. Sensitivity to interference: Similarly, different models demonstrate different levels of sensitivity to interference between jobs.
3. Intra-job predictability: DLT jobs' GPU memory usage reveals a pattern (goes up during forward pass of a minibatch and goes down during backward pass). Gandiva leverages this in three ways:
   1. A job can be split into mini-batch iterations
   2. If suspend/resume is performed during the nadir, less amount of memory needs to be copied from GPU to CPU
   3. The progress rate can be profiled to evaluate the effectiveness of mechanisms

![](<../../.gitbook/assets/Screen Shot 2021-01-19 at 3.35.37 PM.png>)

![When suspending a job, as GPUs are not efficiently virtualizable, the state needs to be moved from GPU to CPU before suspension.](<../../.gitbook/assets/Screen Shot 2021-01-19 at 3.36.40 PM.png>)

## Design and Implementation

![](<../../.gitbook/assets/Screen Shot 2021-01-19 at 3.43.07 PM.png>)

Gandiva employs the following mechanisms:

1. Suspend-Resume and Packing
   1. Suspend-Resume: Intra-job predictability is leveraged to suspend/resume DLT jobs when their GPU usage is at the lowest.
   2. Packing: Run multiple jobs on a GPU simultaneously and let the GPU time-share the jobs, with the premise that the packed jobs do not interfere with each other. It is only considered during overload.
2. Migration: The set of GPUs assigned to a job can be changed for (1) moving time-sliced jobs to vacated GPUs, (2) moving interfering jobs away from each other, and (3) doing de-fragmentation of the cluster. The migration overhead is as little as a second or two.
3. Grow-Shrink: # GPUs available for a job can be increased during idle times and shrank when the load goes up.
4. Profiling: Gandiva profiles each job's time for one forward/backward pass over a minibatch. With this, Gandiva introspects DLT jobs to estimate the rate of progress, e.g. to check if packing helped.

Gandiva's scheduler works in two modes: reactive and introspective. The reactive mode handles events such as job arrivals/departures and machine failures, while the introspective mode monitors and optimizes job placement to improve the overall utilization and the completion time.

![](<../../.gitbook/assets/Screen Shot 2021-01-19 at 4.11.59 PM.png>)

## Evaluation

![Microbenchmark: Time-slicing](<../../.gitbook/assets/Screen Shot 2021-01-19 at 4.38.14 PM.png>)

![Microbenchmark: Packing](<../../.gitbook/assets/Screen Shot 2021-01-19 at 4.38.37 PM.png>)

![Microbenchmark for AutoML: Gandiva provides much faster hyper-parameter exploration](<../../.gitbook/assets/Screen Shot 2021-01-19 at 4.45.16 PM.png>)

![Cluster utilization](<../../.gitbook/assets/Screen Shot 2021-01-19 at 4.43.39 PM.png>)

## New Vocabulary

* Introspection (反省): The examination of one's own conscious thoughts and feelings.

## Links

* [Paper PDF](https://www.usenix.org/system/files/osdi18-xiao.pdf)
* [Presentation audio at OSDI '18](https://www.usenix.org/conference/osdi18/presentation/xiao)
* [Presentation slides at OSDI '18](https://www.usenix.org/sites/default/files/conference/protected-files/osdi18\_slides\_sivathanu.pdf)
* [Presentation video by Muthian Sivathanu, one of the authors and a UW-Madison alumni](https://www.youtube.com/watch?v=i4YOKOLsyFI\&ab\_channel=MicrosoftResearch)
