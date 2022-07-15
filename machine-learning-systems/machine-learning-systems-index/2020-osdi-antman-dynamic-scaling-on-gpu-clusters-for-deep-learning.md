# \[2020 OSDI] AntMan: Dynamic Scaling on GPU Clusters for Deep Learning

## Summary

AntMan is a cluster scheduler for GPU sharing. It introduces two techniques, dynamic memory scaling and opportunistic computation management, to accommodate multiple jobs and avoid interference.

![System architecture/workflow](<../../.gitbook/assets/Screen Shot 2022-07-15 at 4.53.17 PM.png>)

## Background & Motivation

* GPUs in a shared cluster are not properly utilized (both SM and GRAM are under-utilized). One of the reasons is multi-GPU jobs require gang scheduling, which creates GPU idleness. Moreover, DL training jobs have dynamic resource demand over time.&#x20;
* Training jobs in the Alibaba cluster have the following characterstics:
  * Small model size: Most GPU memory can be shared
  * Short mini-batch: Fast resource coordination
  * Similar mini-batch: Mini-batch time can be used to quantify inter-job interference

![](<../../.gitbook/assets/Screen Shot 2022-07-15 at 4.30.16 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2022-07-15 at 4.34.09 PM.png>)

## Design & Implementation

### Dynamic Memory Scaling

AntMan dynamically co-locates jobs on shared GPUs. The goal is for resource-guarantee jobs to maintain the same performance as dedicated execution while co-locating opportunistic jobs to best utilize the resources.

AntMan monitors the memory usage of DL jobs and sets the corresponding memory upper bounds, allowing other jobs to utilize the spare memory. However, since DL jobs have dynamic resource demand, jobs may require more memory than before, which creates OOM and fails all jobs. In this case (Fig. 7a), these memory bursts are cached on the host (CPU) memory, and are moved back to GRAM after re-allocation. The same technique is applied to jobs that need to shrink their memory requirements to make way for other jobs (Fig. 7b).&#x20;

![](<../../.gitbook/assets/Screen Shot 2022-07-15 at 4.38.39 PM.png>)

### Computation management for minimizing interference

The GpuOpManager is introduced in DL frameworks to opportunistically launch computation kernels during idle time slots to reduce interference.

![](<../../.gitbook/assets/Screen Shot 2022-07-15 at 4.50.50 PM.png>)

## Evaluation

![Micro benchmark 1: Memory scaling](<../../.gitbook/assets/Screen Shot 2022-07-15 at 4.58.02 PM.png>)

![Micro benchmark 2: Computation management. Here, ESPnet is a resource-guaranteed job, while ResNet50 is an opportunistic job.](<../../.gitbook/assets/Screen Shot 2022-07-15 at 4.58.25 PM.png>)

![End-to-end evaluation](<../../.gitbook/assets/Screen Shot 2022-07-15 at 5.01.01 PM.png>)

## Links & References

* [Paper PDF](https://www.usenix.org/system/files/osdi20-xiao.pdf)
* [Presentation video at OSDI '20](https://www.youtube.com/watch?v=8PSzcqL0eUA)
* [Presentation slides at OSDI '20](https://www.usenix.org/sites/default/files/conference/protected-files/osdi20\_slides\_xiao.pdf)
* [GPU-cluster-for-deep-learning on GitHub](https://github.com/alibaba/GPU-scheduler-for-deep-learning)
