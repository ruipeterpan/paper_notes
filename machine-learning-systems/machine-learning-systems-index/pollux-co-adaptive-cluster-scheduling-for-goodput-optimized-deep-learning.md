# \[2021 OSDI] Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning

## One-line Summary

Pollux co-adaptively optimizes DL job execution at both per-job level and cluster-wide level. At the per-job level, Pollux uses GNS to dynamically tune the batch size to optimize goodput, a new metric that considers both system throughput and statistical efficiency of DL training. At the cluster-wide level, Pollux optimizes the generalized mean goodput of all jobs, alongside cluster-level goals including fairness and JCT.

## Paper Structure Outline

1. Introduction
2. Background: Distributed DL Training
   1. System Throughput
   2. Statistical Efficiency
   3. Existing DL Schedulers
3. The Goodput of DL Training and Pollux
   1. Modeling Statistical Efficiency
   2. Modeling System Throughput
4. Pollux Design and Architecture
   1. PolluxAgent: Job-Level Optimization
   2. PolluxSched: Cluster-wise Optimization
   3. Implementation
5. Evaluation
   1. Experimental Setup
   2. Testbed Macrobenchmark Experiments
   3. Simulator Experiments
      1. Scheduling Fairness
      2. Other Effects on Scheduling
   4. More Applications of Pollux
      1. Cloud Auto-scaling
      2. Hyper-parameter Optimization (HPO)
   5. Artifact
6. Additional Related Work
7. Conclusion
8. Acknowledgments

## Background & Motivation

### Motivation: System Throughput & Statistical Efficiency, Dynamicity in DL Training Jobs

* Training using larger batch sizes increases the system throughput
* However, as batch sizes increase, potential issues may occur:
  * If the learning rate is not tuned accordingly, the final model quality may be suboptimal
  * Increasing the batch size decreases the statistical efficiency of DL training
    * Statistical efficiency: training progress per unit of training data processed
  * Even further increasing the bs results in worse model generalization (in terms of validation performance, due to unknown reasons)
* Gradient Noise Scale (GNS) measures the noise-to-signal ratio of the stochastic gradient. It allows training jobs to increase their batch sizes later on during training without hurting the statistical efficiency.

![Suboptimal statistical efficiency after increasing batch size](<../../.gitbook/assets/Screen Shot 2021-07-19 at 1.25.41 PM.png>)

![To achieve the best training performance, we should aim for the point where statistical efficiency \* system throughput reaches its maximum](<../../.gitbook/assets/Screen Shot 2021-07-19 at 1.46.02 PM.png>)

![How jobs can check GNS and increase their batch sizes without hurting the statistical efficiency. This graph is excerpted from Kungfu (OSDI '20)](<../../.gitbook/assets/Screen Shot 2021-07-19 at 1.23.10 PM.png>)

### Background: Existing DL Schedulers

* Non-scale-adaptive schedulers are not aware of jobs' performance scalabilities w.r.t. the amount of allocated resources
  * [Tiresias](tiresias-a-gpu-cluster-manager-for-distributed-deep-learning.md) requires users to specify #GPUs during job submission
  * [Gandiva](gandiva-introspective-cluster-scheduling-for-deep-learning.md) does the same, and although it may dynamically change the number of GPUs used by a job, it does so opportunistically
* Scale-adaptive schedulers automatically decide the amount of resources allocated to each job to speed up jobs
  * Optimus: Learns a predictive model for the system throughput given different amounts of resources, and optimizes the avg JCT
  * SLAQ: Minimize the avg loss values for training general ML models
  * [Gavel](gavel-heterogeneity-aware-cluster-scheduling-policies-for-deep-learning-workloads.md): Takes into account the performance heterogeneity of underlying accelerators
  * AntMan: Uses dynamic scaling & fine-grained GPU sharing to improve cluster utilization, resource fairness, and JCTs
  * [Themis](themis-fair-and-efficient-gpu-cluster-scheduling.md): Introduces the notion of finish time fairness
* Importantly, existing schedulers/policies are agnostic to the statistical efficiency of DL training and the inter-dependence of resource decisions and training parameters.

## Design

### Goodput = Throughput \* Statistical Efficiency

* For each job, Pollux optimizes for a new metric called the goodput

![Goodput definition](<../../.gitbook/assets/Screen Shot 2021-07-19 at 1.45.07 PM.png>)

* When a user submits the job, he/she submits an initial batch size and learning rate, and Pollux will run the job with these initial hyperparameters (with s=0). As the job progresses, Pollux learns and refines predictive models for both throughput and efficiency through profiling. Then, Pollux periodically re-tunes (a, m, s).
* Learning rate scaling: Pollux allows users to implement/select scaling rules like AdaScale, square-root scaling, linear scaling, etc.

### Modeling Statistical Efficiency

* The statistical efficiency E is measured relative to the initial batch size & learning rate -> 0 < E <= 1, and training using batch size M will need to process 1/E times as many training examples to make the same progress as using batch size M0.

![Definition of pre-conditioned GNS](<../../.gitbook/assets/Screen Shot 2021-07-19 at 2.33.00 PM.png>)

![Expression for efficiency](<../../.gitbook/assets/Screen Shot 2021-07-19 at 2.01.59 PM.png>)

* During training, Pollux estimates the value of ϕt, then uses the efficiency expression to predict the efficiency at different batch sizes. Note that ϕt varies according to the training progress at iteration t.

![Takeaways: (1) For larger batach sizes, the initial efficiency is low, but this improves later in training. (2) The model can accurately predict the efficiency of using a different batch size without training at that batch size](<../../.gitbook/assets/Screen Shot 2021-07-19 at 2.37.54 PM.png>)

### Modeling System Throughput

* Pollux separately models T\_grad, time for local gradient computations, and T\_sync, the avg time spent in each iteration for gradient averaging/model synchronizations:
  * Tgrad: The run time scales linearly with the per-GPU batch size m. Thus, Tgrad is modeled as `Tgrad(m) = α_grad + β_grad * m`, where α and β are fittable parameters.
  * Tsync: For single-GPU jobs, Tsync = 0. Otherwise, Pollux models Tsync by using a linear function of #GPUs in data parallelism and taking into account the performance retrogressions when using 3+ GPUs (due to stragglers/network bottlenecks)
    * N: Number of physical nodes occupied by at least one replica
    * α and β (local, sync): Constant and retrogression parameters for when all processes are co-located onto the same node
    * α and β (node, sync): Analogous parameters for when at least two processes are located on different nodes
    * This model can also account for rack-level locality by adding a third pair of parameters

![Modeling Tsync](<../../.gitbook/assets/Screen Shot 2021-07-19 at 2.54.55 PM.png>)

* Overlapping computation and synchronization: Then, Pollux combines Tgrad and Tsync. If there is no overlapping between gradient computation and synchronization, then `Titer = Tgrad + Tsync`. With perfect overlapping, `Titer = max(Tgrad, Tsync)`. Realistically, Pollux sets a learnable parameter γ to express the level of overlapping
  * γ >= 1. When γ == 1, there is no overlapping. As γ→∞, it transitions towards perfect overlapping.
* Gradient accumulation is a technique that allows for larger batch sizes, bypassing the GPU memory constraints ([quick tutorial](https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html#2.-What-is-graident-accumulation): instead of updating the network weights on every batch, we save gradient values, proceed to the next batch, and add up the new gradients). With s = 0, there is no gradient accumulation. Otherwise, one iteration of SGD spans s accumulation steps and one synchronization step.
* Combining all the above, we have:

![](<../../.gitbook/assets/Screen Shot 2021-07-19 at 2.01.05 PM.png>)

* ... and the predictions are pretty good!

![](<../../.gitbook/assets/Screen Shot 2021-07-19 at 3.07.57 PM.png>)

* Note that there are many other factors (except number and co-locality of GPUs, bs, gradient accumulation steps in Pollux) that may affect the data-parallel throughput, including specialized hardware, sophisticated synchronization algorithms, different parallelization strategies, larger scales, or hidden resource contention. Pollux does not cover those, but it designs the goodput metric so that different equations for throughput may be easily plugged in. Dayum!

## Implementation

![](<../../.gitbook/assets/Screen Shot 2021-07-19 at 3.10.23 PM.png>)

### Job-level optimization: PolluxAgent

* An instance of PolluxAgent is started with each training job
* PolluxAgent then measures GNS & throughput, fits the EFFICIENCY and THROUGHPUT functions for that job, and tunes its batch size and learning rate for efficient utilization of its current allocated resources
* Finally, PolluxAgnet periodically reports to PolluxSched

### Cluster-wide optimization: PolluxSched

* PolluxSched periodically optimizes the resources allocations for jobs in the cluster to maximize FITNESS, taking into account the goodput function for each job and cluster-wide resource contention. The decisions also account for {re-allocation overhead, slowdowns due to network interference between jobs, resource fairness}

![Note that af is a fair resource allocation for the job, defined to be an exclusive 1/J share of the cluster. This is similar to finish time fairness, but SPEEDUP is related to training performance at a moment in time, whereas FTF is related to end-to-end JCT.](<../../.gitbook/assets/Screen Shot 2021-07-19 at 10.10.51 PM.png>)

* Re-allocation penalty: The per-job SPEEDUP is applied a penalty, which results in jobs with historically higher rates of re-allocations being penalized more for future re-allocations.
* Interference avoidance: Pollux simply disallows different distributed nodes from sharing the same node.
* Non-adaptive jobs: EFFICIENCY is fixed to be 1, and Pollux can continue to adapt its resource allocations based on system throughput.

## Evaluation

* [Tiresias](tiresias-a-gpu-cluster-manager-for-distributed-deep-learning.md) and Optimus are used as the baseline schedulers. Optimus only adapts the number of GPUs and Tiresias adapts neither.
* For baseline schedulers, manually-tuned jobs are used as follows. A "#GPUs" is considered valid if using the optimal batch size for that "#GPUs" achieves 50% - 80% of the ideal/linear scalability vs. using the optimal batch size on a single GPU. For each job, the #GPUs and batch size are selected randomly from its set of valid configurations.
  * These configurations assume that users are highly rational and knowledgeable about the scalability of the models (...) in favor of the baseline schedulers
  * Less than 50% -> undertilization of resources
  * More than 80% -> more GPUs can still be utilized efficiently
* Pollux does really well in trading-off throughput with efficiency. For example, during periods of low cluster contention, Pollux can allocate more GPUs (& larger batch sizes) to boost training throughput & goodput, even if it decreases the statistical efficiency, and vice versa

![](<../../.gitbook/assets/Screen Shot 2021-07-19 at 10.42.57 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-07-19 at 10.45.49 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-07-19 at 10.46.01 PM.png>)

## Links

* [Paper PDF](https://www.pdl.cmu.edu/PDL-FTP/CloudComputing/osdi21-pollux.pdf)
* [Presentation video at OSDI '21](https://www.usenix.org/conference/osdi21/presentation/qiao)
* [Presentation slides at OSDI '21](https://www.usenix.org/system/files/osdi21\_slides\_qiao.pdf)
* [Pollux on GitHub](https://github.com/petuum/adaptdl/tree/osdi21-artifact)
* My [presentation video](https://youtu.be/lJ3\_iM13A5k) on Pollux (in Mandarin Chinese) at the Systems & Networking Reading Group hosted by [Xiangfeng Zhu](https://xzhu27.me/)
