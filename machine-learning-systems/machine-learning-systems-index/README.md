# Machine Learning Systems - Index

### Frameworks & Paradigms

| Title                                                                         | Venue    | Links                                                                                                                                                      |
| ----------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scaling Distributed Machine Learning with the Parameter Server                | OSDI '14 | \[[notes](scaling-distributed-machine-learning-with-the-parameter-server.md)] \[[pdf](http://www.cs.cmu.edu/\~muli/file/parameter\_server\_osdi14.pdf)]    |
| BytePS: A High Performance and Generic Framework for Distributed DNN Training | OSDI '20 | \[[notes](byteps-a-high-performance-and-generic-framework-for-distributed-dnn-training.md)] \[[pdf](https://www.usenix.org/system/files/osdi20-jiang.pdf)] |
| PyTorch Distributed: Experiences on Accelerating Data Parallel Training       | VLDB '20 | \[[notes](pytorch-distributed-experiences-on-accelerating-data-parallel-training.md)] \[[pdf](https://arxiv.org/pdf/2006.15704.pdf)]                       |

### Machine Learning Training

| Title                                                                                    | Venue       | Links                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism            | NeurIPS '19 | \[[notes](gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism.md)] \[[pdf](https://papers.nips.cc/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf)]                 |
| PipeDream: Generalized Pipeline Parallelism for DNN Training                             | SOSP '19    | \[[notes](pipedream-generalized-pipeline-parallelism-for-dnn-training.md)] \[[pdf](https://arxiv.org/pdf/1806.03377.pdf)]                                                                               |
| BlueConnect: Decomposing All-Reduce for Deep Learning on Heterogeneous Network Hierarchy | MLSys '19   | \[notes] \[[pdf](https://mlsys.org/Conferences/2019/doc/2019/130.pdf)]                                                                                                                                  |
| Is Network the Bottleneck of Distributed Training?                                       | NetAI '20   | \[[notes](is-network-the-bottleneck-of-distributed-training.md)] \[[pdf](https://dl.acm.org/doi/pdf/10.1145/3405671.3405810)]                                                                           |
| ZeRO: memory optimizations toward training trillion parameter models                     | SC '20      | \[[notes](zero-memory-optimizations-toward-training-trillion-parameter-models.md)] \[[pdf](https://arxiv.org/pdf/1910.02054.pdf)]                                                                       |
| ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning              | SC '21      | \[notes] \[[pdf](https://arxiv.org/pdf/2104.07857.pdf)]                                                                                                                                                 |
| ZeRO-Offload: Democratizing Billion-Scale Model Training                                 | ATC '21     | \[notes] \[[pdf](http://pasalabs.org/papers/2021/ATC21\_zero-offload.pdf)]                                                                                                                              |
| Accordion: Adaptive Gradient Communication via Critical Learning Regime Identification   | MLSys '21   | \[[notes](accordion-adaptive-gradient-communication-via-critical-learning-regime-identification.md)] \[[pdf](https://proceedings.mlsys.org/paper/2021/file/1d7f7abc18fcb43975065399b0d1e48e-Paper.pdf)] |
| CheckFreq: Frequent, Fine-Grained DNN Checkpointing                                      | FAST '21    | \[[notes](checkfreq-frequent-fine-grained-dnn-checkpointing.md)] \[[pdf](https://www.usenix.org/system/files/fast21-mohan.pdf)]                                                                         |
| Analyzing and Mitigating Data Stalls in DNN Training                                     | VLDB '21    | \[[notes](analyzing-and-mitigating-data-stalls-in-dnn-training.md)] \[[pdf](https://www.cs.utexas.edu/\~vijay/papers/vldb21-datastalls.pdf)]                                                            |
|                                                                                          |             | \[notes] \[pdf]                                                                                                                                                                                         |
|                                                                                          |             | \[notes] \[pdf]                                                                                                                                                                                         |

### Machine Learning Serving

| Title                                                                      | Venue         | Links                                                                      |
| -------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------- |
| Serving DNNs like Clockwork: Performance Predictability from the Bottom Up | OSDI '20      | \[notes] \[[pdf](https://www.usenix.org/system/files/osdi20-gujarati.pdf)] |
| INFaaS: A Model-less Inference Serving System                              | ATC '21       | \[notes] \[[pdf](https://www.usenix.org/system/files/atc21-romero.pdf)]    |
| Interference-Aware Scheduling for Inference Serving                        | EuroMLSys '21 | \[notes] \[[pdf](https://dl.acm.org/doi/pdf/10.1145/3437984.3458837)]      |
|                                                                            |               | \[notes] \[pdf]                                                            |



### Scheduling, Resource Allocation & Management

| Title                                                                                         | Venue     | Links                                                                                                                                                                      |
| --------------------------------------------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DRF: Fair Allocation of Multiple Resource Types                                               | NSDI '11  | \[[notes](dominant-resource-fairness-fair-allocation-of-multiple-resource-types.md)] \[[pdf](https://cs.stanford.edu/\~matei/papers/2011/nsdi\_drf.pdf)]                   |
| Gandiva: Introspective Cluster Scheduling for Deep Learning                                   | OSDI '18  | \[[notes](gandiva-introspective-cluster-scheduling-for-deep-learning.md)] \[[pdf](https://www.usenix.org/system/files/osdi18-xiao.pdf)]                                    |
| Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads                  | ATC '19   | \[[notes](analysis-of-large-scale-multi-tenant-gpu-clusters-for-dnn-training-workloads.md)] \[[pdf](https://www.usenix.org/system/files/atc19-jeon.pdf)]                   |
| Tiresias: A GPU Cluster Manager for Distributed Deep Learning                                 | NSDI '19  | \[[notes](tiresias-a-gpu-cluster-manager-for-distributed-deep-learning.md)] \[[pdf](https://www.usenix.org/system/files/nsdi19-gu.pdf)]                                    |
| ByteScheduler: A Generic Communication Scheduler for Distributed DNN Training Acceleration    | SOSP '19  | \[[notes](bytescheduler-a-generic-communication-scheduler-for-distributed-dnn-training-acceleration.md)] \[[pdf](https://i.cs.hku.hk/\~cwu/papers/yhpeng-sosp19.pdf)]      |
| Themis: Fair and Efficient GPU Cluster Scheduling                                             | NSDI '20  | \[[notes](themis-fair-and-efficient-gpu-cluster-scheduling.md)] \[[pdf](https://www.usenix.org/system/files/nsdi20-paper-mahajan.pdf)]                                     |
| Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications                     | MLSys '20 | \[[notes](salus-fine-grained-gpu-sharing-primitives-for-deep-learning-applications.md)] \[[pdf](https://www.mosharaf.com/wp-content/uploads/salus-mlsys20.pdf)]            |
| Gavel: Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads            | OSDI '20  | \[[notes](gavel-heterogeneity-aware-cluster-scheduling-policies-for-deep-learning-workloads.md)] \[[pdf](https://cs.stanford.edu/\~matei/papers/2020/osdi\_gavel.pdf)]     |
| Wavelet: Efficient DNN Training with Tick-Tock Scheduling                                     | MLSys '21 | \[[notes](wavelet-efficient-dnn-training-with-tick-tock-scheduling.md)] \[[pdf](https://proceedings.mlsys.org/paper/2021/file/c81e728d9d4c2f636f067f89cc14862c-Paper.pdf)] |
| Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning                    | OSDI '21  | \[[notes](pollux-co-adaptive-cluster-scheduling-for-goodput-optimized-deep-learning.md)] \[[pdf](https://www.pdl.cmu.edu/PDL-FTP/CloudComputing/osdi21-pollux.pdf)]        |
| MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters | NSDI '22  | \[notes] \[[pdf](https://qzweng.github.io/files/2022NSDI-MLaaS-Weng.pdf)]                                                                                                  |
|                                                                                               |           | \[notes] \[pdf]                                                                                                                                                            |



### Video Analytics & Streaming



| Title                                                                                        | Venue       | Links                                                                                                                                                                           |
| -------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PCC Allegro: Re-architecting Congestion Control for Consistent High Performance              | NSDI '15    | \[notes] \[pdf]                                                                                                                                                                 |
| Pensieve: Neural Adaptive Video Streaming with Pensieve                                      | SIGCOMM '17 | \[notes] \[pdf]                                                                                                                                                                 |
| Chameleon: Scalable Adaptation of Video Analytics via Temporal and Cross-camera Correlations | SIGCOMM '18 | \[notes] \[pdf]                                                                                                                                                                 |
| PCC Vivace: Online-Learning Congestion Control                                               | NSDI '18    | \[notes] \[pdf]                                                                                                                                                                 |
| Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics                | SIGCOMM '20 | \[[notes](reducto-on-camera-filtering-for-resource-efficient-real-time-video-analytics.md)] \[[pdf](https://www.cs.princeton.edu/\~ravian/publications/reducto\_sigcomm20.pdf)] |
| DDS: Server-Driven Video Streaming for Deep Learning Inference                               | SIGCOMM '20 | \[notes] \[pdf]                                                                                                                                                                 |
| PCC Proteus: Scavenger Transport And Beyond                                                  | SIGCOMM '20 | \[notes] \[pdf]                                                                                                                                                                 |
| Learning in situ: a randomized experiment in video streaming                                 | NSDI '20    | \[notes] \[pdf]                                                                                                                                                                 |
| Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers                  | NSDI '22    | \[notes] \[pdf]                                                                                                                                                                 |
| GEMEL: Model Merging for Memory-Efficient, Real-Time Video Analytics at the Edge             | arXiv '22   | \[notes] \[pdf]                                                                                                                                                                 |
|                                                                                              |             | \[notes] \[pdf]                                                                                                                                                                 |





| Title | Venue | Links           |
| ----- | ----- | --------------- |
|       |       | \[notes] \[pdf] |
|       |       | \[notes] \[pdf] |
|       |       | \[notes] \[pdf] |

