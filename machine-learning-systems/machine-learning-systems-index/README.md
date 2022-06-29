# Machine Learning Systems - Index

### Distributed Training & Parallelism Paradigms

* [\[OSDI '14\] Scaling Distributed Machine Learning with the Parameter Server](scaling-distributed-machine-learning-with-the-parameter-server.md)
* [\[OSDI '20\] BytePS: A High Performance and Generic Framework for Distributed DNN Training](byteps-a-high-performance-and-generic-framework-for-distributed-dnn-training.md)
* [\[VLDB '20\] PyTorch Distributed: Experiences on Accelerating Data Parallel Training](pytorch-distributed-experiences-on-accelerating-data-parallel-training.md)
* Parallelism Paradigms ([Overview by Hugging Face](https://huggingface.co/docs/transformers/v4.16.2/en/parallelism))
  * [\[NIPS '19\] GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism.md)
  * [\[SOSP '19\] PipeDream: Generalized Pipeline Parallelism for DNN Training](pipedream-generalized-pipeline-parallelism-for-dnn-training.md)
  * \[arXiv '19] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
  * \[MLSys '19] Beyond Data and Model Parallelism for Deep Neural Networks
  * [\[SC '20\] ZeRO: memory optimizations toward training trillion parameter models](zero-memory-optimizations-toward-training-trillion-parameter-models.md)
  * \[SC '21] ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning
  * \[SC '21] Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
  * \[ICML '21] Memory-Efficient Pipeline-Parallel DNN Training
  * \[ATC '21] ZeRO-Offload: Democratizing Billion-Scale Model Training



### Workload Scheduling, Cluster Resource Management

* [\[NSDI '11\] DRF: Fair Allocation of Multiple Resource Types](dominant-resource-fairness-fair-allocation-of-multiple-resource-types.md)
* [\[OSDI '18\] Gandiva: Introspective Cluster Scheduling for Deep Learning](gandiva-introspective-cluster-scheduling-for-deep-learning.md)
* \[EuroSys '18] Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters
* [\[ATC '19\] Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads](analysis-of-large-scale-multi-tenant-gpu-clusters-for-dnn-training-workloads.md)
* [\[NSDI '19\] Tiresias: A GPU Cluster Manager for Distributed Deep Learning](tiresias-a-gpu-cluster-manager-for-distributed-deep-learning.md)
* [\[NSDI '20\] Themis: Fair and Efficient GPU Cluster Scheduling](themis-fair-and-efficient-gpu-cluster-scheduling.md)
* [\[MLSys '20\] Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications](salus-fine-grained-gpu-sharing-primitives-for-deep-learning-applications.md)
* [\[OSDI '20\] Gavel: Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads](gavel-heterogeneity-aware-cluster-scheduling-policies-for-deep-learning-workloads.md)
* \[OSDI '20] AntMan: Dynamic Scaling on GPU Clusters for Deep Learning
* \[OSDI '20] HiveD: Sharing a GPU Cluster for Deep Learning with Guarantees
* \[OSDI '20] PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications
* \[EuroSys '20] Gandiva-Fair: Balancing efficiency and fairness in heterogeneous GPU clusters for deep learning
* \[EuroSys '20] AlloX: Compute Allocation in Hybrid Clusters
* [\[MLSys '21\] Wavelet: Efficient DNN Training with Tick-Tock Scheduling](wavelet-efficient-dnn-training-with-tick-tock-scheduling.md)
* [\[OSDI '21\] Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning](pollux-co-adaptive-cluster-scheduling-for-goodput-optimized-deep-learning.md)
* \[NSDI '21] AFS/CoDDL: Elastic Resource Sharing for Distributed Deep Learning
* \[NSDI '22] MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters
* \[OSDI '22] Synergy: Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters
* \[SIGCOMM '22] Multi-Resource Interleaving for Deep Learning Training
* \[arXiv '22] Deep Learning Workload Scheduling in GPU Datacenters: Taxonomy, Challenges and Vision



### Serving/Inference

* \[NSDI '17] Clipper: A Low-Latency Online Prediction Serving System
* \[arXiv '18] Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications
* \[NIPS '18] Dynamic Space-Time Scheduling for GPU Inference
* \[NIPS SysML workshop '18] Accelerating Deep Learning Workloads through Efficient Multi-Model Execution
* \[SOSP '19] Parity Models: Erasure-Coded Resilience for Prediction Serving Systems
* \[SOSP '19] Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis
* \[SoCC '20] GSLICE: controlled spatial sharing of GPUs for a scalable inference platform
* \[SoCC '20] InferLine: Latency-Aware Provisioning and Scaling for Prediction Serving Pipelines
* \[OSDI '20] Serving DNNs like Clockwork: Performance Predictability from the Bottom Up
* \[ATC '21] INFaaS: Automated Model-less Inference Serving
* \[EuroMLSys '21] Interference-Aware Scheduling for Inference Serving
* \[arXiv '21] Serving DNN Models with Multi-Instance GPUs: A Case of the Reconfigurable Machine Scheduling Problem
* \[ICML '21] Boosting the Throughput and Accelerator Utilization of Specialized CNN Inference Beyond Increasing Batch Size



### Optimizing Networks/Communications for ML

* \[ATC '17] Poseidon: An Efficient Communication Architecture for Distributed Deep Learning on GPU Clusters
* \[MLSys '19] BlueConnect: Decomposing All-Reduce for Deep Learning on Heterogeneous Network Hierarchy
* Communication scheduling
  * \[MLSys '19] TicTac: Accelerating Distributed Deep Learning with Communication Scheduling
  * \[MLSys '19] P3: Priority-Based Parameter Propagation for Distributed DNN Training
  * [\[SOSP '19\] ByteScheduler: A Generic Communication Scheduler for Distributed DNN Training Acceleration](bytescheduler-a-generic-communication-scheduler-for-distributed-dnn-training-acceleration.md)
* [\[NetAI '20\] Is Network the Bottleneck of Distributed Training?](is-network-the-bottleneck-of-distributed-training.md)
* \[MLSys '20] Blink: Fast and Generic Collectives for Distributed ML
* \[SoCC '20] Network-accelerated Distributed Machine Learning for Multi-Tenant Settings
* [\[NSDI '21\] SwitchML: Scaling Distributed Machine Learning with In-Network Aggregation](switchml-scaling-distributed-machine-learning-with-in-network-aggregation.md)
* \[NSDI '21] ATP: In-network Aggregation for Multi-tenant Learning
* \[arXiv '21] Synthesizing Collective Communication Algorithms for Heterogeneous Networks with TACCL
* \[arXiv '21] Cloud Collectives: Towards Cloud-aware Collectives for ML Workloads with Rank Reordering
* \[NSDI '22] Accelerating Collective Communication in Data Parallel Training across Deep Learning Frameworks



### ML for Systems, Video Analytics & Streaming

* [Kuntai Du's overview on video analytics](https://kuntai.notion.site/Video-analytics-literature-review-90947b73637f427da7d8adc82e764c77)
* [CS34702 @ UChi: Machine Learning for Networking and Systems](https://people.cs.uchicago.edu/\~junchenj/34702-fall21/)
* \[SIGCOMM '17] Pensieve: Neural Adaptive Video Streaming with Pensieve
* \[HotNets '17] Congestion-Control Throwdown
* [\[SIGCOMM '18\] Chameleon: Scalable Adaptation of Video Analytics via Temporal and Cross-camera Correlations](chameleon-scalable-adaptation-of-video-analytics-via-temporal-and-cross-camera-correlations.md)
* \[NSDI '18] PCC Vivace: Online-Learning Congestion Control
* [\[SIGCOMM '20\] Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics](reducto-on-camera-filtering-for-resource-efficient-real-time-video-analytics.md)
* \[SIGCOMM '20] DDS: Server-Driven Video Streaming for Deep Learning Inference
* \[NSDI '20] Learning in situ: a randomized experiment in video streaming
* \[NSDI '22] Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers
* \[arXiv '22] GEMEL: Model Merging for Memory-Efficient, Real-Time Video Analytics at the Edge
* \[HotMobile '22] Understanding the Potential of Server-Driven Edge Video Analytics



### Learning/System Tricks/Relaxations: Compression, Pruning, Quantization, and many more

* \[NIPS '13] More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server
* \[ICLR '16] Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding
* \[NIPS '17] Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent
* \[ICLR '18] Mixed precision training
* \[ICLR '19] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
* \[MLSys '21] Accordion: Adaptive Gradient Communication via Critical Learning Regime Identification
* \[MLSys '21] Pufferfish: Communication-efficient Models At No Extra Cost
* \[PVLDB '21] BAGUA: Scaling up Distributed Learning with System Relaxations
* \[MLSys '22] On the utility of gradient compression in distributed training systems



### Misc: Storage, Hyperparameter Tuning, etc.

* \[MLSys '19] Bandana: Using Non-Volatile Memory for Storing Deep Learning Models
* [\[FAST '21\] CheckFreq: Frequent, Fine-Grained DNN Checkpointing](checkfreq-frequent-fine-grained-dnn-checkpointing.md)
* [\[VLDB '21\] Analyzing and Mitigating Data Stalls in DNN Training](analyzing-and-mitigating-data-stalls-in-dnn-training.md)
* \[MLSys '21] Fluid: Resource-aware Hyperparameter Tuning Engine

