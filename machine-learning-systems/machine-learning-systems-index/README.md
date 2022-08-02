---
cover: ../../.gitbook/assets/ImageForArticle_487_16476156253715992.webp
coverY: 48.74919820397691
---

# Machine Learning Systems - Index

### Distributed Training & Parallelism Paradigms

* [\[OSDI '14\] Scaling Distributed Machine Learning with the Parameter Server](scaling-distributed-machine-learning-with-the-parameter-server.md)
* \[SoCC '18] Parameter Hub: a Rack-Scale Parameter Server for Distributed Deep Neural Network Training
* [\[OSDI '20\] BytePS: A High Performance and Generic Framework for Distributed DNN Training](byteps-a-high-performance-and-generic-framework-for-distributed-dnn-training.md)
* [\[VLDB '20\] PyTorch Distributed: Experiences on Accelerating Data Parallel Training](pytorch-distributed-experiences-on-accelerating-data-parallel-training.md)
* \[NSDI '23] Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs
* Parallelism Paradigms & Strategies ([Overview by Hugging Face](https://huggingface.co/docs/transformers/v4.16.2/en/parallelism))
  * [\[NIPS '19\] GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism.md)
  * [\[SOSP '19\] PipeDream: Generalized Pipeline Parallelism for DNN Training](pipedream-generalized-pipeline-parallelism-for-dnn-training.md)
  * [\[arXiv '19\] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](mlsys-papers-short-notes.md#2019-arxiv-megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism)
  * \[MLSys '19] Beyond Data and Model Parallelism for Deep Neural Networks
  * [\[SC '20\] ZeRO: memory optimizations toward training trillion parameter models](2019-sc-zero-memory-optimizations-toward-training-trillion-parameter-models.md)
  * \[ATC '20] HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism
  * [\[SC '21\] ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning](2019-sc-zero-memory-optimizations-toward-training-trillion-parameter-models.md#zero-infinity-and-zero-offload)
  * \[SC '21] Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
  * [\[SC '21\] Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines](mlsys-papers-short-notes.md#2021-sc-chimera-efficiently-training-large-scale-neural-networks-with-bidirectional-pipelines)
  * \[ICML '21] Memory-Efficient Pipeline-Parallel DNN Training
  * [\[ATC '21\] ZeRO-Offload: Democratizing Billion-Scale Model Training](2019-sc-zero-memory-optimizations-toward-training-trillion-parameter-models.md#zero-infinity-and-zero-offload)
  * \[PPoPP '21] DAPPLE: A Pipelined Data Parallel Approach for Training Large Models&#x20;
  * \[OSDI '22] Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
  * \[arXiv '22] Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model

### Workload Scheduling, Cluster Resource Management

* [\[NSDI '11\] DRF: Fair Allocation of Multiple Resource Types](dominant-resource-fairness-fair-allocation-of-multiple-resource-types.md) ([pdf](https://www.usenix.org/legacy/events/nsdi11/tech/full\_papers/Ghodsi.pdf))
* [\[OSDI '18\] Gandiva: Introspective Cluster Scheduling for Deep Learning](gandiva-introspective-cluster-scheduling-for-deep-learning.md) ([pdf](https://www.usenix.org/system/files/osdi18-xiao.pdf))
* \[EuroSys '18] Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters ([pdf](https://dl.acm.org/doi/pdf/10.1145/3190508.3190517))
* [\[ATC '19\] Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads](analysis-of-large-scale-multi-tenant-gpu-clusters-for-dnn-training-workloads.md) ([pdf](https://www.usenix.org/system/files/atc19-jeon.pdf))
* [\[NSDI '19\] Tiresias: A GPU Cluster Manager for Distributed Deep Learning](tiresias-a-gpu-cluster-manager-for-distributed-deep-learning.md) ([pdf](https://www.usenix.org/system/files/nsdi19-gu.pdf))
* [\[NSDI '20\] Themis: Fair and Efficient GPU Cluster Scheduling](themis-fair-and-efficient-gpu-cluster-scheduling.md) ([pdf](https://www.usenix.org/system/files/nsdi20-paper-mahajan.pdf))
* [\[MLSys '20\] Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications](2020-sigcomm-reducto-on-camera-filtering-for-resource-efficient-real-time-video-analytics/salus-fine-grained-gpu-sharing-primitives-for-deep-learning-applications.md) ([pdf](https://www.mosharaf.com/wp-content/uploads/salus-mlsys20.pdf))
* [\[OSDI '20\] Gavel: Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads](gavel-heterogeneity-aware-cluster-scheduling-policies-for-deep-learning-workloads.md) ([pdf](https://www.usenix.org/system/files/osdi20-narayanan\_deepak.pdf))
* [\[OSDI '20\] AntMan: Dynamic Scaling on GPU Clusters for Deep Learning](2020-osdi-antman-dynamic-scaling-on-gpu-clusters-for-deep-learning.md) ([pdf](https://www.usenix.org/system/files/osdi20-xiao.pdf))
* \[OSDI '20] HiveD: Sharing a GPU Cluster for Deep Learning with Guarantees ([pdf](https://www.usenix.org/system/files/osdi20-zhao\_hanyu.pdf))
* \[EuroSys '20] Gandiva-Fair: Balancing efficiency and fairness in heterogeneous GPU clusters for deep learning ([pdf](https://dl.acm.org/doi/pdf/10.1145/3342195.3387555))
* \[EuroSys '20] AlloX: Compute Allocation in Hybrid Clusters ([pdf](https://www.mosharaf.com/wp-content/uploads/allox-eurosys20.pdf))
* [\[MLSys '21\] Wavelet: Efficient DNN Training with Tick-Tock Scheduling](wavelet-efficient-dnn-training-with-tick-tock-scheduling.md) ([pdf](https://proceedings.mlsys.org/paper/2021/file/c81e728d9d4c2f636f067f89cc14862c-Paper.pdf))
* [\[OSDI '21\] Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning](pollux-co-adaptive-cluster-scheduling-for-goodput-optimized-deep-learning.md) ([pdf](https://www.usenix.org/system/files/osdi21-qiao.pdf))
* \[ATC '21] Zico: Efficient GPU Memory Sharing for Concurrent DNN Training ([pdf](https://www.usenix.org/system/files/atc21-lim.pdf))
* \[NSDI '21] AFS/CoDDL: Elastic Resource Sharing for Distributed Deep Learning ([pdf](https://www.usenix.org/system/files/nsdi21-hwang.pdf))
* \[NSDI '22] MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters ([pdf](https://www.usenix.org/system/files/nsdi22-paper-weng.pdf))
* [\[OSDI '22\] Synergy: Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters](mlsys-papers-short-notes.md#2022-osdi-looking-beyond-gpus-for-dnn-scheduling-on-multi-tenant-clusters) ([pdf](https://www.usenix.org/system/files/osdi22-mohan.pdf))
* \[SIGCOMM '22] Multi-Resource Interleaving for Deep Learning Training
* \[arXiv '22] Deep Learning Workload Scheduling in GPU Datacenters: Taxonomy, Challenges and Vision ([pdf](https://arxiv.org/pdf/2205.11913.pdf))
* \[NSDI '23] Shockwave: Proactive, Fair, and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning

### Serving/Inference

* \[NSDI '17] Clipper: A Low-Latency Online Prediction Serving System ([pdf](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf))
* \[NIPS '17 MLSys workshop] TensorFlow-Serving: Flexible, High-Performance ML Serving ([pdf](http://learningsys.org/nips17/assets/papers/paper\_1.pdf))
* \[arXiv '18] Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications ([pdf](https://arxiv.org/pdf/1811.09886.pdf))
* [\[NIPS '18\] Dynamic Space-Time Scheduling for GPU Inference](2018-nips-dynamic-space-time-scheduling-for-gpu-inference.md) ([pdf](http://learningsys.org/nips18/assets/papers/102CameraReadySubmissionGPU\_Virtualization%20\(8\).pdf))
* [\[SOSP '19\] Parity Models: Erasure-Coded Resilience for Prediction Serving Systems](2019-sosp-parity-models-erasure-coded-resilience-for-prediction-serving-systems.md) ([pdf](https://www.cs.cmu.edu/\~rvinayak/papers/sosp2019parity-models.pdf))
* \[SOSP '19] Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis ([pdf](https://dl.acm.org/doi/pdf/10.1145/3341301.3359658))
* \[arXiv '19] No DNN left behind: Improving inference in the cloud with Multi-Tenancy ([pdf](https://arxiv.org/pdf/1901.06887.pdf))
* \[ATC '19] MArk: Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving ([pdf](https://www.usenix.org/system/files/atc19-zhang-chengliang.pdf))
* \[SoCC '20] GSLICE: controlled spatial sharing of GPUs for a scalable inference platform ([pdf](https://dl.acm.org/doi/pdf/10.1145/3419111.3421284))
* \[SoCC '20] InferLine: Latency-Aware Provisioning and Scaling for Prediction Serving Pipelines ([pdf](https://dl.acm.org/doi/pdf/10.1145/3419111.3421285))
* \[OSDI '20] Serving DNNs like Clockwork: Performance Predictability from the Bottom Up ([pdf](https://www.usenix.org/system/files/osdi20-gujarati.pdf))
* \[OSDI '20] PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications ([pdf](https://www.usenix.org/system/files/osdi20-bai.pdf))
* \[ATC '21] INFaaS: Automated Model-less Inference Serving ([pdf](https://www.usenix.org/system/files/atc21-romero.pdf))
* [\[EuroMLSys '21\] Interference-Aware Scheduling for Inference Serving](2021-euromlsys-interference-aware-scheduling-for-inference-serving.md)
* \[arXiv '21] Serving DNN Models with Multi-Instance GPUs: A Case of the Reconfigurable Machine Scheduling Problem
* \[arXiv '21] Gati: Accelerating Deep Learning Inference via Learned Caches
* [\[ICML '21\] Boosting the Throughput and Accelerator Utilization of Specialized CNN Inference Beyond Increasing Batch Size](mlsys-papers-short-notes.md#2021-icml-boosting-the-throughput-and-accelerator-utilization-of-specialized-cnn-inference-beyond-in)
* \[ICML '22] DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale
* \[OSDI '22] Achieving μs-scale Preemption for Concurrent GPU-accelerated DNN Inferences
* \[OSDI '22] Orca: A Distributed Serving System for Transformer-Based Generative Models
* \[SIGMOD '22] Serverless Data Science - Are We There Yet? A Case Study of Model Serving

### Optimizing Networks/Communications for ML

* \[ATC '17] Poseidon: An Efficient Communication Architecture for Distributed Deep Learning on GPU Clusters
* [\[MLSys '19\] BlueConnect: Decomposing All-Reduce for Deep Learning on Heterogeneous Network Hierarchy](mlsys-papers-short-notes.md#2019-mlsys-blueconnect-decomposing-all-reduce-for-deep-learning-on-heterogeneous-network-hierarchy)
* [\[MLSys '19\] TicTac: Accelerating Distributed Deep Learning with Communication Scheduling](2019-sosp-bytescheduler-a-generic-communication-scheduler-for-distributed-dnn-training-....md#comparisons-with-p3-and-tictac)
* [\[MLSys '19\] P3: Priority-Based Parameter Propagation for Distributed DNN Training](2019-sosp-bytescheduler-a-generic-communication-scheduler-for-distributed-dnn-training-....md#comparisons-with-p3-and-tictac)
* [\[SOSP '19\] ByteScheduler: A Generic Communication Scheduler for Distributed DNN Training Acceleration](2019-sosp-bytescheduler-a-generic-communication-scheduler-for-distributed-dnn-training-....md)
* [\[NetAI '20\] Is Network the Bottleneck of Distributed Training?](2020-netai-is-network-the-bottleneck-of-distributed-training.md)
* [\[MLSys '20\] Blink: Fast and Generic Collectives for Distributed ML](mlsys-papers-short-notes.md#2020-mlsys-blink-fast-and-generic-collectives-for-distributed-ml)
* \[SoCC '20] Network-accelerated Distributed Machine Learning for Multi-Tenant Settings
* [\[NSDI '21\] SwitchML: Scaling Distributed Machine Learning with In-Network Aggregation](2021-nsdi-switchml-scaling-distributed-machine-learning-with-in-network-aggregation.md)
* \[NSDI '21] ATP: In-network Aggregation for Multi-tenant Learning
* \[MLSys '21] In-network Aggregation for Shared Machine Learning Clusters
* [\[arXiv '21\] Synthesizing Collective Communication Algorithms for Heterogeneous Networks with TACCL](mlsys-papers-short-notes.md#2021-arxiv-synthesizing-collective-communication-algorithms-for-heterogeneous-networks-with-taccl)
* \[arXiv '21] Cloud Collectives: Towards Cloud-aware Collectives for ML Workloads with Rank Reordering
* \[PPoPP '21] Synthesizing Optimal Collective Algorithms
* \[NSDI '22] Accelerating Collective Communication in Data Parallel Training across Deep Learning Frameworks
* Optical Networks for ML
  * \[SIGCOMM '21] SiP-ML: High-Bandwidth Optical Network Interconnects for Machine Learning Training
  * \[SIGCOMM '21 OptSys workshop] IOI: In-network Optical Inference
  * \[OFC '22] Emerging Optical Interconnects for AI Systems
  * \[NSDI '23] TOPOOPT: Optimizing the Network Topology for Distributed DNN Training

### ML for Systems, Video Analytics & Streaming

* [Kuntai Du's overview on video analytics](https://kuntai.notion.site/Video-analytics-literature-review-90947b73637f427da7d8adc82e764c77)
* [CS34702 @ UChi: Machine Learning for Networking and Systems](https://people.cs.uchicago.edu/\~junchenj/34702-fall21/)
* \[SIGCOMM '17] Pensieve: Neural Adaptive Video Streaming with Pensieve
* \[HotNets '17] Congestion-Control Throwdown
* [\[SIGCOMM '18\] Chameleon: Scalable Adaptation of Video Analytics via Temporal and Cross-camera Correlations](2018-sigcomm-chameleon-scalable-adaptation-of-video-analytics-via-temporal-and-cross-camera-....md)
* \[NSDI '18] PCC Vivace: Online-Learning Congestion Control
* [\[SIGCOMM '20\] Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics](2020-sigcomm-reducto-on-camera-filtering-for-resource-efficient-real-time-video-analytics/)
* \[SIGCOMM '20] DDS: Server-Driven Video Streaming for Deep Learning Inference
* \[NSDI '20] Learning in situ: a randomized experiment in video streaming
* \[NSDI '22] Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers
* \[HotMobile '22] Understanding the Potential of Server-Driven Edge Video Analytics
* \[NSDI '23] GEMEL: Model Merging for Memory-Efficient, Real-Time Video Analytics at the Edge

### Tricks and Relaxations in Learning and Systems: Compression, Pruning, Freezing, and many more

* \[NIPS '13] More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server
* \[arXiv '16] Training Deep Nets with Sublinear Memory Cost
* \[ICLR '16] Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding
* \[NIPS '17] Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent
* \[ICLR '18] Mixed precision training
* \[ICLR '19] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
* \[arXiv '21] AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning
* \[PVLDB '21] BAGUA: Scaling up Distributed Learning with System Relaxations
* \[arXiv '22] BagPipe: Accelerating Deep Recommendation Model Training
* \[arXiv '22] Efficient DNN Training with Knowledge-Guided Layer Freezing
* Hongyi Wang's talk: [On the Utility of Gradient Compression in Distributed Training Systems](https://www.youtube.com/watch?v=gprhrinr3I4)
  * \[NIPS '18] ATOMO: Communication-efficient Learning via Atomic Sparsification ([pdf](https://proceedings.neurips.cc/paper/2018/file/33b3214d792caf311e1f00fd22b392c5-Paper.pdf))
  * [\[MLSys '21\] Accordion: Adaptive Gradient Communication via Critical Learning Regime Identification](accordion-adaptive-gradient-communication-via-critical-learning-regime-identification.md) ([pdf](https://proceedings.mlsys.org/paper/2021/file/1d7f7abc18fcb43975065399b0d1e48e-Paper.pdf))
  * \[MLSys '21] Pufferfish: Communication-efficient Models At No Extra Cost ([pdf](https://arxiv.org/pdf/2103.03936.pdf))
  * \[MLSys '22] On the utility of gradient compression in distributed training systems ([pdf](https://proceedings.mlsys.org/paper/2022/file/cedebb6e872f539bef8c3f919874e9d7-Paper.pdf))
  * \[arXiv '22] Cuttlefish: Factorized Model Training without All the Tuning

### Misc: Storage, Hyperparameter Tuning, FL, etc.

* \[NIPS '16 workshop] Federated Learning: Strategies for Improving Communication Efficiency
* \[ICML '18 workshop] Tune: A research platform for distributed model selection and training
* \[MLSys '19] Bandana: Using Non-Volatile Memory for Storing Deep Learning Models
* \[MLSys '19] Towards Federated Learning at Scale: System Design
* \[MLSys '20] A System for Massively Parallel Hyperparameter Tuning
* \[ICLR '20] Federated Learning with Matched Averaging
* \[EuroSys '21] RubberBand: Cloud-based Hyperparameter Tuning
* [\[FAST '21\] CheckFreq: Frequent, Fine-Grained DNN Checkpointing](checkfreq-frequent-fine-grained-dnn-checkpointing.md)
* [\[VLDB '21\] Analyzing and Mitigating Data Stalls in DNN Training](analyzing-and-mitigating-data-stalls-in-dnn-training.md)
* \[MLSys '21] Fluid: Resource-aware Hyperparameter Tuning Engine
* \[OSDI '21] Oort: Efficient Federated Learning via Guided Participant Selection
* \[NSDI '22] Check-N-Run: a Checkpointing System for Training Deep Learning Recommendation Models
* \[ICML '22] FedScale: Benchmarking Model and System Performance of Federated Learning at Scale
* \[HotCarbon '22] Treehouse: A Case For Carbon-Aware Datacenter Software
