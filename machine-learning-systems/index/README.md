# Machine Learning Systems - Index

## Meta stuff

* Reading lists
  * [CS 294 @ Berkeley: Machine Learning Systems](https://ucbrise.github.io/cs294-ai-sys-fa19/)
  * [CS 744 @ UW-Madison: Big Data Systems](http://pages.cs.wisc.edu/~shivaram/cs744-fa20/)
  * [CS 6787 @ Cornell: Advanced Machine Learning Systems](https://www.cs.cornell.edu/courses/cs6787/2020fa/), with a focus on the ML side
  * [Awesome-System-for-Machine-Learning](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning): An open-sourced reading list
  * [The MLSys conference](https://mlsys.org/)
  * [SOSP AI Systems workshop](http://learningsys.org/sosp19/acceptedpapers.html)
* Some other stuff
  * Meta papers
    * [A Berkeley View of Systems Challenges for AI](https://thodrek.github.io/CS839_spring18/papers/EECS-2017-159.pdf)
    * [MLSys: The New Frontier of Machine Learning Systems](https://arxiv.org/pdf/1904.03257.pdf)
  * [Systems Benchmarking Crimes](https://www.cse.unsw.edu.au/~gernot/benchmarking-crimes.html)
  * [CSE 559W @ U Washington Slides](http://dlsys.cs.washington.edu/schedule): Not a paper reading class, more of an end-to-end comprehensive introduction of foundations of DL Systems
  * [CS 759 @ UW-Madison \(HPC\) Course Notes](../cs759-hpc-course-notes/): A great overview of HPC, CUDA, OpenMP, MPI

## Table of Contents

### Infrastructure

| Title | Venue |
| :--- | :--- |
| [NFS: Sun's Network File System](../../operating-systems/index/nfs-suns-network-file-system.md) | USENIX '86 |
| The Google File System | SOSP '03 |
| MapReduce: Simplified Data Processing on Large Clusters | OSDI '04 |
| The Hadoop Distributed File System | ??? '07 |
| Spark: Cluster Computing with Working Sets | HotCloud '10 |
| Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics | CIDR '21 |

### Scheduling & Resource Allocation

| Title | Venue |
| :--- | :--- |
| The Power of Choice in Data-aware Cluster Scheduling | OSDI '14 |
| [Gandiva: Introspective Cluster Scheduling for Deep Learning](gandiva-introspective-cluster-scheduling-for-deep-learning.md) | OSDI '18 |
| Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters | EuroSys '18 |
| [Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads](analysis-of-large-scale-multi-tenant-gpu-clusters-for-dnn-training-workloads.md) | ATC '19 |
| [Tiresias: A GPU Cluster Manager for Distributed Deep Learning](tiresias-a-gpu-cluster-manager-for-distributed-deep-learning.md) | NSDI '19 |
| [Themis: Fair and Efficient GPU Cluster Scheduling](themis-fair-and-efficient-gpu-cluster-scheduling.md) | NSDI '20 |
| [Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications](salus-fine-grained-gpu-sharing-primitives-for-deep-learning-applications.md) | MLSys '20 |
| [Gavel: Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads](gavel-heterogeneity-aware-cluster-scheduling-policies-for-deep-learning-workloads.md) | OSDI '20 |
| [Wavelet: Efficient DNN Training with Tick-Tock Scheduling](wavelet-efficient-dnn-training-with-tick-tock-scheduling.md) | MLSys '21 |

### Machine Learning

| Title | Venue |
| :--- | :--- |
| Large Scale Distributed Deep Networks | NeurIPS '12 |
| [Scaling Distributed Machine Learning with the Parameter Server](scaling-distributed-machine-learning-with-the-parameter-server.md) | OSDI '14 |
| TensorFlow: A system for large-scale machine learning | OSDI '16 |
| Clipper: A Low-Latency Online Prediction Serving System | NSDI '17 |
| Ray: A Distributed Framework for Emerging AI Applications | OSDI '18 |
| Horovod: Fast and Easy Distributed Deep Learning in TensorFlow | arXiv '18 |
| Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training | ICLR '18 |
| PyTorch: An Imperative Style, High-Performance Deep Learning Library | NeurIPS '19 |
| [PipeDream: Generalized Pipeline Parallelism for DNN Training](pipedream-generalized-pipeline-parallelism-for-dnn-training.md) | SOSP '19 |
| ByteScheduler: A Generic Communication Scheduler for Distributed DNN Training Acceleration | SOSP '19 |
| [BytePS: A High Performance and Generic Framework for Distributed DNN Training](byteps-a-high-performance-and-generic-framework-for-distributed-dnn-training.md) | OSDI '20 |
| [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](pytorch-distributed-experiences-on-accelerating-data-parallel-training.md) | VLDB '20 |
| [Accordion: Adaptive Gradient Communication via Critical Learning Regime Identification](accordion-adaptive-gradient-communication-via-critical-learning-regime-identification.md) | MLSys '21 |

### SQL Frameworks

| Title | Venue |
| :--- | :--- |
| Spark SQL: Relational Data Processing in Spark | SIGMOD '15 |

### Stream Processing

| Title | Venue |
| :--- | :--- |
| The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing | VLDB '15 |

### Graph Processing

| Title | Venue |
| :--- | :--- |
| PowerGraph: Distributed Graph-Parallel Computation on Neural Graphs | OSDI '12 |

### New Data, Hardware Models

| Title | Venue |
| :--- | :--- |
| In-Datacenter Performance Analysis of a Tensor Processing Unit | ISCA '17 |

## To Read

* [ ] MapReduce: Simplified Data Processing on Large Clusters
* [ ] ByteScheduler: A Generic Communication Scheduler for Distributed DNN Training Acceleration
* [ ] Horovod: Fast and Easy Distributed Deep Learning in TensorFlow
* [ ] Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters
* [ ] TensorFlow: A system for large-scale machine learning
* [ ] In-Datacenter Performance Analysis of a Tensor Processing Unit
* [ ] PipeDream: Generalized Pipeline Parallelism for DNN Training
* [ ] CNTK: Microsoft’s Open-Source Deep-Learning Toolkit
* [ ] Caffe: Convolutional Architecture for Fast Feature Embedding
* [ ] PyTorch: An Imperative Style, High-Performance Deep Learning Library
* [ ] The Hadoop Distributed File System
* [ ] The Power of Choice in Data-aware Cluster Scheduling
* [ ] [Spark: Cluster Computing with Working Sets](https://www.usenix.org/legacy/event/hotcloud10/tech/full_papers/Zaharia.pdf)
* [ ] L[arge Scale Distributed Deep Networks](https://proceedings.neurips.cc/paper/2012/file/6aca97005c68f1206823815f66102863-Paper.pdf)
* [ ] Ray: A Distributed Framework for Emerging AI Applications
* [ ] [On the importance of initialization and momentum in deep learning \(ICML '13\)](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)
* [ ] Serving DNNs like Clockwork: Performance Predictability from the Bottom Up
* [ ] HiveD: Sharing a GPU Cluster for Deep Learning with Guarantees
* [ ] antman: dynamic scaling on gpu clusters for deep learning
* [ ] PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications
* [ ] Allox
* [ ] Fluid
* [ ] DRF
* [ ] GPipe
* [ ] Blink

