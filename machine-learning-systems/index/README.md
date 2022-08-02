# Big Data Systems - Index

## Table of Contents

### Infrastructure, Frameworks, and Paradigms

* [NFS: Sun's Network File System](../../earlier-readings-and-notes/index/nfs-suns-network-file-system.md)
* [\[SOSP '03\] The Google File System](the-google-file-system.md)
* [\[OSDI '04\] MapReduce: Simplified Data Processing on Large Clusters](mapreduce-simplified-data-processing-on-large-clusters.md)
* \[SOSP '09] FAWN: A Fast Array of Wimpy Nodes
* [\[NSDI '11\] Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center](mesos-a-platform-for-fine-grained-resource-sharing-in-the-data-center.md)
* [\[NSDI '12\] Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](resilient-distributed-datasets-a-fault-tolerant-abstraction-for-in-memory-cluster-computing.md)
* [\[HotOS '21\] From Cloud Computing to Sky Computing](from-cloud-computing-to-sky-computing.md)

### Scheduling & Resource Allocation

* \[NSDI '11] Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center
* \[EuroSys '13] Omega: flexible, scalable schedulers for large compute clusters
* \[SoCC '13] Apache Hadoop YARN: Yet Another Resource Negotiator
* \[SoCC '14] Wrangler: Predictable and Faster Jobs using Fewer Resources
* \[SIGCOMM '14] Tetris: Multi-Resource Packing for Cluster Schedulers ([pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tetris\_sigcomm14.pdf))
* \[ASPLOS '14] Quasar: Resource-Efficient and QoS-Aware Cluster Management
* \[SIGCOMM '15] Network-Aware Scheduling for Data-Parallel Jobs: Plan When You Can
* \[OSDI '16] CARBYNE: Altruistic Scheduling in Multi-Resource Clusters ([pdf](https://www.usenix.org/system/files/conference/osdi16/osdi16-grandl-altruistic.pdf))
* \[OSDI '16] Packing and Dependency-aware Scheduling for Data-Parallel Clusters
* \[NSDI '16] HUG: Multi-Resource Fairness for Correlated and Elastic Demands
* \[EuroSys '16] TetriSched: global rescheduling with adaptive plan-ahead in dynamic heterogeneous clusters
* \[SoCC '17] Selecting the best vm across multiple public clouds: A data-driven performance modeling approach
* \[ATC '18] On the diversity of cluster workloads and its impact on research results

### Cloud/Serverless Computing

* \[SoCC '17] Occupy the Cloud: Distributed Computing for the 99%
* \[arXiv '19] Cloud Programming Simplified: A Berkeley View on Serverless Computing
* \[SoCC '19] Centralized Core-granular Scheduling for Serverless Functions
* \[SoCC '19] Cirrus: a Serverless Framework for End-to-end ML Workflows
* \[NSDI '19] Shuffling, Fast and Slow: Scalable Analytics on Serverless Infrastructure
* \[SIGMOD '20] Le Taureau: Deconstructing the Serverless Landscape & A Look Forward
* \[SoCC '20] Serverless linear algebra
* \[ATC '20] Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider
* \[SIGMOD '21] Towards Demystifying Serverless Machine Learning Training
* \[OSDI '21] Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads
* \[SoCC '21] Atoll: A Scalable Low-Latency Serverless Platform
* \[NSDI '21] Caerus: Nimble Task Scheduling for Serverless Analytics
* \[ASPLOS '22] Serverless computing on heterogeneous computers

### Network Flow Scheduling

* \[SIGCOMM '11] Managing Data Transfers in Computer Clusters with Orchestra
* [\[HotNets '12\] Coflow: A Networking Abstraction for Cluster Applications](big-data-systems-papers-short-notes.md#2012-hotnets-coflow-a-networking-abstraction-for-cluster-applications)
* [\[SIGCOMM '14\] Efficient coflow scheduling with Varys](big-data-systems-papers-short-notes.md#2014-sigcomm-efficient-coflow-scheduling-with-varys)
* \[SIGCOMM '14] Barrat: Decentralized task-aware scheduling for data center networks
* [\[SIGCOMM '15\] Aalo: Efficient coflow scheduling without prior knowledge](big-data-systems-papers-short-notes.md#2015-sigcomm-aalo-efficient-coflow-scheduling-without-prior-knowledge)
* \[SIGCOMM '16] CODA: Toward Automatically Identifying and Scheduling COflows in the DArk
* \[SIGCOMM '18] Sincronia: Near-Optimal Network Design for Coflows
* \[SPAA '19] Near Optimal Coflow Scheduling in Networks

### Graphs

* [\[SIGMOD '10\] Pregel: A System for Large-Scale Graph Processing](pregel-a-system-for-large-scale-graph-processing.md)
* [\[OSDI '12\] PowerGraph: Distributed Graph-Parallel Computation on Neural Graphs](powergraph-distributed-graph-parallel-computation-on-natural-graphs.md)
* \[OSDI '14] GraphX: Graph Processing in a Distributed Dataflow Framework
* \[ATC '17] Garaph: Efficient GPU-accelerated Graph Processing on a Single Machine with Balanced Replication
* [\[EuroSys '21\] NextDoor: Accelerating graph sampling for graph machine learning using GPUs](accelerating-graph-sampling-for-graph-machine-learning-using-gpus.md)
* \[OSDI '21] Marius: Learning Massive Graph Embeddings on a Single Machine
* \[arXiv '22] Marius++: Large-Scale Training of Graph Neural Networks on a Single Machine

### Tracing

* \[SOSP '15] Pivot tracing: dynamic causal monitoring for distributed systems
* \[HotNets '21] Snicket: Query-Driven Distributed Tracing
* \[NSDI '23] The Benefit of Hindsight: Tracing Edge-Cases in Distributed Systems

### Caching

* \[SoCC '11] Small Cache, Big Effect: Provable Load Balancing for Randomly Partitioned Cluster Services
* \[NSDI '16] Be Fast, Cheap and in Control with SwitchKV
* \[SOSP '17] NetCache: Balancing Key-Value Stores with Fast In-Network Caching
* [\[FAST '19\] DistCache: Provable Load Balancing for Large-Scale Storage Systems with Distributed Caching](2019-fast-distcache-provable-load-balancing-for-large-scale-storage-systems-with-distributed....md)

### New Data, Hardware Models

* \[ISCA '17] In-Datacenter Performance Analysis of a Tensor Processing Unit

### Databases

* \[SIGMOD '12] Towards a Unified Architecture for in-RDBMS Analytics
* \[arXiv '13] Bayesian Optimization in a Billion Dimensions via Random Embeddings
* \[SIGMOD '17] Automatic Database Management System Tuning Through Large-scale Machine Learning
* \[HotStorage '20] Too Many Knobs to Tune? Towards Faster Database Tuning by Pre-selecting Important Knobs
* \[arXiv '21] Facilitating Database Tuning with Hyper-Parameter Optimization: A Comprehensive Experimental Evaluation
* \[VLDB '22] LlamaTune: Sample-Efficient DBMS Configuration Tuning

## Meta stuff

* Reading lists
  * [CS 294 @ Berkeley: Machine Learning Systems](https://ucbrise.github.io/cs294-ai-sys-fa19/)
  * [CS 744 @ UW-Madison: Big Data Systems](http://pages.cs.wisc.edu/\~shivaram/cs744-fa20/)
  * [CS 6787 @ Cornell: Advanced Machine Learning Systems](https://www.cs.cornell.edu/courses/cs6787/2020fa/), with a focus on the ML side
  * [Awesome-System-for-Machine-Learning](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning): An open-sourced reading list
  * [The MLSys conference](https://mlsys.org/)
  * [SOSP AI Systems workshop](http://learningsys.org/sosp19/acceptedpapers.html)
* Some other stuff
  * Meta papers
    * [A Berkeley View of Systems Challenges for AI](https://thodrek.github.io/CS839\_spring18/papers/EECS-2017-159.pdf)
    * [MLSys: The New Frontier of Machine Learning Systems](https://arxiv.org/pdf/1904.03257.pdf)
  * [Systems Benchmarking Crimes](https://www.cse.unsw.edu.au/\~gernot/benchmarking-crimes.html)
  * [CSE 559W @ U Washington Slides](http://dlsys.cs.washington.edu/schedule): Not a paper reading class, more of an end-to-end comprehensive introduction of foundations of DL Systems
  * [CS 759 @ UW-Madison (HPC) Course Notes](../../earlier-readings-and-notes/cs759-hpc-course-notes/): A great overview of HPC, CUDA, OpenMP, MPI
