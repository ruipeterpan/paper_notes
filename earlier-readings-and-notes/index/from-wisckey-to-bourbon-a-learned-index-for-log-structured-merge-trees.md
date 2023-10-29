# From WiscKey to Bourbon: A Learned Index for Log-Structured Merge Trees

## One-line Summary

Bourbon is a log-structured merge tree that utilizes machine learning to provide fast lookups.

## Paper Structure Outline

1. Introduction
2. Background
   1. LSM and LevelDB
   2. WiscKey
3. Learned Indexes: a Good Match for LSMs?
   1. Learned Indexes: Beneficial Regimes
   2. Learned Indexes with Writes
4. Bourbon Design
   1. Learning the Data
   2. Supporting Variable-size Values
   3. Level vs. File Learning
   4. Cost vs. Benefit Analyzer
      1. Wait Before Learning
      2. To Learn a File or Not
   5. Bourbon: Putting it All Together
5. Evaluation
   1. Which Portions does Bourbon Optimize?
   2. Performance under No Writes
      1. Datasets
      2. Load Orders
      3. Request Distributions
   3. Range Queries
   4. Efficacy of Cost-benefit Analyzer with Writes
   5. Real Macrobenchmarks
      1. YCSB
      2. SOSD
   6. Performance on Fast Storage
   7. Performance with Limited Memory
   8. Error Bound and Space Overheads
6. Related Work
7. Conclusions

## Background & Motivation

The work is based on an existing LSM system, WiscKey, which is significantly faster than LevelDB and RocksDB. The authors analyzed WiscKey and derived five learning guidelines that aid an LSM system to successfully incorporate learned indexes. These guidelines are applied to build Bourbon, a learned-index implementation of WiscKey.

## Design and Implementation

### Five learning guidelines

1. Favor learning files at lower levels (which live longer)
2. Wait before learning a file (very short-lived files in every level due to consecutive compactions to newly generated files)
3. Do not neglect files at higher levels (serve more internal lookups)
4. Be workload- and data-aware (#lookups vary a lot in different scenarios)
5. Do not learn levels for write-heavy workloads (level changes quickly)

### Beneficial regimes

![Read-only experiments on WiscKey when data resides on memory and different storage devices. Works better with faster data access (InMemory > Optane > SATA SSD)](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.18.12 AM.png>)

Learned indexes can only speed up indexing time.

### Bourbon Learning

![](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.23.03 AM.png>)

Bourbon uses piecewise linear regression (PLR) to model the data as it has low overheads during learning and lookups, and the space overhead is small as well. Bourbon can learn individual sstables files (file learning) or entire levels (level learning). Level learning can be beneficial for read-only workloads, while for mixed workloads, level learning performs worse than file learning.

### Cost-benefit analyzer (CBA)

The motivation for an online Cost vs. Benefit Analyzer (CBA) is to filter out short-lived files as they are not worth learning. Doing so wastes resources and has little benefit. CBA uses stats of previous files at the same level.

To filter out short-lived files, Bourbon waits for a time threshold, Twait, before learning a file. The max time to learn a file is \~40ms, thus Bourbon sets Twait to be 50ms. However, learning a long-lived file may not be beneficial (and vice versa). Intuitively, as long as the benefit of the model (B\_model) outweighs the cost of building the model (C\_model), learning a file is profitable.

#### Estimating the cost (C\_model)

If we assume learning happens in the background (using idle cores), then C\_model is 0. Bourbon takes a conservative approach, though, and assumes that the learning threads will interfere and cause some slow down. As a result, we define Cmodel to be equal to T\_build, the time to train the PLR model for a file. As T\_build is linearly proportional to the number of data points in a file, we define T\_build to be the product of (1) the number of data points in the file and (2) the avg time to train a data point (measured offline).

#### Estimating the benefit (B\_model)

Bourbon defines the benefit of learning a file to be:

$$
B_{model} = (T_b - T_m) * N
$$

{% hint style="info" %}
* T\_b: Average time for the lookup in baseline
* T\_m: Average time for the lookup in model paths
* N: Number of lookups the file serves in its lifetime
{% endhint %}

Then, we divide the internal lookups into negative and positive ones as most negative lookups terminate at the filer.

$$
B_{model} = ((T_{n.b} - T_{n.m}) * N_n) + ((T_{p.b} - T_{p.m}) * N_p)
$$

{% hint style="info" %}
* N\_n & N\_p: Number of negative and positive internal lookups, respectively
* T\_(n.b) and T\_(p.b): Time in the baseline path for negative and positive internal lookups, respectively
* T\_(n.m) and T\_(p.m): Model counterparts
{% endhint %}

To estimate the number of lookups (N\_n & N\_p) and the time the lookups take (T\_{n.b} & T\_{p.b}), CBA keeps track of the statistics of files that (1) have lived their lifetimes and (2) are at the same level (as stats vary significantly across levels).

Estimation of the above quantities are done during T\_wait:

* T\_(n.b) and T\_(p.b): DuringTwait, lookups are served in the baseline path. These times are used to estimate Tn.b & Tp.b.
* T\_(n.m) and T\_(p.m): Estimated as the avg of those of all other files at the same level.
* N\_n & N\_p: Same as above but normalized by a factor f (f = s / s’: s is the size of the file, while s’ is the avg size of files at this level). While estimating the above quantities, short-lived files are filtered out.

If C\_model < B\_model, a file will be learned. If multiple files are chosen to be learned at the same time, they are put on a max priority queue so that files that would deliver the most benefit are prioritized.

Possible directions for future improvements include:

* Better estimations of N\_n, N\_p, T\_(n.m) & T\_(p.m)
* Develop a model of T\_build on-line or calculate C\_build such that it is not simply T\_build
* Sort the work queue by some function other than B\_model - C\_model

![Bourbon lookups](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.23.29 AM.png>)

## Evaluation

![](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.25.28 AM.png>)

![Bourbon works better with datasets of fewer segments](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.25.42 AM.png>)

![Bourbon works better with sequential loads than random loads, as random loads cause many negative lookups and Bourbon offers less gain for negative lookups](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.26.16 AM.png>)

![CBA: At lower write rates, learn most files, resulting in best foreground latency. At higher write rates, limited learning, resulting in significantly less learning time and a foreground latency close to always-learn policy. At all write rates: Minimal total CPU time.](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.27.45 AM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.29.02 AM.png>)

![YCSB & SOSD: Read-only gains holds for real benchmarks; Have less gain when write rate is higher; Accelerate reads without affecting writes](<../../.gitbook/assets/Screen Shot 2021-01-04 at 11.29.13 AM.png>)

## New Vocabulary

* [Log-structured merge trees](https://en.wikipedia.org/wiki/Log-structured\_merge-tree)

## Links

* [Paper PDF](https://www.usenix.org/system/files/osdi20-dai\_0.pdf)
* [Presentation video at OSDI '20](https://www.youtube.com/watch?v=EUxEx5hwLXk)
* [Presentation slides at OSDI '20](https://www.usenix.org/sites/default/files/conference/protected-files/osdi20\_slides\_dai.pdf)
* Thanks to Yifan Dai and Yien Xu for the paper review notes!

{% file src="../../.gitbook/assets/ContractSSDs.pptx" %}
Prof. Andrea's course slides on the SSD paper and Bourbon
{% endfile %}
