# Data Domain: Avoiding the Disk Bottleneck in the Data Domain Deduplication File System

## One-line Summary

> If you understand this, you get the basic idea. But as with all complicated systems, the devil is in the details.

## Paper Structure Outline

1. Introduction
2. Challenges and Observations
   1. Variable vs. Fixed Length Segments
   2. Segment Size
   3. Performance-Capacity Balance
   4. Fingerprint vs. Byte Comparisons
3. Deduplication Storage System Architecture
   1. Content Store
   2. Segment Store
   3. Container Manager
4. Acceleration Methods
   1. Summary Vector
   2. Stream-Informed Segment Layout
   3. Locality Preserved Caching
   4. Accelerated Segment Filtering
5. Experimental Results
   1. Results with Real World Data
   2. I/O Savings with Summary Vector and Locality Preserved Caching
   3. Throughput
   4. Discussion
6. Related Work
7. Conclusions

## Background & Motivation

## Design and Implementation

## Evaluation

## New Vocabulary

* 
## Links

* [Paper PDF](https://www.usenix.org/legacy/events/fast08/tech/full_papers/zhu/zhu.pdf)
* [Presentation Video at FAST '08](https://www.usenix.org/conference/fast-08/avoiding-disk-bottleneck-data-domain-deduplication-file-system)
* [Presentation Audio at FAST '08](https://c59951.ssl.cf2.rackcdn.com/legacy_media/fast08/tech/full_papers/zhu/zhu.mp3)

