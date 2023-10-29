# \[2004 OSDI] MapReduce: Simplified Data Processing on Large Clusters

## One-line Summary

MapReduce is a simple programming model on large clusters with frequent failures. It provides a set of limited but general functional API (Map, Reduce, Sort), fault tolerance, and straggler mitigation through retries.

## Paper Structure Outline

1. Introduction
2. Programming Model
   1. Example
   2. Types
   3. More Examples
3. Implementation
   1. Execution Overview
   2. Master Data Structures
   3. Fault Tolerance
   4. Locality
   5. Task Granularity
   6. Backup Tasks
4. Refinements
   1. Partitioning Function
   2. Ordering Guarantees
   3. Combiner Function
   4. Input and Output Types
   5. Side-effects
   6. Skipping Bad Records
   7. Local Execution
   8. Status Information
   9. Counters
5. Performance
   1. Cluster Configuration
   2. Grep
   3. Sort
   4. Effect of Backup Tasks
   5. Machine Failures
6. Experience
   1. Large-Scale Indexing
7. Related Work
8. Conclusions

## Programming Model

The data type for each record is of the form (key, value).&#x20;

The terms "map" and "reduce" are borrowed from functional languages like Lisp. The Map function (parallelly) processes (a large number of) individual records to generate intermediate (key, value) pairs. The Reduce function (parallelly) processes and merges all intermediate values associated per key by partitioning keys (e.g., hash partitioning).

![Map function & Reduce function](<../../.gitbook/assets/Screen Shot 2021-06-25 at 12.09.59 PM.png>)

## Example Workloads

### Word Count

![](<../../.gitbook/assets/Screen Shot 2021-06-25 at 12.16.47 PM.png>)

### Distributed grep

![](<../../.gitbook/assets/Screen Shot 2021-06-25 at 12.17.12 PM.png>)

### Reversed Web-Link Graph

![](<../../.gitbook/assets/Screen Shot 2021-06-25 at 12.17.36 PM.png>)

### Count of URL Access Frequency

![](<../../.gitbook/assets/Screen Shot 2021-06-25 at 12.18.04 PM.png>)

### Sort

![](<../../.gitbook/assets/Screen Shot 2021-06-25 at 12.18.31 PM.png>)

## MapReduce Scheduling

### Inside MapReduce

For the users, they only need to write the map & reduce programs, then submit the job and wait for the results, without the need to know about the internal parallel/distributed computing. For the paradigm and the scheduler, the following need to be handled:

1. Parallelize Map
2. Transfer data from Map to Reduce: Use partitioning function, ensuring all map output records with the same key are assigned to the same Reduce task
3. Parallelize Reduce
4. Implement storage for Map input, Map output, Reduce input, Reduce output
   1. Map input: From distributed FS (GFS, HDFS, etc.)
   2. Map output: To local FS/disk at Map node
   3. Reduce input: From (multiple) remote disks; Uses local FS
   4. Reduce output: To distributed FS
5. Ensure the barrier between the Map phase and Reduce phase

![](<../../.gitbook/assets/Screen Shot 2021-06-25 at 12.30.26 PM.png>)

### The YARN Scheduler

Yet Another Resource Negotiator (YARN) is offered in Hadoop 2.x+. It treats each server as a collection of containers (some CPU + some memory). It has three main components:

1. Global Resource Manager (RM): Scheduling
2. Per-Server Node Manager (NM): Daemon and Server-specific functions
3. Per-Application/job Application Master (AM): Handles container negotiations with RMs and NMs, detect task failures of that job

![](<../../.gitbook/assets/Screen Shot 2021-06-25 at 12.36.42 PM.png>)

## Other Designs

### Fault Tolerance: Failures

* Server Failure
  * NM heartbeats to RM: If server fails, RM lets all affected AMs know, and AMs take action
  * NM keeps track of each task running at its server: If a task fails while in progress, mark the task as idle and restart it. If the same task fails repeatedly, end the job
  * AM heartbeats to RM: On failure, RM restarts AM, which then syncs up with its running tasks
* RM Failure
  * Use old checkpoints and bring up secondary RM&#x20;

### Fault Tolerance: Stragglers

* The slowest machine slows the entire job down
* Possible reasons: bad disk, network bandwidth, CPU, or memory
* Keep track of the progress of each task (% done). When a straggler appears, launch a second copy of a task on another node and take the output of whichever finishes first (this is called Speculative Execution).

### Locality

* Cloud has hierarchical topology (e.g., racks)
* GFS/HDFS stores 3 replicas of each chunk (e.g., 64 MB in size), possibly on different racks
* MapReduce attempts to schedule a Map task on (preference from high to low):
  * A machine that contains a replica of corresponding input data
  * On the same rack as a machine containing the input
  * Anywhere

## Links

* [Paper PDF](https://static.googleusercontent.com/media/research.google.com/en/archive/mapreduce-osdi04.pdf)
* [Course notes from CS 744 @ UW-Madison](http://pages.cs.wisc.edu/\~shivaram/cs744-fa20-slides/cs744-mapred-notes.pdf)
* [Official MapReduce Tutorial from Apache Hadoop](https://hadoop.apache.org/docs/r1.2.1/mapred\_tutorial.html)
* [Cloud Computing by Prof. Indranil Gupta from UIUC, offered on Coursera](https://www.coursera.org/specializations/cloud-computing)
