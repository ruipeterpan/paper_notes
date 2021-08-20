---
description: >-
  Here are some key words for understanding research papers, classified by
  topics. A lot of the stuff here are shamelessly stolen from the Internet w/o
  proper credits.
---

# Vocabulary List

## Meta vocabulary

* STOA: State-Of-The-Art
* Shepherd: The context for this word is that in the acknowledgment section of some papers, there's usually this sentence: "We thank our shepherd for xxx". During the review of a publication, sometimes the authors will get assigned a contact person a shepherd if their paper is worthy of acceptance with some revisions. The shepherd then communicates with the authors to improve the submission. For more information, see [this question on StackOverFlow](https://academia.stackexchange.com/q/138110) and [these Shepherding Guidelines](https://hillside.net/plop/plop98/shepherding.html).

## Systems

* Monolithic: \(of an organization or system\) large, powerful, and intractably indivisible and uniform.
* Preemption \(in scheduling\): In preemptive scheduling, tasks have priorities, and preemption is the act of suspending a low priority task for a high priority task to go through first before resuming the low priority task.
* Multiplex: It is basically "the doing of multiple things at the same time" or "interleaving of many things".
* Overhead: The expense \(resources like excess or indirect computation time, memory, bandwidth, etc.\) required to perform a task.
* Latency: The amount of time delay, e.g. between a user action and a web application's response.
* Turnaround: The amount of time taken to complete a process or fulfill a request. TODO: Figure out the difference between latency and turnaround :P
* in-place
* Benchmark: To benchmark something typically means running some standard tests on a piece of software to assess its performance.
  * Microbenchmark: A microbenchmark is \(quite literally\) a micro benchmark. In other words, benchmark on a smaller scale. 
* Atomic: Atomic operations are indivisible. If an operation is atomic, then it is guaranteed to either happen or not at all, never partially completed.
* Throughput: The rate of production/data delivery over a channel. Compared to latency, throughput measures the quantity of data transferred within a unit of time, while latency measures the amount of time for some units of data to be transferred.
  * Goodput: Application-level throughput, i.e. the amount of useful information.
* Vanilla \(software\): It is used to refer when a piece of software is not customized from its original form.
* Nightly build \(e.g., a nightly build of PyTorch\): This is a software engineering term that is somewhat in contrast to a stable build. It \(quite literally\) means that every night, everything that is in a large project is built so that the individual developers do not need to spend a lot of time rebuilding the whole project. 
* Pareto efficiency: Pareto efficiency is when an economy has its resources and goods allocated to the maximum level of efficiency, and no change can be made without making someone worse off. This term is borrowed by scheduling frameworks sometimes.
* Utility function: Economics, everywhere I go I see his face...

## Soccer Analytics

* xG \(expected goal\): The traditional soccer analysis works on the statistics of "total shots" and "shots on target". Nowadays, with the help of big data, it is possible to calculate the probability of a shot to convert to a goal based on where and how a shot is taken. For example, a tap-in might have an xG of 0.93, which means a shot like this goes in 93% of the time \(cues Raheem Sterling\). Conversely, a goal like [this](https://twitter.com/TotalMLS/status/853052198372200450?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E853052198372200450%7Ctwgr%5E%7Ctwcon%5Es1_c10&ref_url=https%3A%2F%2Fwww.americansocceranalysis.com%2Fexplanation) by David Villa has an xG of 0.009 \([source](https://www.americansocceranalysis.com/explanation)\), which means a shot like this only goes in once every 111 times it's tried. The factors taken into account when calculating the xG include the type of assist, whether it was a shot taken with the head or foot, the angle and distance of the shot, and whether it was a big chance. The exact methodologies for these calculations are most likely confidential. The best source to check the xGs in a game is [this website](https://understat.com/), IMO. A similar terminology is xA, expected assist.

