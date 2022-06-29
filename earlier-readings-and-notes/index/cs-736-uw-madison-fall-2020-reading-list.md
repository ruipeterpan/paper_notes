---
description: >-
  Imported from https://canvas.wisc.edu/courses/205576/pages/paper-list. The
  reading list was put together by Prof. Andrea Arpaci-Dusseau.
---

# CS 736 @ UW-Madison Fall 2020 Reading List

This semester, we are reading many of the paper that the OS community has placed into the SIGOPS [Hall of Fame (Links to an external site.)](https://www.sigops.org/award-hof.html). The SIGOPS Hall of Fame Award was instituted in 2005 to recognize the most influential Operating Systems papers that were published at least ten years in the past.   We've marked those papers on our reading list that are in the Hall of Fame.

[Schedule](https://canvas.wisc.edu/courses/205576/pages/schedule)

## File and Storage Systems&#x20;

1. **Background: Traditional Local File Systems -- FFS and LFS**
   1. &#x20;**FFS -** [**Questions,**](https://canvas.wisc.edu/courses/205576/pages/ffs-questions) **Background:** [**Disk Questions**](https://canvas.wisc.edu/courses/205576/pages/disk-questions)\
      McKusick, M.K., Joy, W.N., Leffler, S.J., and Fabry, R.S. , [**A Fast File System for UNIXLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/ffs.ps) **,** ACM Transactions on Computer Systems, Vol. 2, No. 3, August 1984, pp. 181-197.  SIGOPS Hall of Fame Award&#x20;
   2. &#x20;**LFS -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20LFS?titleize=0)\
      Rosenblum, M. and Ousterhout, J.  [**The Design and Implementation of a Log-Structured File SystemLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/lfs.ps) **,** ACM Transactions on Computer Systems, Vol. 10, No. 1, February 1992, pp. 26-52.  SIGOPS Hall of Fame Award
2. &#x20;**Background: Storage Technology -- RAID**&#x20;
   1. &#x20;**RAID** [**- Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20RAID?titleize=0)\
      Patterson, D., Gibson, G., and Katz, R., [**A Case for Redundant Arrays of Inexpensive Disks (RAID)Links to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/raid.ps) Proceedings of the 1988 ACM SIGMOD Conference on Management of Data, Chicago IL, June 1988.  SIGOPS Hall of Fame Award
   2. &#x20;**RDP (No questions yet)**\
      ****[**Row-Diagonal Parity for Double Disk Failure Correction (Links to an external site.)**](https://www.usenix.org/conference/fast-04/row-diagonal-parity-double-disk-failure-correction)[**, (Links to an external site.)**](https://www.usenix.org/conference/fast-04/row-diagonal-parity-double-disk-failure-correction) Proceedings of USENIX File and Storage Technology (FAST), 2004, FAST Test of Time Award
3. &#x20;**Measurement**
   1. &#x20;**iBench** [**- Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20iBench?titleize=0)__\
      _Tyler Harter, Chris Dragga, Michael Vaughn, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau_ [ **A file is not a file: understanding the I/O behavior of Apple desktop applications  Links to an external site.**](http://research.cs.wisc.edu/wind/Publications/ibench-sosp11.pdf)SOSP '11 Proceedings of the Twenty-Third ACM Symposium on Operating Systems Principles Pages 71-83 SOSP Best Paper, UW-Madison Authors
4. &#x20;**Archival Storage and Deduplication-**[**Questions**](https://canvas.wisc.edu/courses/205576/pages/questions-archival-storage)
   1. &#x20;**SnapMirror** [\
      **SnapMirror: File-System-Based Asynchronous Mirroring for Disaster Recovery,**](broken-reference)2002\
      FAST Test of Time Award
   2. &#x20;**Venti**[\
      **Venti: A New Approach to Archival Storage**,](broken-reference) 2002\
      FAST Test of Time Award
   3. &#x20;**Deduplication**[\
      **Avoiding the Disk Bottleneck in the Data Domain Deduplication File System,**](broken-reference)2008\
      FAST Test of Time Award
5. &#x20;**Caching**
   1. **ARC (No questions yet)**[****\
      **ARC: A Self-Tuning, Low Overhead Replacement Cache,**](broken-reference) 2003 FAST Test of Time Award
6. &#x20;**Crash Consistency**
   1. &#x20;**Alice -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Alice?titleize=0)\
      Thanumalayan Sankaranarayana Pillai, Vijay Chidambaram, Ramnatthan Alagappan, Samer Al-Kiswany, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau, [**All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications** Links to an external site.](http://research.cs.wisc.edu/adsl/Publications/alice-osdi14.pdf)Proceedings of the 11th Symposium on Operating Systems Design and Implementation (OSDI '14) Broomfield, CO, October 2014. UW-Madison Authors
   2. &#x20;**OptFS -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20OptFS?titleize=0)\
      Vijay Chidambaram, Thanumalayan Sanakaranarayana Pillai, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau [**Optimistic Crash Consistency Links to an external site.**](http://research.cs.wisc.edu/adsl/Publications/optfs-sosp13.pdf)Symposium on Operating System Principles, SOSP 2013 , UW-Madison Authors
7. &#x20;**SSDs  and  Key-Value  Stores**
   1. &#x20;[**Unwritten SSD Contract**\
      _Jun He_Links to an external site.](http://www.cs.wisc.edu/\~jhe/)_,_ [_Sudarsun KannanLinks to an external site._](http://www.cs.wisc.edu/\~sudarsun/)_,_ [_Andrea C. Arpaci-DusseauLinks to an external site._](http://www.cs.wisc.edu/\~dusseau/)_,_ [_Remzi H. Arpaci-DusseauLinks to an external site._](http://www.cs.wisc.edu/\~remzi/)  [**The Unwritten Contract of Solid State Drives** Links to an external site.](http://research.cs.wisc.edu/adsl/Publications/eurosys17-he.pdf)Proceedings of the 20th European Conference on Computer Systems (EuroSys '17) Belgrade, Serbia, April 2017. UW-Madison Authors
   2. &#x20;**Bourbon (preprint in hotcrp)**\
      _Yifan Dai, Yien Xu, Aishwarya Ganesan, Ramnatthan Alagappan, Brian Kroth, Andrea Arpaci-Dusseau, and Remzi Arpaci-Dusseau_. From WiscKey to Bourbon: A Learned Index for Log-Structured Merge Trees. _In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI’20), October 2020._  UW-Madison Authors
      1. **Optional Background: WiscKey -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/questions-wisckey)[\
         _Lanyue Lu_Links to an external site.](http://www.cs.wisc.edu/\~ll/)_,_ [_Thanumalayan Sankaranarayana PillaiLinks to an external site._](http://www.cs.wisc.edu/\~madthanu/)_,_ [_Andrea C. Arpaci-DusseauLinks to an external site._](http://www.cs.wisc.edu/\~dusseau/)_,_[_Remzi H. Arpaci-DusseauLinks to an external site._](http://www.cs.wisc.edu/\~remzi/) **** [**WiscKey: Separating Keys from Values in SSD-conscious Storage** Links to an external site.](http://research.cs.wisc.edu/adsl/Publications/wisckey-fast16.pdf)Proceedings of the 14th USENIX Conference on File and Storage Technologies (FAST '16) UW-Madison Authors
8. &#x20;**Persistent Memory**
   1. &#x20;**Mnemosyne**\
      _Haris Volos, Andres Jaan Tack, Michael M. Swift._ [_Mnemosyne: Lightweight Persistent MemoryLinks to an external site._](https://pages.cs.wisc.edu/\~swift/papers/asplos11\_mnemosyne.pdf)_, ASPLOS '11: Proceedings of the 16th International Conference on Architectural, UW-Madison Authors_
   2. &#x20;**Level Hash -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/index-for-persistent-memory)\
      _Pengfei Zuo, Yu Hua, and Jie Wu_, [**Write-Optimized and High-Performance Hashing Index Scheme for Persistent Memory  (Links to an external site.)**](https://www.usenix.org/conference/osdi18/presentation/zuo)_Huazhong University of Science and Technology, OSDI'18_
9. &#x20;Graph Processing - Don't read
   1. [**Links to an external site.**](http://www.cs.wisc.edu/\~jhe/)**GraphChi -** [**Questions for all 3 papers**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Graph?titleize=0)\
      _Aapo Kyrola and Guy Blelloch and Carlos Guestrin_ [**GraphChi: Large-Scale Graph Computation on Just a PC.**  (Links to an external site.)](https://www.usenix.org/system/files/conference/osdi12/osdi12-final-126.pdf)USENIX Symposium on Operating Systems Design and Implementation (OSDI'12).
   2. **Xstream**\
      _Amitabha Roy, Ivo Mihailovic, Willy Zwaenepoel_ [**Xstream: Edge-centric graph processing using streaming partitions.   (Links to an external site.)**](https://infoscience.epfl.ch/record/188535/files/paper.pdf)Symposium on Operating Systems Principles (2013).
   3. &#x20;**FlashGraph**\
      _Da Zheng and Disa Mhembere and Randal Burns and Joshua Vogelstein and Carey E. Priebe and Alexander S. Szalay,_ [**FlashGraph: Processing Billion-Node Graphs on an Array of Commodity SSDs,  (Links to an external site.)**](https://www.usenix.org/system/files/conference/fast15/fast15-paper-zheng.pdf)Conference on File and Storage Technologies (FAST 2015)

## Process Synchronization and Scalability

1. &#x20;**Background: Monitors, Theory and Practice-** [**Questions: Monitors**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Monitors?titleize=0) ****&#x20;
   1. &#x20;**Monitors**\
      C.A.R. Hoare  [**Monitors: An Operating System Structuring Concept Links to an external site.**](http://pages.cs.wisc.edu/\~remzi/Classes/736/Fall2010/Papers/hoare-monitors.pdf)Communications of the ACM 17, 10, October 1974, pp. 549-557 &#x20;
   2. &#x20;**Mesa**\
      Butler W. Lampson, David D. Redell [**Experiences with Processes and Monitors in Mesa Links to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/mesa.ps)Communications of the ACM, 23 2, February 1980, pp. 105-117.  SIGOPS Hall of Fame Award
2. &#x20;**OS Scalability: Measurement and Redesign**
   1. [  (Links to an external site.)](http://doi.acm.org/10.1145/1368506.1368525)**Measurement -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Scalability%20Measurement?titleize=0)\
      Silas Boyd-Wickizer, Austin T. Clements, Yandong Mao, Aleksey Pesterev, M. Frans Kaashoek, Robert Morris, and Nickolai Zeldovich  [**An Analysis of Linux Scalability to Many Cores (Links to an external site.)**](http://people.csail.mit.edu/nickolai/papers/boyd-wickizer-scaling.pdf) In Proceedings of the 9th Symposium on Operating Systems Design and Implementation (OSDI), Vancouver, Canada, October 2010
   2. &#x20;**Scalable Commutativity -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Commutativity?titleize=0)\
      Austin T. Clements, M. Frans Kaashoek, Nickolai Zeldovich, Robert T. Morris, and Eddie Kohler [**The Scalable Commutativity Rule: Designing Scalable Software for Multicore Processors. (Links to an external site.)**](http://people.csail.mit.edu/nickolai/papers/clements-sc.pdf) In Proceedings of the 24th ACM Symposium on Operating Systems Principles (SOSP), Farmington, PA, November 2013.
3. &#x20;**Alternate Locking Primitives**
   1. &#x20;**Delegation -** [**Questions: Delegation**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Delegation?titleize=0)\
      Jean-Pierre Lozi and Florian David and Gael Thomas and Julia Lawall and Gilles Muller, [**Remote Core Locking: Migrating Critical-Section Execution to Improve the Performance of Multithreaded** **Applications,**  (Links to an external site.)](https://www.usenix.org/system/files/conference/atc12/atc12-final237.pdf)USENIX Annual Technical Conference (ATC'12), 2012. &#x20;
   2. &#x20;**Shuffle Locks**\
      ****Sanidhya Kashyap,  Irina Calciu, Xiaohe Cheng, Changwoo Min, Taesoo Kim, [**Scalable and Practical Locking with Shufflin** (Links to an external site.)](https://dl.acm.org/doi/10.1145/3341301.3359629)[**g,** ](broken-reference) SOSP'19

## Scheduling

1. &#x20;**Background: Threads and Events**
   1. &#x20;**Scheduler Activations -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Scheduler%20Activations?titleize=0)\
      Anderson, T., Bershad, B., Lazowska, E., and Levy, H. [**Scheduler Activations: Effective Kernel Support for the User-Level Management of ParallelismLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/scheduler.pdf) ACM Transactions on Computer Systems, Vol. 10, No. 1, February 1992, pp. 53-79.&#x20;
   2. &#x20;**SEDA**\
      ****Matt Welsh, David Culler, Eric Brewer (UC Berkeley) [**SEDA: An Architecture for Well-Conditioned, Scalable Internet Services (Links to an external site.)**](http://www.sosp.org/2001/papers/welsh.pdf) SOSP'01
2. &#x20;**Background: Local CPU Schedulers and Resource Tracking**
   1. &#x20;**Lottery Scheduling -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Lottery?titleize=0)\
      Waldspurger, C.A. and Weihl, W.E. [**Lottery Scheduling: Flexible Proportional-Share Resource Mangement Links to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/lottery-osdi94.ps)Proceedings of the First Symposium on Operating Systems Design and Implementation, Monterey CA, November 1994, pp. 1-11.
   2. &#x20;**Resource Containers -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Resource%20Containers?titleize=0)\
      Banga, G., Druschel, P,. Mogul, J. [**Resource Containers: A New Facility for Resource Management in Server SystemsLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/rc-osdi99.ps.gz) Proceedings of the Third Symposium on Operating System Design and Implementation (OSDI-III), New Orleans, LA, February, 1999, 45-58. &#x20;
3. &#x20;**Measurement: Linux and System Services**
   1. &#x20;**Linux Scheduler**\
      _Jean-Pierre Lozi (Université de Nice Sophia-Antipolis), Baptiste Lepers (Ecole Polytechnique Fédérale de Lausanne), Justin Funston (University of British Columbia), Fabien Gaud (Coho Data), Vivien Quéma (Grenoble INP / ENSIMAG), Alexandra Fedorova_ [**The Linux Scheduler: A Decade of Wasted Cores.**  (Links to an external site.)](https://dl.acm.org/citation.cfm?doid=2901318.2901326)_Eurosys 2016_
   2. &#x20;**TAM -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20TAM?titleize=0)\
      Suli Yang, Jing Liu, Andrea C. Arpaci-Dusseau, and Remzi H. Arpaci-Dusseau [**Principled Schedulability Analysis for Distributed Storage Systems using Thread Architecture Models**  (Links to an external site.)](https://www.usenix.org/conference/osdi18/presentation/yang)_(OSDI'18) UW-Madison Authors_
      1. &#x20;_Optional Background_\
         __**Split-Level I/O Scheduling**[\
         Links to an external site.](http://www.cs.wisc.edu/\~suli/)_Suli Yang, Tyler Harter, Nishant Agrawal, Salini Selvaraj Kowsalya, Anand Krishnamurthy, Samer Al-Kiswany, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau_ __ [**Split-Level I/O Scheduling** Links to an external site.](http://research.cs.wisc.edu/adsl/Publications/split-sosp15.pdf)Proceedings of the 25th ACM Symposium on Operating Systems Principles (SOSP '15) UW-Madison Authors
4. &#x20;**Current System Scheduling**
   1. &#x20;_**Monotasks -**_ [_**Questions**_](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Monotasks?titleize=0)__\
      _Kay Ousterhout (UC Berkeley); Christopher Canel (Carnegie Mellon University); Sylvia Ratnasamy (UC Berkeley); Scott Shenker_ [_**Monotasks: Architecting for Performance Clarity in Data Analytics Frameworks (Links to an external site.)**_](https://dl.acm.org/authorize?N47256)_**,** SOSP'17_
   2. &#x20;**Arachne -** [**Questions: Arachne**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Arachne?titleize=0)\
      _Henry Qin, Qian Li, Jacqueline Speiser, Peter Kraft, and John Ousterhout,Stanford University,_[  (Links to an external site.)](https://www.usenix.org/conference/osdi18/presentation/qin)[**Arachne: Core-Aware Thread Management  (Links to an external site.)**](https://www.usenix.org/conference/osdi18/presentation/qin)[_OSDI'18_ (Links to an external site.)](https://www.usenix.org/conference/osdi18/presentation/qin)
5. &#x20;_**Current System Scheduling 2**_
   1. &#x20;**Themis**[ **(Links to an external site.)**](http://shivaram.org/publications/themis-nsdi2020.pdf)[__\
      &#x20;_(Links to an external site.)_](http://shivaram.org/publications/themis-nsdi2020.pdf)_Kshiteej Mahajan, Arjun Balasubramanian, Arjun Singhvi, Shivaram Venkataraman, and Aditya Akella, University of Wisconsin-Madison;_Amar Phanishayee,_Microsoft Research;_Shuchi Chawla,_University of Wisconsin-Madison,_ [Themis: Fair and Efficient GPU Cluster Scheduling (Links to an external site.)](http://shivaram.org/publications/themis-nsdi2020.pdf)- NSDI 2020, UW-Madison Authors

## OS Structure and Virtual Machines

1. &#x20;**Background: Layered vs. Extensible Kernels**
   1. &#x20;**THE -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20THE?titleize=0)\
      Edsger W. Dijkstra[**The Structure of the "THE" Multiprogramming SystemLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/theTHE.pdf) Communications of the ACM 11(5), May 1968.  SIGOPS Hall of Fame Award
   2. &#x20;**Nucleus -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Nucleus?titleize=0)\
      Per Brinch Hansen, [**The Nucleus of a Multiprogramming SystemLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/nucleus.pdf) Communications of the ACM 13(4), April 1970 &#x20;
2. &#x20;**Microkernels: Concepts and Measurements**
   1. &#x20;**Exokernel -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Exokernel?titleize=0)\
      Dawson R. Engler, M. Frans Kaashoek, and James O’Toole Jr [**Exokernel: An Operating System Architecture for Application-Level Resource Management (Links to an external site.)**](https://pdos.csail.mit.edu/6.828/2008/readings/engler95exokernel.pdf) SOSP '95 Proceedings of the fifteenth ACM symposium on Operating systems principles
   2. &#x20;**Arrakis**\
      Simon Peter, Jialin Li, Irene Zhang, Dan R. K. Ports, Doug Woos, Arvind Krishnamurthy, and Thomas Anderson, _University of Washington;_ Timothy Roscoe, _ETH Zürich_ [Arrakis: The Operating System is the Control Plane, (Links to an external site.)](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-peter\_simon.pdf) OSDI'14
      1. Optional Background - **Barrelfish**\
         Andrew Baumann, Paul Barham, Pierre-Evariste Dagand, Tim Harris, Rebecca Isaacs, Simon Peter, Timothy Roscoe, Adrian Schüpbach, and Akhilesh Singhania. [**The Multikernel: A new OS architecture for scalable multicore systems** (Links to an external site.)](http://www.barrelfish.org/publications/barrelfish\_sosp09.pdf)[. (Links to an external site.)](http://www.barrelfish.org/publications/barrelfish\_sosp09.pdf) In _Proceedings of the 22nd ACM Symposium on OS Principles_, Big Sky, MT, USA, October 2009
3. &#x20;**Monolithic, Disaggregation, and HLLs**
   1. &#x20;**UNIX**\
      Ritchie, D.M. and Thompson, K. [**The UNIX Time-Sharing SystemLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/unix-cacm.ps.gz) Communications of the ACM, Vol. 17, No. 7, July 1974, pp. 365-375.  SIGOPS Hall of Fame Award
   2. &#x20;**Disaggregation**\
      _Yizhou Shan, Yutong Huang, Yilun Chen, and Yiying Zhang,_ [**LegoOS: A Disseminated, Distributed OS for Hardware Resource Disaggregation,** (Links to an external site.)](https://www.usenix.org/conference/osdi18/presentation/shan)OSDI 2018
   3. &#x20;**HLLs -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Biscuit?titleize=0)****\
      ****_Cody Cutler, M. Frans Kaashoek, and Robert T. Morris, MIT CSAIL,_ [**The benefits and costs of writing a POSIX kernel in a high-level language,** (Links to an external site.)](https://www.usenix.org/conference/osdi18/presentation/cutler)OSDI 2018
4. &#x20;**Virtual Machines**
   1. &#x20;**Disco -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20Disco?titleize=0)\
      Edouard Bugnion, Scott Devine, Mendel Rosenblum. [**Disco: Running Commodity Operating Systems on Scalable MultiprocessorsLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/disco.ps.gz) Proceedings of The Sixteenth Symposium on Operating Systems Principles (October 1997).  SIGOPS Hall of Fame Award
   2. [  (Links to an external site.)](http://doi.acm.org/10.1145/1368506.1368525)**ESX -** [**Questions**](https://canvas.wisc.edu/courses/205576/pages/Questions%3A%20ESX?titleize=0)\
      Carl A. Waldspurger  [**Memory Resource Management in VMware ESX Server  Links to an external site.**](http://pages.cs.wisc.edu/\~remzi/Classes/736/Fall2010/Papers/esx-osdi02.pdf)In Proc. Fifth Symposium on Operating Systems Design and Implementation (OSDI ’02), Dec. 2002  SIGOPS Hall of Fame Award
   3. &#x20;**Revirt**\
      George W. Dunlap, Samuel T. King, Sukru Cinar, Murtaza A. Basrai, and Peter M. Chen.\
      [**ReVirt: Enabling intrusion analysis through virtual-machine logging and replay (Links to an external site.)**](http://doi.acm.org/10.1145/844128.844148)**.**\
      In Proceedings of the 5th Symposium on Operating Systems Design and Implementation (OSDI '02), 2002, 211-224. SIGOPS Hall of Fame Award
      1. &#x20;**Optional Overview**\
         Bugnion, Nief, Tsafir, [**Hardware and Software Support for Virtualization**](https://canvas.wisc.edu/courses/205576/files/13611425/download?wrap=1)  Synthesis Lectures on Computer Architecture

## Testing, Debugging, and Design

1. &#x20;**Profiling and Binary Code**
   1. &#x20;**KernInst**\
      ****_Ariel Tamches and Barton P. Miller,_ "Fine-Grained Dynamic Instrumentation of Commodity Operating System Kernels",3rd Symposium on Operating Systems Design and Implementation (OSDI),New Orleans, Louisiana, February 1999. UW-Madison Authors
      1. &#x20;**Optional**
         1. _Nathan E. Rosenblum, Xiaojin (Jerry) Zhu and Barton P. Miller,_ "Who Wrote This Code? Identifying the Authors of Program Binaries", 2011 European Symposium on Research in Computer Security (ESORICS), Leuven, Belgium, September 2011. UW-Madison Authors
         2. &#x20;[_Xiaozhu Meng (Links to an external site.)_](https://www.researchgate.net/profile/Xiaozhu\_Meng) _and Barton P. Miller,_ Binary Code is Not Easy, International Symposium on Software Testing and Analysis, 2016 UW-Madison Authors
2. &#x20;**Symbolic Execution and Debugging Experience**
   1. &#x20;**KLEE**\
      ****_Cristian Cadar, Daniel Dunbar, and Dawson Engler._ **KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs.** In OSDI’08, SIGOPS Hall of Fame Award
   2. **Debug**\
      Kirk Glerum, Kinshuman Kinshumann, Steve Greenberg, Gabriel Aul, Vince Orgovan, Greg Nichols, David Grant, Gretchen Loihle, and Galen Hunt.[**Debugging in the (Very) Large: Ten Years of Implementation and Experience**](broken-reference)**. In SOSP ’09,** SIGOPS Hall of Fame Award
3. &#x20;**Summary of System Design**
   1. **Hints**\
      Butler Lampson [Hints for Computer System Design (Links to an external site.)](http://portal.acm.org/citation.cfm?doid=800217.806614),  Proceedings of the Ninth ACM Symposium on Operating Systems Principles, pp. 33-48, October 1983, Bretton Woods, NH, USA.  SIGOPS Hall of Fame Award

## Other Relevant SIGOPS Hall of Fame Papers (not covered)

1. Daniel G. Bobrow, Jerry D. Burchfiel, Daniel L. Murphy and Raymond S. Tomlinson.\
   [Tenex, A Paged Time Sharing System for the PDP-10 (Links to an external site.)](http://dl.acm.org/citation.cfm?id=361271) \
   Communications of the ACM 15(3), March 1972. [SIGOPS Hall of Fame Award (Links to an external site.)](http://doi.acm.org/10.1145/1368506.1368525)
2. Daley, R.C., and Dennis, J.B. \
   [**Virtual Memory, Processes, and Sharing in MULTICSLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/multics.pdf) **** \
   Communications of the ACM, Vol. 11, No. 5, May 1968, pp. 306-312. (Multics paper in [SIGOPS Hall of Fame (Links to an external site.)](http://doi.acm.org/10.1145/1368506.1368525))
3. &#x20;R. Rashid and A. Tevanian and M. Young and D. Golub and R. Baron and D. Black and W. Bolosky and J. Chew, \
   [**Machine-Independent Virtual Memory Management for Paged Uniprocessor and Multiprocessor ArchitecturesLinks to an external site.**](http://pages.cs.wisc.edu/\~dusseau/Classes/CS736/Papers/mach-vm.pdf) **** [**SIGOPS Hall of Fame Award (Links to an external site.)**](http://doi.acm.org/10.1145/1368506.1368525)\
   Proceedings of the 2nd International Conference on Architectural Support for Programming Languages and Operating System (ASPLOS), 1987.  (Mach in [SIGOPS Hall of Fame (Links to an external site.)](http://doi.acm.org/10.1145/1368506.1368525))
4. J. Liedtke. \
   [On micro-kernel construction (Links to an external site.)](http://doi.acm.org/10.1145/224056.224075).\
   In Proceedings of the 15th ACM symposium on Operating Systems Principles (SOSP '95), December 1995, 237-250. [SIGOPS Hall of Fame Award (Links to an external site.)](http://doi.acm.org/10.1145/1368506.1368525)
