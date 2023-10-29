# Resource Containers: A New Facility for Resource Management in Server Systems

## One-line Summary

A resource container is an operating systems abstraction that separates the notion of a protection domain from a resource principal. It allows for fine-grained resource management.

## Paper Structure Outline

1. Introduction
2. Typical models for high-performance servers
3. Shortcomings of current resource management models
   1. The distinction between scheduling entities and activities
   2. Integrating network processing with resource management
   3. Consequences of misidentified resource principals
4. A new model for resource management
   1. Resource containers
   2. Containers, processes, and threads
   3. Resource containers and CPU scheduling
   4. Other resources
   5. The resource container hierarchy
   6. Operations on resource containers
   7. Kernel execution model
   8. The use of resource containers
5. Performance
   1. Prototype implementation
   2. Experimental environment
   3. Baseline throughput
   4. Costs of new primitives
   5. Prioritized handling of clients
   6. Controlling resource usage of CGI processing
   7. Immunity against SYN-flooding
   8. Isolation of virtual servers
6. Related Work
7. Conclusion

## Background & Motivation

In resource management, current systems do not separate the notion of "protection domain" (where the accounting of the tasks is done) and "resource principal" (where actual work gets performed). Processes and threads are both resource principals as well as protection domains. Also, an application does not have control over how much resources the kernel consumes on behalf of the application. In this work, the two notions are separated by the new operating systems abstraction, resource containers.

Existing models include (1) process-per connection, (2) single-process event-driven connection, and (3) multi-threaded server.

![](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.14.37 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.14.49 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.15.01 PM.png>)

The authors then presented some analyses in section 3 to analyze the shortcomings of the current models.

![](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.54.07 PM.png>)

![](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.54.16 PM.png>)

## Design and Implementation

![](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.55.11 PM.png>)

Instead of binding threads of application space processes, threads bind to different resource containers (RC), a new abstraction that extends into the kernel and thus has knowledge of which os entities are doing work for which application-level clients. Resource containers logically contain all types of resources (CPU time, sockets, control blocks, network buffers) being used by an application. Containers are also attached with attributes to limit resources (CPU availability), provide scheduling priorities, and network QoS values.

To use resource containers with multiple threads:

* Threads are assigned to RCs for each new connection as soon as identified from request info
* Kernel processing is charged to this container
* We can have different RCs for different QoS policies
* The OS scheduler appropriately schedules threads by priority/resource usage. If the thread consumes more than its fair share of resources, its scheduling priority decays.

To use resource containers with events:

* Threads switch to the desired resource containers as they handle new events
* No need for scheduling by OS
* The associated container will be charged for the processing the thread performs for them
* Resource containers just track accounting
* Applications use the information to determine the order to handle events

## Evaluation

Resource containers allow explicit and fine-grained control over resource consumption at both user-level and kernel-level. They can provide accurate resource accounting, enabling web servers to provide differentiated QoS.

![Restricting resource usage of dynamic requests allows the OS to deliver a certain degree of fairness across containers](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.48.40 PM.png>)

![RCs are effective in the prioritized handling of clients (have better response time) as they prevent work in the kernel](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.48.55 PM.png>)

![Resource utilization can be controlled by isolating static requests being serviced](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.49.02 PM.png>)

![Containers help provide a certain amount of immunity against a DOS attack by SYN flooding.](<../../.gitbook/assets/Screen Shot 2021-01-02 at 11.52.24 PM.png>)

## New Vocabulary

* Resource principals: Entities for which separate resource allocation and accounting are done
* Protection domain: Entities that need to be isolated from each other
* QoS: Quality of Service

## Links

* [Paper PDF](https://www.usenix.org/legacy/publications/library/proceedings/osdi99/full\_papers/banga/banga.pdf)
* [Lecture slides from EECS 443 @ Northwestern University](https://users.cs.northwestern.edu/\~fabianb/classes/eecs-443-w09/lectures/RContainers.pdf)
* [Lecture slides from CS 5204 @ Virginia Polytechnic](http://courses.cs.vt.edu/\~cs5204/fall14-butt/lectures/ResourceContainers.pdf)
* [Discussion panel from CS 736 @ UW-Madison](http://pages.cs.wisc.edu/\~swift/classes/cs736-fa12/blog/2012/10/resource\_containers\_a\_new\_faci.html)

{% file src="../../.gitbook/assets/17-resourcecontainers.pptx" %}
CS 736 course slides on resource containers by Prof. Andrea Arpaci-Dusseau
{% endfile %}
