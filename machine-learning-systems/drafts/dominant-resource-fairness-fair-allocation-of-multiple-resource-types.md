# Dominant Resource Fairness: Fair Allocation of Multiple Resource Types

## One-line Summary

DRF is a generalization of max min fairness for multiple resources. Each user has a demand vector (containing multiple resources; for one task of the user) and a dominant resource (xxx-bound) -- calculated using the demand vector and total resources available -- and DRF tries to equalize the dominant share (fraction of the dominant resource user is allocated) for all users.

## Paper Structure Outline

1. Introduction
2. Motivation
3. Allocation Properties
4. Dominant Resource Fairness (DRF)
   1. An Example
   2. DRF Scheduling Algorithm
   3. Weighted DRF
5. Alternative Fair Allocation Policies
   1. Asset Fairness
   2. Competitive Equilibrium from Equal Incomes
   3. Comparison with DRF
6. Analysis
   1. Fairness Properties
      1. Properties Violated by Asset Fairness
      2. Properties Violated by CEEI
      3. Resource Monotonicity vs. Sharing Incentives and Pareto efficiency
   2. Discrete Resource Allocation
7. Experimental Results
   1. Dynamic Resource Sharing
   2. DRF vs. Alternative Allocation Policies
   3. Simulations using Facebook Traces
8. Related Work
9. Conclusion and Future Work
10. Acknowledgments

## Background & Motivation

### Background

* Desired properties for resource allocation policies
  * Sharing incentive: Each user should be better off sharing the cluster, than exclusively using her own partition of the cluster. Consider a cluster with identical nodes and n users. Then a user should not be able to allocate more tasks in a cluster partition consisting of 1/n of all resources.
  * Strategy-proofness: Users should not be able to benefit by lying about their resource demands. This provides incentive compatibility, as a user cannot improve her allocation by lying.
    * The paper included two really interesting examples of users benefitting from lying
  * Pareto efficiency: It should not be possible to increase the allocation of a user without decreasing the allocation of at least another user. This property is important as it leads to maximizing system utilization subject to satisfying the other properties.
  * Envy freeness: A user should not prefer the allocation of another user.
  * Four other nice properties that are nice-to-have
    * Single resource fairness: For a single resource, the solution should reduce to max-min fairness.
    * Bottleneck fairness: If there is one resource that is percent-wise demanded most of by every user, then the solution should reduce to max-min fairness for that resource.
    * Population monotonicity: When a user leaves the system and relinquishes her resources, none of the allocations of the remaining users should decrease.
    * Resource monotonicity: If more resources are added to the system, none of the allocations of the existing users should decrease.
* Existing (fair) allocation policies
  * Equal share: Not work conserving
  * Max min fairness: Maximize the allocation for most poorly-treated users
  * Asset fairness: Equalizes each user's sum of resource shares (violates sharing incentive)
  * (Microeconomic theory) Competitive Equilibrium from Equal Incomes (CEEI): (not strategy-proof)

### Motivation

* Previous work on fair allocation focuses on one single resource type

## Design and Implementation

![Pseudo-code for DRF](<../../.gitbook/assets/Screen Shot 2021-09-30 at 1.10.33 AM.png>)



## Evaluation





## Links & References

* Paper PDF
* Presentation video at xxx
* Presentation slides at xxx
* xxx on GitHub
