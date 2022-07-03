---
description: ...Correlations
---

# \[2018 SIGCOMM] Chameleon: Scalable Adaptation of Video Analytics via Temporal and Cross-camera ...

## Summary

Chameleon is a video analytics system that optimizes the tradeoff between resource consumption and accuracy by continuously adapting an application's configurations in real time.

## Background & Motivation

Video analytics pipelines consist of several video processing modules (e.g., count vehicles: decoder -> resize & sample frames -> object detection), each of which has a few configuration knobs (e.g., frame resolution, frame sampling rate, the model used for object detection) that collectively determine both the resource consumption and accuracy of the video analytics application. Our target objective is thus to strike the best tradeoff between resources and accuracy.

The thing is, the best configuration for a video analytics pipeline may vary over time. For example, when there is congestion going on and cars are moving slowly, a lower frame sampling rate saves a huge amount of resources without hurting the accuracy much. This kind of optimization technique is a constant theme in system research, IMO -- for example, see the bit on pixel-level frame differencing in [Reducto](2020-sigcomm-reducto-on-camera-filtering-for-resource-efficient-real-time-video-analytics/#background-and-motivation). Anyway, the challenge comes down to how we can continuously adapt to different configurations.

## Design & Implementation

A straw man approach is to periodically profile the configurations, but this is super expensive, because (1) the configuration search space is exponential in size, and (2) executing certain candidate configurations may be orders of magnitude more costly than executing the optimal one. How can we reduce the resource cost of periodic configuration profiling?

![Chameleon's periodic reprofiling pipeline](<../../.gitbook/assets/Screen Shot 2022-03-04 at 8.57.41 PM.png>)

Due to the non-stationary setting of video analytics applications, traditional modeling approaches like Bayesian optimization are similarly expensive. To address the problem, the authors exploited the domain-specific characteristics of the configurations, namely the temporal and spatial correlations.

* Temporal correlation (good configurations are always good): Although the best configuration varies over time, the top-k best configurations are relatively stable over time, and vice versa. This allows the search space to be significantly pruned.
* Cross-camera correlation (cameras near each other are similar): For example, if there is congestion on the highway, two cameras on that same highway share the same properties (e.g., the velocities and sizes of objects) that affect the optimal configuration. This allows the profiling cost to be amortized across multiple cameras.
* Independence of configuration knobs: An expensive, golden configuration is used to establish the ground truth. To reduce the cost of running this golden config, the authors rely on an empirical observation that (unlike say in DBMS tuning) the knobs are typically independent.

![](<../../.gitbook/assets/Screen Shot 2022-03-04 at 9.00.08 PM.png>)

## Evaluation

![Chameleon good!](<../../.gitbook/assets/Screen Shot 2022-03-04 at 9.00.37 PM.png>)

![Contribution breakdown](<../../.gitbook/assets/Screen Shot 2022-03-04 at 8.56.23 PM.png>)

## Links & References

* [Paper PDF](https://people.cs.uchicago.edu/\~junchenj/docs/Chameleon\_SIGCOMM\_CameraReady\_faceblurred.pdf)
* [Presentation slides at SIGCOMM '18](https://conferences.sigcomm.org/sigcomm/2018/files/slides/paper\_5.2.pptx)
