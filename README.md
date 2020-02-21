# MonoLayout: Amodal Scene Layout from a single image
#### Kaustubh Mani, Swapnil Daga, Shubhika Garg, N. Sai Shankar, [J. Krishna Murthy](https://krrish94.github.io), and [K. Madhava Krishna](http://robotics.iiit.ac.in)

<p align="center">
    <img src="figs/teaserv5.png" />
</p>

## Abstract

In this paper, we address the novel, highly challenging problem of estimating the layout of a complex urban driving scenario. Given a single color image captured from a driving platform, we aim to predict the bird's-eye view layout of the road and other traffic participants. The estimated layout should reason beyond what is visible in the image, and compensate for the loss of 3D information due to projection. We dub this problem amodal scene layout estimation, which involves "hallucinating" scene layout for even parts of the world that are occluded in the image. To this end, we present MonoLayout, a deep neural network for real-time amodal scene layout estimation from a single image. We represent scene layout as a multi-channel semantic occupancy grid, and leverage adversarial feature learning to hallucinate plausible completions for occluded image parts. Due to the lack of fair baseline methods, we extend several state-of-the-art approaches for road-layout estimation and vehicle occupancy estimation in bird's-eye view to the amodal setup for rigorous evaluation. By leveraging temporal sensor fusion to generate training labels, we significantly outperform current art over a number of datasets. On the KITTI and Argoverse datasets, we outperform all baselines by a significant margin. We also make all our annotations, and code publicly available. A video abstract of this paper is available at https://www.youtube.com/watch?v=HcroGyo6yRQ

## TL;DR

State-of-the-art amodal scene layout from a single image @ 32 fps*

* Benchmarked on an Nvidia GeForce GTX 1080Ti GPU

## Contributions

* We propose MonoLayout, a practically motivated deep architecture to estimate the amodal scene layout from just a single image.
* We demonstrate that adversarial learning can be used to further enhance the quality of the estimated layouts, specifically when hallucinating large missing chunks of a scene.
* We evaluate against several state-of-the-art approaches, and outperform all of them by a significant margin on a number of established benchmarks (KITTI-Raw, KITTI-Object, KITTIOdometry, Argoverse).
* Further, we show that MonoLayout can also be efficiently trained on datasets that do not contain lidar scans by leveraging recent successes in monocular depth estimation.

## Results (KITTI Dataset)


<p align="center">
    <img src="figs/kitti1.gif" />
</p>

<p align="center">
    <img src="figs/kitti_final.gif" />
</p>


## Results (Argoverse Dataset)

<p align="center">
    <img src="figs/argo_2.gif" />
</p>

<p align="center">
    <img src="figs/argo_1.gif" />
</p>

**Code will be released by late March 2020**
