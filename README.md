# MonoLayout: Amodal Scene Layout from a single image
#### Kaustubh Mani, Swapnil Daga, Shubhika Garg, N. Sai Shankar, [J. Krishna Murthy](https://krrish94.github.io), and [K. Madhava Krishna](http://robotics.iiit.ac.in)

<p align="center">
![](figs/teaserv5.png)
</p>


## Setting up

Once you've cloned this repo, optionally, setup the requirements for MonoLayout by running
```
python setup.py install
```
inside your favourite `conda` or virtual enviornment.

**NOTE:** We _assume_ that an appropriate version of PyTorch is pre-installed inside the environment. We find the codebase to be working fairly well with most versions of PyTorch (>= 1.1.0).


## Code

#### MonoLayout-Static

```
python3 train.py --type static --split odometry --data_path ./data/odometry/sequences/ 
```



#### MonoLayout-Dynamic

```
python3 train.py --type dynamic --split 3Dobject --data_path ./data/object/training/
```


#### Layout Prediction (Inference)


```
python3 test.py --type static --model_path <path to the model folder> --image_path <path to the image directory>  
```




## Results (KITTI Dataset)

<p align="center">
![](figs/kitti1.gif)
</p>


<p align="center">
![](figs/kitti_final.gif)
</p>


## Results (Argoverse Dataset)

<p align="center">
![](figs/argo_2.gif)
</p>

<p align="center">
![](figs/argo_1.gif)
</p>
