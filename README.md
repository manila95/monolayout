# MonoLayout: Amodal Scene Layout from a single image
#### Kaustubh Mani, Swapnil Daga, Shubhika Garg, N. Sai Shankar, [J. Krishna Murthy](https://krrish94.github.io), and [K. Madhava Krishna](http://robotics.iiit.ac.in)


![](figs/teaserv5.png)


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


![](figs/kitti1.gif)


![](figs/kitti_final.gif)


## Results (Argoverse Dataset)

![](figs/argo_2.gif)


![](figs/argo_1.gif)


**Code will be released by late March 2020**
