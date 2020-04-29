# MonoLayout: Amodal Scene Layout from a single image
#### Kaustubh Mani, Swapnil Daga, Shubhika Garg, N. Sai Shankar, [J. Krishna Murthy](https://krrish94.github.io), and [K. Madhava Krishna](http://robotics.iiit.ac.in)


![](figs/teaserv5.png)


## Usage

You need to download the KITTI 3Dobject and odometry dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), including left color images and labels corresponding to 3D objects. The generated top-views using our data preparation method can be downloaded from [here](https://www.google.com/url?q=https://drive.google.com/file/d/1KhqsHbruE16BFEiIcvtbzuXxKGMbxogk/view?usp%3Dsharing&sa=D&source=hangouts&ust=1586514007721000&usg=AFQjCNGaBbJtbNyVWhv2Zf7AwKeKz-xBJQ). The data needs to be organized in the following way.



```angular2html
data/
    object/
      training/
          calib/
          image_2/ #left image
          label_2/
          TV_car/
        
      testing/
          calib/
          image_2/

    odometry/
      sequences/
          00/
            image_2/ #left image
            road_dense128/
          01/
            image_2/ #left image
            road_dense128/
          02/
          ...

```


Trained models for static and dynamic version of MonoLayout can be downloaded from [here](https://drive.google.com/drive/folders/10YYjjqS5Qa4N61E9MT2FA5Zxb-X1xhsI?usp=sharing).

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
python3 test.py --type static --model_path <path to the model folder> --image_path <path to the image directory>  --out_dir <path to the output directory> 
```




## Results (KITTI Dataset)


![](figs/kitti1.gif)


![](figs/kitti_final.gif)


## Results (Argoverse Dataset)

![](figs/argo_2.gif)


![](figs/argo_1.gif)


**Code will be released by late March 2020**
