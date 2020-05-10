## Data Preparation



### Static Layout (Weak Supervision)
Download KITTI RAW dataset from [here](http://www.cvlibs.net/datasets/kitti/raw_data.php). For a given KITTI RAW sequence, use your preferred 2D segmentation method(we made use of [inplace-abn](https://github.com/mapillary/inplace_abn)) to generate the results for the corresponding sequence and store it following the same file hierarchy. 

```angular2html
./kitti_raw/
└── 2011_09_26
    ├── 2011_09_26_drive_0001_sync
    │   ├── oxts
    │   │   ├── data
    │   │   │   ├── 0000000000.txt
    │   │   │   ├── 0000000001.txt
    │   │   │   ├── 0000000002.txt
    │   │   │   ├── ........
    │   │   ├── dataformat.txt
    │   │   ├── oxts
    │   │   └── timestamps.txt
    │   ├── segmentation
    │   │   ├── 0000000000.png
    │   │   ├── 0000000001.png
    │   │   ├── 0000000002.png
    │   │   ├── ..........
    │   └── velodyne_points
    │       ├── data
    │       │   ├── 0000000000.bin
    │       │   ├── 0000000001.bin
    │       │   ├── ........
    │       ├── timestamps_end.txt
    │       ├── timestamps_start.txt
    │       └── timestamps.txt
    ├── calib_cam_to_cam.txt
    ├── calib_imu_to_velo.txt
    └── calib_velo_to_cam.txt
```



```
python3 dataprep.py --base_path <path to kitti raw root directory> --date <date of recorded KITTI sequence> --sequence <sequence number> --range <range of the rectangular layout (in m)> --occ_map_size <size of the rectangular occupancy map (in px)> --seg_class <road/sidewalk>
```

For eg.


```
python3 dataprep.py --base_path ./kitti_raw --seg_class road --date 2011_09_26 --sequence 0001 --range 40 --occ_map_size 128 
```



### Dynamic Layout 

Download KITTI 3D Object dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Organize folders in the following way.



```angular2html
data/
├── training
│   ├── calib
│   ├── image_2
│   ├── label_2
├── testing
    └── calib



```


```
python3 dataprep.py --base_path ./data/object/training/label_2 --seg_class vehicle --range 40 --occ_map_size 128
```





