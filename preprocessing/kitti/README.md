## Data Preparation(KITTI dataset)


### Weak Supervision
For a given KITTI RAW sequence, use your preferred 2D segmentation method(we made use of [inplace-abn](https://github.com/mapillary/inplace_abn)) to generate the results for the corresponding sequence and store it following the same file hierarchy. Pre-generated 
segmentation using `inplace_abn` is provided [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kaustubh_mani_research_iiit_ac_in/EusRBOyxAOBOgvsXVREo-j8BxLCZtutZNgrwvQxfOZaGOA?e=6MKEjw). Preprocessed weak static layouts for KITTI RAW dataset can be downloaded from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kaustubh_mani_research_iiit_ac_in/Eh7Edp76zt5Pgi2OMe7D5l8B3JKGErrx25P7DNc-enkbRA?e=pdj7u9).
```angular2html
.data/raw/
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
python3 generate_supervision.py --base_path <path to kitti raw root directory> --date <date of recorded KITTI sequence> --sequence <sequence number> --range <range of the rectangular layout (in m)> --occ_map_size <size of the rectangular occupancy map (in px)> --seg_class <road/sidewalk>
```

For eg.


```
python3 generate_supervision.py --base_path ../../data/raw/ --seg_class road --date 2011_09_26 --sequence 0001 --range 40 --occ_map_size 256
```


For obtaininig static layouts for entire KITTI RAW dataset run:

```
python3 generate_suprevision.py --base_path <path to kitti raw root directory> --seg_class road --process all 
```



### GroundTruth

Groundtruth static layouts for KITTI RAW and KITTI Odometry datasets are manually annotated in bird's eye view and can be downloaded from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kaustubh_mani_research_iiit_ac_in/EuDcoyoPbH5KsFnd3DbYgj4BxjU2MdcoUS9a8Md4WJG39Q?e=yRb5dQ). For generating groundtruth as well as training data for KITTI 3Dobject one should organize the data in the following hierarchy. 


```angular2html
data/object/
├── training
│   ├── calib
│   ├── image_2
│   ├── label_2
├── testing
    └── calib



```


```
python3 generate_supervision.py --base_path ./data/object/training/label_2 --seg_class vehicle --range 40 --occ_map_size 256
```


## Usage

```
usage: generate_supervision.py [-h] [--base_path BASE_PATH] [--date DATE]
                               [--sequence SEQUENCE] [--out_dir OUT_DIR]
                               [--range RANGE] [--occ_map_size OCC_MAP_SIZE]
                               [--seg_class {road,sidewalk,vehicle}]
                               [--process {all,one}]

MonoLayout DataPreparation options

optional arguments:
  -h, --help            show this help message and exit
  --base_path BASE_PATH
                        Path to the root data directory
  --date DATE           Corresponding date from the KITTI RAW dataset
  --sequence SEQUENCE   Sequence number corresponding to a particular date
  --out_dir OUT_DIR     Output directory to save layouts
  --range RANGE         Size of the rectangular grid in metric space (in m)
  --occ_map_size OCC_MAP_SIZE
                        Occupancy map size (in pixels)
  --seg_class {road,sidewalk,vehicle}
                        Data Preparation for Road/Sidewalk/Vehicle
  --process {all,one}   Process entire KITTI RAW dataset or one sequence at a
                        time
```


