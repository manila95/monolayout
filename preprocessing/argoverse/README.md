## Data Preparation(Argoverse Tracking v1.0)


### Weak Supervision
Use your preferred 2D segmentation method(we made use of [inplace-abn](https://github.com/mapillary/inplace_abn)) to generate the results for the corresponding sequence and store it following the same file hierarchy. Pre-generated  segmentation using `inplace_abn` for the entire Argoverse Tracking dataset is provided [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kaustubh_mani_research_iiit_ac_in/ErUnXuCUYkhPhq2LIzj3z_wBrwHAZMGUFuYAz0_hBNfkCQ?e=ll3jq3).

For obtaininig static layouts for entire Argoverse Tracking v1.0 dataset run:

```
python3 generate_weak_suprevision.py --base_path <path to kitti raw root directory> --seg_class road  
```



### GroundTruth

Groundtruth static and dynamic layouts for Argoverse Tracking v1.0 dataset in bird's eye view and can be downloaded from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/kaustubh_mani_research_iiit_ac_in/Ego35MXxh69IqnTHrzPdn9sBNsoWnUdCifM0DzAUgGm6iQ?e=HasUUc). The following script generates static and dynamic groundtruth layouts for Argoverse Tracking dataset using the [argoverse-api](https://github.com/argoai/argoverse-api.git).

```

# Generating Dynamic Layouts for Argoverse Tracking v1.0 dataset (GroundTruth)
./generate_groundtruth.py --base_path <path to Argoverse Tracking root directory> --seg_class vehicle --range 40 --occ_map_size 256 

# Generating Static Layouts for Argoverse Tracking v1.0 dataset (GroundTruth)
./generate_groundtruth.py --base_path <path to Argoverse Tracking root directory> --seg_class road --range 40 --occ_map_size 256

 ```


## Usage

```
usage: generate_argo_groundtruth.py [-h] [--base_path BASE_PATH]
                                    [--out_dir OUT_DIR] [--range RANGE]
                                    [--occ_map_size OCC_MAP_SIZE]
                                    [--seg_class {road,sidewalk,vehicle}]

MonoLayout DataPreparation options

optional arguments:
  -h, --help            show this help message and exit
  --base_path BASE_PATH
                        Path to the root data directory
  --out_dir OUT_DIR     Output directory to save layouts
  --range RANGE         Size of the rectangular grid in metric space
  --occ_map_size OCC_MAP_SIZE
                        Occupancy map size
  --seg_class {road,sidewalk,vehicle}
                        Data Preparation for Road/Sidewalk

```


