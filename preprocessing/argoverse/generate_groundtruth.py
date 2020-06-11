import argparse
import os
from pathlib import Path

from argoverse.data_loading.argoverse_tracking_loader\
        import ArgoverseTrackingLoader
from argoverse.data_loading.frame_label_accumulator \
        import PerFrameLabelAccumulator
from argoverse.data_loading.synchronization_database \
        import SynchronizationDB
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

import cv2

import numpy as np


IS_OCCLUDED_FLAG = 100
LANE_TANGENT_VECTOR_SCALING = 4


def get_args():
    parser = argparse.ArgumentParser(
        description="MonoLayout DataPreparation options")
    parser.add_argument("--base_path", type=str, default="./kitti_raw",
                        help="Path to the root data directory")
    parser.add_argument("--out_dir", type=str, default='',
                        help="Output directory to save layouts")
    parser.add_argument("--range", type=int, default=40,
                        help="Size of the rectangular grid in metric space")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="Occupancy map size ")
    parser.add_argument(
        "--seg_class",
        type=str,
        choices=[
            "road",
            "sidewalk",
            "vehicle"],
        help="Data Preparation for Road/Sidewalk")

    return parser.parse_args()


args = get_args()

"""
Code to plot track label trajectories on a map, for the tracking benchmark.
"""
res = args.range / float(args.occ_map_size)
size_x = args.occ_map_size
size_y = args.occ_map_size

x_lb = -args.range / 2
x_ub = args.range / 2
y_lb = 0
y_ub = args.range

# resolution: 1 px corresponds to how many m in real world
# We want to display 40m * 40m of real world in the area
step_x = (x_ub - x_lb) / size_x
step_y = (y_ub - y_lb) / size_y


# shape of the image
size = (2056, 2464)


def get_lane_bev(
        req_lane_ids,
        avm,
        city_name,
        city_to_egovehicle_se3,
        egovehicle_SE3_cam,
        top_view,
        res,
        lane_colour):

    for req_lane_id in req_lane_ids:

        polygon = avm.get_lane_segment_polygon(req_lane_id, city_name)

        local_lane_polygon = city_to_egovehicle_se3.inverse().\
            transform_point_cloud(polygon)
        local_lane_polygon = egovehicle_SE3_cam.inverse(
        ).transform_point_cloud(local_lane_polygon)

        X = local_lane_polygon[:, 0]
        Z = local_lane_polygon[:, 2]

        y_img = (-Z / res).astype(np.int32)
        x_img = (X / res).astype(np.int32)
        x_img -= int(np.floor(-20 / res))
        y_img += int(np.floor(40 / res))
        x_img = x_img.reshape(-1, 1).astype(np.int32)
        y_img = y_img.reshape(-1, 1).astype(np.int32)
        pts = np.stack((x_img, y_img), axis=2)
        cv2.drawContours(top_view, [pts], 0, lane_colour, -1)

    return top_view


def generate_vehicle_bev(dataset_dir="", log_id="", output_dir=""):

    argoverse_loader = ArgoverseTrackingLoader(dataset_dir)
    argoverse_data = argoverse_loader.get(log_id)
    camera = argoverse_loader.CAMERA_LIST[7]
    calib = argoverse_data.get_calibration(camera)
    sdb = SynchronizationDB(dataset_dir, collect_single_log_id=log_id)

    calib_path = f"{dataset_dir}/{log_id}/vehicle_calibration_info.json"
    calib_data = read_json_file(calib_path)
    ind = 0
    for i in range(9):
        if calib_data['camera_data_'][i]['key'] == \
                'image_raw_stereo_front_left':
            ind = i
            break
    rotation = np.array(calib_data['camera_data_'][ind]['value']
                        ['vehicle_SE3_camera_']['rotation']['coefficients'])
    translation = np.array(
        calib_data['camera_data_'][ind]['value']['vehicle_SE3_camera_']
        ['translation'])
    egovehicle_SE3_cam = SE3(
        rotation=quat2rotmat(rotation),
        translation=translation)

    if not os.path.exists(os.path.join(output_dir, log_id, "car_bev_gt")):
        os.makedirs(os.path.join(output_dir, log_id, "car_bev_gt"))

    lidar_dir = os.path.join(dataset_dir, log_id, "lidar")
    ply_list = os.listdir(lidar_dir)

    pfa = PerFrameLabelAccumulator(
        dataset_dir,
        dataset_dir,
        "argoverse_bev_viz",
        save=False,
        bboxes_3d=True)
    pfa.accumulate_per_log_data(log_id)
    log_timestamp_dict = pfa.log_timestamp_dict
    for i, ply_name in enumerate(ply_list):
        lidar_timestamp = ply_name.split('.')[0].split('_')[1]
        lidar_timestamp = int(lidar_timestamp)

        cam_timestamp = sdb.get_closest_cam_channel_timestamp(
            lidar_timestamp, "stereo_front_left", str(log_id))
        image_path = os.path.join(
            output_dir,
            str(log_id),
            "car_bev_gt",
            "stereo_front_left_" +
            str(cam_timestamp) +
            ".jpg")
        objects = log_timestamp_dict[log_id][lidar_timestamp]
        top_view = np.zeros((256, 256))

        all_occluded = True
        for frame_rec in objects:
            if frame_rec.occlusion_val != IS_OCCLUDED_FLAG:
                all_occluded = False

        if not all_occluded:
            for i, frame_rec in enumerate(objects):
                bbox_ego_frame = frame_rec.bbox_ego_frame
                uv = calib.project_ego_to_image(bbox_ego_frame).T
                idx_ = np.all(np.logical_and(np.logical_and
                                             (np.logical_and(
                                                 uv[0, :] >= 0.0,
                                                 uv[0, :] < size[1] - 1.0),
                                              np.logical_and(
                                                 uv[1, :] >= 0.0,
                                                 uv[1, :] < size[0] - 1.0)),
                                             uv[2, :] > 0))
                if not idx_:
                    continue
                bbox_cam_fr = egovehicle_SE3_cam.inverse().\
                    transform_point_cloud(bbox_ego_frame)
                X = bbox_cam_fr[:, 0]
                Z = bbox_cam_fr[:, 2]

                if (frame_rec.occlusion_val !=
                        IS_OCCLUDED_FLAG and
                        frame_rec.obj_class_str == "VEHICLE"):
                    y_img = (-Z / res).astype(np.int32)
                    x_img = (X / res).astype(np.int32)
                    x_img -= int(np.floor(-20 / res))
                    y_img += int(np.floor(40 / res))
                    box = np.array([x_img[2], y_img[2]])
                    box = np.vstack((box, [x_img[6], y_img[6]]))
                    box = np.vstack((box, [x_img[7], y_img[7]]))
                    box = np.vstack((box, [x_img[3], y_img[3]]))
                    cv2.drawContours(top_view, [box], 0, 255, -1)

        cv2.imwrite(image_path, top_view)


def generate_road_bev(dataset_dir="", log_id="", output_dir=""):

    argoverse_loader = ArgoverseTrackingLoader(dataset_dir)
    argoverse_data = argoverse_loader.get(log_id)
    city_name = argoverse_data.city_name
    avm = ArgoverseMap()
    sdb = SynchronizationDB(dataset_dir, collect_single_log_id=log_id)
    try:
        path = os.path.join(output_dir, log_id, 'road_gt')
        command = "rm -r " + path
        os.system(command)
    except BaseException:
        pass
    try:
        os.makedirs(os.path.join(output_dir, log_id, 'road_gt'))
    except BaseException:
        pass

    ply_fpath = os.path.join(dataset_dir, log_id, 'lidar')

    ply_locs = []
    for idx, ply_name in enumerate(os.listdir(ply_fpath)):
        ply_loc = np.array([idx, int(ply_name.split('.')[0].split('_')[-1])])
        ply_locs.append(ply_loc)
    ply_locs = np.array(ply_locs)
    lidar_timestamps = sorted(ply_locs[:, 1])

    calib_path = f"{dataset_dir}/{log_id}/vehicle_calibration_info.json"
    calib_data = read_json_file(calib_path)
    ind = 0
    for i in range(9):
        if calib_data['camera_data_'][i]['key'] ==\
                'image_raw_stereo_front_left':
            ind = i
            break
    rotation = np.array(calib_data['camera_data_'][ind]['value']
                        ['vehicle_SE3_camera_']['rotation']['coefficients'])
    translation = np.array(
        calib_data['camera_data_']
                  [ind]['value']['vehicle_SE3_camera_']['translation'])
    egovehicle_SE3_cam = SE3(
        rotation=quat2rotmat(rotation),
        translation=translation)

    for idx in range(len(lidar_timestamps)):
        lidar_timestamp = lidar_timestamps[idx]
        cam_timestamp = sdb.get_closest_cam_channel_timestamp(
            lidar_timestamp, "stereo_front_left", str(log_id))
        occupancy_map = np.zeros((256, 256))
        pose_fpath = os.path.join(
            dataset_dir,
            log_id,
            "poses",
            "city_SE3_egovehicle_" +
            str(lidar_timestamp) +
            ".json")
        if not Path(pose_fpath).exists():
            continue
        pose_data = read_json_file(pose_fpath)
        rotation = np.array(pose_data["rotation"])
        translation = np.array(pose_data["translation"])
        xcenter = translation[0]
        ycenter = translation[1]
        city_to_egovehicle_se3 = SE3(
            rotation=quat2rotmat(rotation),
            translation=translation)
        ego_car_nearby_lane_ids = avm.get_lane_ids_in_xy_bbox(
            xcenter, ycenter, city_name, 50.0)
        occupancy_map = get_lane_bev(
            ego_car_nearby_lane_ids,
            avm,
            city_name,
            city_to_egovehicle_se3,
            egovehicle_SE3_cam,
            occupancy_map,
            res,
            255)
        output_loc = os.path.join(
            output_dir,
            log_id,
            'road_gt',
            'stereo_front_left_' +
            str(cam_timestamp) +
            '.png')
        cv2.imwrite(output_loc, occupancy_map)


if __name__ == "__main__":
    args = get_args()
    base_dir = args.base_path
    out_dir = base_dir if args.out_dir == "" else args.out_dir
    folders = os.listdir(base_dir)
    for folder in folders:
        if folder[-3:] != "txt":
            dataset_dir = os.path.join(base_dir, folder)
            output_dir = os.path.join(out_dir, folder)
            try:
                os.makedirs(output_dir)
            except BaseException:
                pass
            sub_folders = os.listdir(dataset_dir)
            for sub_folder in sub_folders:
                print("Processing sequence: ",
                      os.path.join(dataset_dir, sub_folder))
                if args.seg_class == "vehicle":
                    generate_vehicle_bev(dataset_dir, sub_folder, output_dir)
                elif args.seg_class == "road":
                    generate_road_bev(dataset_dir, sub_folder, output_dir)
