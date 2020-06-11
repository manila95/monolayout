import argparse
import os
from pathlib import Path

from argoverse.data_loading.argoverse_tracking_loader\
    import ArgoverseTrackingLoader
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

        local_lane_polygon = city_to_egovehicle_se3\
            .inverse().transform_point_cloud(polygon)
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


def generate_weak_static(dataset_dir="", log_id="", output_dir=""):

    argoverse_loader = ArgoverseTrackingLoader(dataset_dir)
    argoverse_data = argoverse_loader.get(log_id)
    camera = argoverse_loader.CAMERA_LIST[7]
    calib = argoverse_data.get_calibration(camera)
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
        if calib_data['camera_data_'][i]['key'] == \
                                    'image_raw_stereo_front_left':
            ind = i
            break
    rotation = np.array(calib_data['camera_data_'][ind]['value']
                        ['vehicle_SE3_camera_']['rotation']['coefficients'])
    translation = np.array(
        calib_data['camera_data_']
                  [ind]['value']['vehicle_SE3_camera_']['translation'])
    sparse_road_bev_loc = os.path.join(output_dir, log_id, 'sparse_road_bev')
    try:
        os.makedirs(sparse_road_bev_loc)
    except BaseException:
        pass
    dense_city_road_pts = []

    for idx in range(len(lidar_timestamps)):

        occupancy_map = np.zeros((256, 256))
        lidar_timestamp = lidar_timestamps[idx]
        try:
            img_loc = argoverse_data.get_image_list_sync(camera=camera)[idx]
        except BaseException:
            continue
        cam_timestamp = int(img_loc.split('.')[0].split('_')[-1])
        segment_loc = os.path.join(
            dataset_dir,
            log_id,
            'segment',
            'stereo_front_left_' +
            str(cam_timestamp) +
            '.png')
        segment_img = cv2.imread(segment_loc, 0)

        pc = argoverse_data.get_lidar(idx)
        uv = calib.project_ego_to_image(pc).T
        idx_ = np.where(np.logical_and.reduce(
                    (uv[0, :] >= 0.0,
                        uv[0, :] < np.shape(segment_img)[1] - 1.0,
                        uv[1, :] >= 0.0,
                        uv[1, :] < np.shape(segment_img)[0] - 1.0,
                        uv[2, :] > 0)))
        idx_ = idx_[0]
        uv1 = uv[:, idx_]
        uv1 = uv1.T

        x = uv1[:, 0].astype(np.int32)
        y = uv1[:, 1].astype(np.int32)
        col = segment_img[y, x]
        filt = np.logical_or(col == 13, np.logical_or(col == 24, col == 24))
        road_uv1 = uv1[filt, :]

        lidar_road_pts = calib.project_image_to_ego(road_uv1)

        cam_road_pts = calib.project_ego_to_cam(lidar_road_pts)
        X = cam_road_pts[:, 0]
        Z = cam_road_pts[:, 2]
        filt2 = np.logical_and(X > -20, X < 20)
        filt2 = np.logical_and(filt2, np.logical_and(Z > 0, Z < 40))
        y_img = (-Z[filt2] / res).astype(np.int32)
        x_img = (X[filt2] / res).astype(np.int32)
        x_img -= int(np.floor(-20 / res))
        y_img += int(np.floor(40 / res))
        occupancy_map[y_img, x_img] = 255
        occ_map_loc = os.path.join(
            sparse_road_bev_loc,
            'stereo_front_left_' +
            str(cam_timestamp) +
            '.png')
        cv2.imwrite(occ_map_loc, occupancy_map)
        pose_fpath = os.path.join(
            dataset_dir,
            log_id,
            "poses",
            "city_SE3_egovehicle_" +
            str(lidar_timestamp) +
            ".json")
        if not Path(pose_fpath).exists():
            print("not a apth")
            continue
        pose_data = read_json_file(pose_fpath)
        rotation = np.array(pose_data["rotation"])
        translation = np.array(pose_data["translation"])
        city_to_egovehicle_se3 = SE3(
            rotation=quat2rotmat(rotation),
            translation=translation)
        sparse_city_road_pts = city_to_egovehicle_se3.transform_point_cloud(
            lidar_road_pts)
        try:
            dense_city_road_pts = np.concatenate(
                (dense_city_road_pts, sparse_city_road_pts), axis=0)
        except BaseException:
            dense_city_road_pts = sparse_city_road_pts

    dense_road_bev_loc = os.path.join(output_dir, log_id, 'dense_road_bev')
    try:
        os.makedirs(dense_road_bev_loc)
    except BaseException:
        pass

    for idx in range(len(lidar_timestamps)):

        occupancy_map = np.zeros((256, 256))
        lidar_timestamp = lidar_timestamps[idx]
        try:
            img_loc = argoverse_data.get_image_list_sync(camera=camera)[idx]
        except BaseException:
            continue
        cam_timestamp = int(img_loc.split('.')[0].split('_')[-1])
        pose_fpath = os.path.join(
            dataset_dir,
            log_id,
            "poses",
            "city_SE3_egovehicle_" +
            str(lidar_timestamp) +
            ".json")
        if not Path(pose_fpath).exists():
            print("not a path")
            continue
        pose_data = read_json_file(pose_fpath)
        rotation = np.array(pose_data["rotation"])
        translation = np.array(pose_data["translation"])
        city_to_egovehicle_se3 = SE3(
            rotation=quat2rotmat(rotation),
            translation=translation)
        current_ego_frame_road_pts =\
            city_to_egovehicle_se3\
            .inverse_transform_point_cloud(dense_city_road_pts)
        current_cam_frame_road_pts = calib.project_ego_to_cam(
            current_ego_frame_road_pts)
        X = current_cam_frame_road_pts[:, 0]
        Z = current_cam_frame_road_pts[:, 2]
        filt2 = np.logical_and(X > -20, X < 20)
        filt2 = np.logical_and(filt2, np.logical_and(Z > 0, Z < 40))
        y_img = (-Z[filt2] / res).astype(np.int32)
        x_img = (X[filt2] / res).astype(np.int32)
        x_img -= int(np.floor(-20 / res))
        y_img += int(np.floor(40 / res)) - 1

        occupancy_map[y_img, x_img] = 255
        occ_map_loc = os.path.join(
            dense_road_bev_loc,
            'stereo_front_left_' +
            str(cam_timestamp) +
            '.png')
        cv2.imwrite(occ_map_loc, occupancy_map)


if __name__ == "__main__":
    args = get_args()
    base_dir = args.base_path
    out_dir = base_dir if args.out_dir == "" else args.out_dir
    folders = os.listdir(base_dir)
    for folder in folders:
        if folder[-3:] != "txt" and\
                ("val" not in folder and "test" not in folder):
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
                generate_weak_static(dataset_dir, sub_folder, output_dir)
